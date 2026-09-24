#!/usr/bin/env julia
# Nereus daemon — one warm Julia process, JSON over a unix socket.
#
# WHY A DAEMON, NOT juliacall: embedding Julia in the Python process deadlocks
# under `@threads` (the samplers are multithreaded and the GIL is held for the
# duration of an embedded call). A separate OS process owns a GIL-free Julia
# runtime end to end. It also amortises load: `using Nereus` costs ~20 s even
# with a fully precompiled depot, paid once at boot instead of per job.
#
# WIRE FORMAT: 8-byte big-endian length prefix, then that many bytes of UTF-8
# JSON. Same framing in both directions. A length prefix rather than newline
# framing because payloads legitimately contain newlines.
#
# Request : {"action": "<name>", "payload": {...}}
# Response: {"ok": true, "result": ...} | {"ok": false, "error": "...", "backtrace": "..."}
#
# Usage: julia daemon.jl <socket-path> [<ready-file>]

using Sockets
using JSON3

const READY_SENTINEL = "NEREUS_DAEMON_READY"

function read_msg(io)
    hdr = read(io, 8)
    length(hdr) == 8 || return nothing
    n = Int(ntoh(reinterpret(UInt64, hdr)[1]))
    n == 0 && return nothing
    JSON3.read(String(read(io, n)); allow_inf = true)
end

function write_msg(io, obj)
    body = Vector{UInt8}(JSON3.write(obj; allow_inf = true))
    write(io, reinterpret(UInt8, [hton(UInt64(length(body)))]))
    write(io, body)
    flush(io)
end

# --- actions -----------------------------------------------------------------
# Kept deliberately small here. The real surface is Nereus.run_job plus the
# compute actions; each is a thin adapter so the daemon stays transport-only.

function act_ping(_)
    # `api` is the Nereus<->astronereus contract version. A runtime old enough
    # to predate PY_API_VERSION reports 0, which the client treats as "too old"
    # -- correct, since every bundle without it also predates the 0.3.0
    # priors-only API. Reported from Nereus (the cached bundle), NOT from this
    # file, which ships with the pip package and is therefore always current.
    api = isdefined(Main, :Nereus) && isdefined(Main.Nereus, :PY_API_VERSION) ?
          Int(Main.Nereus.PY_API_VERSION) : 0
    Dict("pong" => true,
         "julia" => string(VERSION),
         "cpu" => Sys.CPU_NAME,
         "threads" => Threads.nthreads(),
         "api" => api,
         "nereus" => isdefined(Main, :Nereus) ?
                     string(pkgversion(Main.Nereus)) : nothing)
end

function act_loaded(_)
    Dict("modules" => sort!(unique(String(nameof(m))
                                    for m in values(Base.loaded_modules))))
end

function act_run_job(payload)
    # Nereus.run_job takes a config path or dict and writes summary.json,
    # chains.nc and the plot tree into cfg.output_dir, returning the summary.
    isdefined(Main, :Nereus) || error("Nereus not loaded in this daemon")
    Base.invokelatest(Main.Nereus.run_job, copy(payload))
end

const DISPATCH = Dict{String,Function}(
    "ping" => act_ping,
    "loaded" => act_loaded,
    "run_job" => act_run_job,
)


# --- fit_* operations ---------------------------------------------------------
# The daemon is transport only: it converts the JSON payload into Julia values
# and calls the public API. Any science decision belongs in api.jl, not here.

"""JSON3 objects/arrays -> plain Julia Dict/Vector, recursively."""
_j(x::JSON3.Object) = Dict{String,Any}(String(k) => _j(v) for (k, v) in pairs(x))
_j(x::JSON3.Array)  = [_j(v) for v in x]
_j(x) = x

"""Per-instrument RV/photometry blobs -> the NamedTuples the API expects."""
function _channel_data(d::AbstractDict)
    out = Dict{String,Any}()
    for (inst, blob) in d
        b = _j(blob)
        nt = NamedTuple(Symbol(k) => (v isa AbstractVector ? Float64.(v) : v)
                        for (k, v) in b)
        out[String(inst)] = nt
    end
    return out
end

function _prepare_channels(payload)
    chs = Any[]
    for c in get(payload, :channels, [])
        d = _j(c)
        haskey(d, "data") && (d["data"] = _channel_data(d["data"]))
        push!(chs, d)
    end
    return chs
end

# `nothing` -- NOT a Dict naming "pt". `nothing` is what makes api.jl pick the
# engine by shape (pt_emcee, or transdim_pt_emcee when `transdim` is present);
# any name here defeats that and, for a trans-dim fit, makes run_engine reject
# the `td` option the named sampler does not declare. The client sends
# `"engine": null` when the user passed none, and JSON null arrives here as
# `nothing`; this default covers a payload with no `engine` key at all.
_engine_of(payload) = _j(get(payload, :engine, nothing))

# A figure manifest is built by scanning `<output_dir>/plots` for PNGs, so it
# reports EVERY png sitting there -- including ones a previous fit into the
# same output_dir left behind. That turned `plots = ["corner"]` into a result
# advertising three figures, two of them stale, with no way for the caller to
# tell which.
#
# The fix is a before/after snapshot, NOT a "newer than now" timestamp test:
# comparing mtimes against `time()` needs slack for filesystems whose mtime is
# coarser than the wall clock, and any slack wide enough to be safe also lets
# through a figure written a second earlier -- which is exactly the fit-then-
# replot case. Comparing each file against its OWN previous mtime makes no
# clock assumption: a file is this call's iff it did not exist before or its
# mtime moved.
function _png_mtimes(proot::AbstractString)
    seen = Dict{String,Float64}()
    isdir(proot) || return seen
    for (root, _, files) in walkdir(proot), f in files
        endswith(f, ".png") || continue
        full = joinpath(root, f)
        try; seen[full] = stat(full).mtime; catch; end
    end
    return seen
end

_is_fresh(full, before) = !haskey(before, full) ||
    (try stat(full).mtime != before[full] catch; true end)

"""Drop entries of an already-built manifest that no call rewrote."""
function _fresh_figs(figs, before::Dict{String,Float64})
    figs isa AbstractDict || return figs
    out = Dict{String,Any}()
    for (k, v) in pairs(figs)
        v isa AbstractString || continue
        _is_fresh(v, before) && (out[String(k)] = v)
    end
    return out
end

function _fit_dispatch(op::String, payload)
    # api.jl/features.jl are Nereus members now; fall back to Main for the
    # bring-up path where they were include()d loose.
    mod = isdefined(Main, :Nereus) && isdefined(Main.Nereus, :fit_rv) ? Main.Nereus :
          isdefined(Main, :fit_rv) ? Main :
          error("Nereus API not available in this daemon")
    chs = _prepare_channels(payload)
    planets = get(payload, :planets, 1)
    eng = _engine_of(payload)
    outdir = get(payload, :output_dir, nothing)
    # Per-parameter priors. Every fit_* takes them through `kwargs...`, but
    # they never crossed the wire, so a Python caller could not bracket a
    # known period or bound a mass -- which a workshop-length run needs to
    # converge at all. `parallax` was already forwarded separately because
    # fit_astrometry declares it explicitly.
    # `_target_from` types this Union{Nothing, Dict{String,<:PriorSpec}} -- it
    # wants PRIOR OBJECTS, not the JSON dicts that arrive on the wire. Shipping
    # the raw dicts through (0.2.11) therefore died with a TypeError on the
    # first fit that used priors at all. `_as_prior` is the same converter
    # `parallax` already went through; empty must become `nothing`, because an
    # empty Dict{String,Any} fails that type bound too.
    priors = let raw = _j(get(payload, :priors, Dict{String,Any}()))
        # Dict{String,Any} does NOT satisfy Dict{String,<:PriorSpec}; the
        # element type has to be the abstract PriorSpec, not Any.
        isempty(raw) ? nothing :
            Dict{String, mod.PriorSpec}(
                String(k) => mod._as_prior(_j(v)) for (k, v) in pairs(raw))
    end
    outdir = outdir === nothing ? nothing : String(outdir)

    # GENERIC KWARG PASSTHROUGH. Everything the client sends that is not part
    # of the transport envelope is forwarded to the fit_* function as a
    # keyword. Before this, the daemon hardcoded `planets/engine/output_dir/
    # priors`, so any other kwarg -- transdim, external_priors, plots,
    # parametrization, stability, sharing, ttv_* -- was silently DROPPED on
    # the way to Julia. The Python side could accept them and nothing would
    # happen, which is worse than not offering them.
    #
    # Symbol-valued options are spelled as strings on the wire (JSON has no
    # Symbol), so they are converted here the same way engine options are.
    _ENVELOPE = (:op, :channels, :planets, :engine, :stopping,
                       :output_dir, :priors, :orbit, :nights)
    _SYMBOL_KW = (:parametrization, :time_anchor, :stability,
                        :ttv_backend, :limb_darkening, :flavour)
    extra = Dict{Symbol,Any}()
    for (k, v) in pairs(payload)
        sk = Symbol(k)
        sk in _ENVELOPE && continue
        v === nothing && continue
        jv = _j(v)
        extra[sk] = (sk in _SYMBOL_KW && jv isa AbstractString) ? Symbol(jv) : jv
    end

    f = getfield(mod, Symbol(op))
    _before = outdir === nothing ? Dict{String,Float64}() :
              _png_mtimes(joinpath(outdir, "plots"))
    r = if op == "fit_tomography"
        # Not a channel fit: the estimator needs the transit geometry
        # (P, a_Rs, inc, vsini, T14) and a list of nights, and it takes no
        # engine because it is a matched filter rather than a sampler.
        ob = _j(get(payload, :orbit, Dict{String,Any}()))
        need(k) = haskey(ob, k) && ob[k] !== nothing ? Float64(ob[k]) :
                  error("fit_tomography: orbit.$k is required")
        nl = Int(get(ob, "n_lambda", 721))
        Base.invokelatest(f, _j(get(payload, :nights, []));
                          P = need("P"), a_Rs = need("a_Rs"),
                          inc = need("inc"), vsini = need("vsini"),
                          T14 = need("T14"),
                          vsys = Float64(get(ob, "vsys", 0.0)),
                          λs = range(-π, π; length = nl),
                          n_null = Int(get(ob, "n_null", 300)),
                          output_dir = outdir)
    elseif op == "fit_joint"
        Base.invokelatest(f, chs...; planets, engine = eng, output_dir = outdir, priors = priors, extra...)
    elseif op == "fit_astrometry"
        c = isempty(chs) ? Dict{String,Any}() : chs[1]
        Base.invokelatest(f; iad = get(c, "iad", nothing),
                          hgca = get(c, "hgca", nothing),
                          gost = get(c, "gost", nothing),
                          relast = get(c, "relast", nothing),
                          planets, engine = eng, output_dir = outdir, priors = priors, extra...)
    elseif op == "fit_rm"
        # fit_rm(; rv, phot, ...) is keyword-only and REQUIRES phot: the RM
        # amplitude is degenerate with the transit geometry, so a Transit
        # channel has to accompany the RM one.
        isrm(c) = startswith(string(get(c, "source", "")), "RM")
        rmi = findfirst(isrm, chs)
        rmi === nothing && error("fit_rm: no RM channel in payload")
        phi = findfirst(c -> !isrm(c), chs)
        phi === nothing && error(
            "fit_rm: needs a Transit channel alongside the RM channel")
        rmc, phc = chs[rmi], chs[phi]
        Base.invokelatest(f; rv = get(rmc, "data", nothing),
                          phot = get(phc, "data", nothing),
                          flavour = Symbol(get(rmc, "flavour", "reloaded")),
                          planets, engine = eng, output_dir = outdir, priors = priors, extra...)
    elseif op == "fit_ttv"
        c = isempty(chs) ? Dict{String,Any}() : chs[1]
        Base.invokelatest(f; transit_times = get(c, "transit_times", nothing),
                          nbody = Bool(get(c, "nbody", false)),
                          planets, engine = eng, output_dir = outdir, priors = priors, extra...)
    elseif op == "fit_binary"
        # fit_binary takes NO `planets` keyword: the companion count is fixed
        # by the SB2/BINARY_RV model itself.
        c = isempty(chs) ? Dict{String,Any}() : chs[1]
        Base.invokelatest(f; rv = get(c, "primary", nothing),
                          secondary = get(c, "secondary", nothing),
                          engine = eng, output_dir = outdir, priors = priors, extra...)
    else
        isempty(chs) && error("$op: no data channel in payload")
        Base.invokelatest(f, get(chs[1], "data", chs[1]);
                          planets, engine = eng, output_dir = outdir, priors = priors, extra...)
    end
    summary = r.summary
    if summary isa AbstractDict && haskey(summary, "figures")
        summary["figures"] = _fresh_figs(summary["figures"], _before)
    end
    return summary
end

# feature operations, if features.jl was loaded alongside api.jl
let _M = isdefined(Main, :Nereus) && isdefined(Main.Nereus, :FEATURE_ACTIONS) ?
         Main.Nereus : Main
if isdefined(_M, :FEATURE_ACTIONS)
    for (k, v) in _M.FEATURE_ACTIONS
        DISPATCH[k] = v
    end
end
if isdefined(_M, :FEATURE_NOT_IMPLEMENTED)
    for (k, why) in _M.FEATURE_NOT_IMPLEMENTED
        # A not-implemented STUB MUST NOT SHADOW A REAL IMPLEMENTATION. Nereus
        # currently lists "detrend.gp" in both FEATURE_ACTIONS (it was wired up
        # in 90c5373) and FEATURE_NOT_IMPLEMENTED (the entry was never removed).
        # This loop runs second, so without the guard the working function is
        # replaced by an error stub and `s.detrend.gp(...)` is dead.
        haskey(DISPATCH, k) && continue
        DISPATCH[k] = let w = why, n = k
            _ -> error("$n is not implemented: $w")
        end
    end
end
end

for _op in ("fit_rv", "fit_transit", "fit_astrometry", "fit_rm",
            "fit_tomography", "fit_ttv", "fit_binary", "fit_joint")
    DISPATCH[_op] = let o = _op
        payload -> _fit_dispatch(o, payload)
    end
end

# Render figures from chains already on disk. NOT a fit: no engine, no
# sampling. Plotting is a separate concern from fitting, and changing a figure
# should not cost the posterior again.
#
# This lives in the DAEMON rather than in Nereus.jl deliberately: it is pure
# composition of things that already exist there -- load_chains, _target_from,
# _make_plots -- so putting it in the package would have meant a bundle rebuild
# to ship a feature that needs no new Julia capability. Here it rides along
# with the pip package instead.
#
# It does reach for three unexported names. That is the trade: no rebuild, at
# the cost of coupling to internals. PY_API_VERSION is what guards it.
function act_replot(payload)
    mod = isdefined(Main, :Nereus) ? Main.Nereus :
          error("Nereus not loaded in this daemon")
    for f in (:load_chains, :_target_from, :_make_plots, :_planet_spec,
              :default_rv_planet)
        isdefined(mod, f) || error("runtime too old for replot: Nereus.$f missing")
    end

    outdir = String(get(payload, :output_dir, "")) 
    isempty(outdir) && error("replot: output_dir is required")
    chpath = let c = get(payload, :chains, nothing)
        c === nothing ? joinpath(outdir, "chains.nc") : String(c)
    end
    isfile(chpath) || error("no chains at $chpath -- run the fit first, " *
                            "or pass chains=...")
    chs_mcmc, _meta = Base.invokelatest(mod.load_chains, chpath)

    channels = _prepare_channels(payload)
    priors = let raw = _j(get(payload, :priors, Dict{String,Any}()))
        isempty(raw) ? nothing :
            Dict{String, mod.PriorSpec}(
                String(k) => mod._as_prior(_j(v)) for (k, v) in pairs(raw))
    end
    planets = get(payload, :planets, 1)

    # Mirror fit_rv: a bare planet COUNT with no block yields planets with no
    # parameters, and build_target then refuses. _target_from supplies an
    # astrometric block itself when astrometry is present; RV-only does not.
    if !(planets isa NamedTuple)
        rvi = findfirst(c -> String(get(c, "source", "")) == "RV", channels)
        has_as = any(c -> String(get(c, "source", "")) == "AS", channels)
        if rvi !== nothing && !has_as
            dv = collect(values(get(channels[rvi], "data", Dict())))
            if !isempty(dv)
                allt  = reduce(vcat, [collect(x.t)  for x in dv])
                allrv = reduce(vcat, [collect(x.rv) for x in dv])
                planets = Base.invokelatest(mod._planet_spec, planets;
                              block = Base.invokelatest(mod.default_rv_planet,
                                                        allt, allrv))
            end
        end
    end

    envelope = (:op, :channels, :priors, :planets, :output_dir, :chains,
                :plots, :plot_kwargs, :save_pdf)
    extra = Dict{Symbol,Any}()
    for (k, v) in pairs(payload)
        sk = Symbol(k); sk in envelope && continue; v === nothing && continue
        jv = _j(v)
        extra[sk] = (sk in (:parametrization, :time_anchor, :stability,
                            :ttv_backend) && jv isa AbstractString) ? Symbol(jv) : jv
    end
    tgt = Base.invokelatest(mod._target_from, channels, planets;
                            priors = priors, extra...)

    plots_raw = get(payload, :plots, String[])
    plist = plots_raw isa AbstractString ? [String(plots_raw)] :
            String.(collect(plots_raw))
    isempty(plist) && error("replot: plots is required")
    pk = _j(get(payload, :plot_kwargs, Dict{String,Any}()))
    pcfg = Dict{String,Any}(
        "output"  => Dict{String,Any}("plots" => plist, "plot_kwargs" => pk,
                                      "save_pdf" => Bool(get(payload, :save_pdf, false))),
        "sampler" => Dict{String,Any}("name" => "pt_emcee",
                          "kwargs" => Dict{String,Any}("n_walkers" => size(chs_mcmc, 3))))
    mkpath(outdir)
    proot = joinpath(outdir, "plots")
    before = _png_mtimes(proot)
    Base.invokelatest(mod._make_plots, pcfg, chs_mcmc, tgt.params, tgt.data, outdir)

    figs = Dict{String,Any}()
    if isdir(proot)
        for (root, _, files) in walkdir(proot), f in files
            endswith(f, ".png") || continue
            full = joinpath(root, f)
            _is_fresh(full, before) || continue
            figs[splitext(relpath(full, proot))[1]] = full
        end
    end
    return Dict{String,Any}("status" => "ok", "op" => "replot",
                            "output_dir" => outdir, "figures" => figs)
end
DISPATCH["replot"] = act_replot

# --- data loaders -------------------------------------------------------------
# Nereus EXPORTS a full loader suite (src/Nereus.jl:225-228) -- load_vizier_rv,
# load_orvara_rv, load_orvara_relast, load_tess_lc, load_hip_iad, load_gost,
# load_gaia_dr3, load_hgca_row -- and NONE of it was reachable from Python, so
# every notebook re-implemented the same parse loop. These ops are thin
# adapters: call the exported function, reshape its result into the channel
# wire format. The only parsing done here is `_read_labeled_rv`, flagged below.

_nereus() = isdefined(Main, :Nereus) ? Main.Nereus :
            error("Nereus not loaded in this daemon")

_sym(x) = x === nothing ? nothing : Symbol(String(x))

_wire(v::AbstractVector{<:Integer}) = Int.(v)
_wire(v::AbstractVector{<:Real})    = Float64.(v)
_wire(m::AbstractMatrix{<:Real})    = [Float64.(m[i, :]) for i in axes(m, 1)]
_wire(v) = v

"""Any loader return value (NamedTuple or struct) -> a JSON-able Dict."""
_as_dict(r::NamedTuple) = Dict{String,Any}(String(k) => _wire(v) for (k, v) in pairs(r))
_as_dict(r) = Dict{String,Any}(String(f) => _wire(getfield(r, f))
                               for f in fieldnames(typeof(r)))

"""(t, y, e, 1-based inst index, names) -> {name: {t, <ykey>, <ekey>}}."""
function _by_instrument(t, y, e, idx, names, ykey, ekey)
    out = Dict{String,Any}()
    for i in eachindex(t)
        ii = idx[i]
        nm = (ii >= 1 && ii <= length(names)) ? names[ii] : "INST$(ii)"
        d = get!(out, nm) do
            Dict{String,Any}("t" => Float64[], ykey => Float64[], ekey => Float64[])
        end
        push!(d["t"], Float64(t[i]))
        push!(d[ykey], Float64(y[i]))
        push!(d[ekey], Float64(e[i]))
    end
    return out
end

# SHIM, with an end date. A whitespace-delimited RV table with STRING
# instrument labels falls in the gap between the two shipped readers:
# load_orvara_rv handles whitespace but parses column 4 with `parse(Int, ...)`,
# load_vizier_rv handles string labels but hardcodes ','. Nereus's OWN
# gaia4_rv.dat and hd114762_rv.dat are in that gap, so no shipped loader reads
# the package's own data. Fixed here because daemon.jl ships with the pip
# package and needs no 600 MB runtime rebuild; it belongs in load_orvara_rv and
# moves there in Nereus 0.5.4, at which point this function goes away.
function _read_labeled_rv(path)
    t = Float64[]; rv = Float64[]; er = Float64[]; lab = String[]
    for line in eachline(path)
        s = strip(line)
        (isempty(s) || startswith(s, "#")) && continue
        toks = occursin(',', s) ? strip.(split(s, ',')) : split(s)
        length(toks) >= 4 || continue
        push!(t,  parse(Float64, toks[1]))
        push!(rv, parse(Float64, toks[2]))
        push!(er, parse(Float64, toks[3]))
        push!(lab, String(toks[4]))
    end
    isempty(t) && throw(ArgumentError("no data rows in $path"))
    names = sort!(unique(lab))
    idx   = [findfirst(==(l), names) for l in lab]
    return (t = t, rv = rv, rv_err = er, rv_inst = idx, instruments = names)
end

"""RV table -> {instrument: {t, rv, rv_err}}, the shape `RV(data=...)` wants."""
function act_load_rv(payload)
    N    = _nereus()
    path = String(payload[:path])
    isfile(path) || error("RV file not found: $path")
    fmt = String(get(payload, :format, "auto"))
    fmt == "auto" && (fmt = endswith(lowercase(path), ".csv") ? "vizier" : "labeled")

    r = if fmt == "vizier"
        # `inst_col = nothing` is meaningful (single-instrument file), so it is
        # only defaulted when the key is absent, not when it is null.
        ic = haskey(payload, :inst_col) ? _sym(payload[:inst_col]) : :inst
        Base.invokelatest(N.load_vizier_rv, path;
                          t_col   = _sym(get(payload, :t_col, "bjd")),
                          rv_col  = _sym(get(payload, :rv_col, "rv")),
                          err_col = _sym(get(payload, :err_col, "rv_err")),
                          inst_col = ic)
    elseif fmt == "orvara"
        # jd_to_mjd=false: the offset is applied once below for every route, so
        # the three readers cannot disagree about the time scale. load_vizier_rv
        # does NOT convert, load_orvara_rv does -- that inconsistency stops here.
        Base.invokelatest(N.load_orvara_rv, path; jd_to_mjd = false)
    elseif fmt == "labeled"
        _read_labeled_rv(path)
    else
        error("unknown RV format $(fmt); use \"vizier\", \"orvara\", \"labeled\" or \"auto\"")
    end

    names = if haskey(payload, :instrument_names)
        String[String(x) for x in payload[:instrument_names]]
    elseif hasproperty(r, :instruments)
        String[String(x) for x in r.instruments]
    else
        # orvara instrument IDs are 0-based in the file, 1-based after loading.
        String["INST$(i - 1)" for i in 1:maximum(r.rv_inst)]
    end
    if haskey(payload, :rename)
        rn = _j(payload[:rename])
        names = String[get(rn, n, n) for n in names]
    end

    off = Float64(get(payload, :time_offset, 2_400_000.5))
    t   = Float64.(r.t) .- off
    return Dict{String,Any}(
        "data"        => _by_instrument(t, r.rv, r.rv_err, r.rv_inst, names, "rv", "rv_err"),
        "instruments" => names,
        "n"           => length(t),
        "format"      => fmt,
        "path"        => path)
end
DISPATCH["load_rv"] = act_load_rv

"""Light curve -> {instrument: {t, flux, flux_err}}, for `Transit(data=...)`."""
function act_load_photometry(payload)
    N    = _nereus()
    path = String(payload[:path])
    isfile(path) || error("light curve not found: $path")
    tw = get(payload, :trim_window, nothing)
    r = tw === nothing ?
        Base.invokelatest(N.load_tess_lc, path) :
        Base.invokelatest(N.load_tess_lc, path;
                          trim_window = (Float64(tw[1]), Float64(tw[2])))
    inst = String(get(payload, :instrument, "TESS"))
    off  = Float64(get(payload, :time_offset, 0.0))
    return Dict{String,Any}(
        "data" => Dict{String,Any}(inst => Dict{String,Any}(
            "t"        => Float64.(r.t) .- off,
            "flux"     => Float64.(r.flux),
            "flux_err" => Float64.(r.flux_err))),
        "instruments" => [inst],
        "n"           => length(r.t),
        "path"        => path)
end
DISPATCH["load_photometry"] = act_load_photometry

"""orvara-format relative astrometry -> the `values` block `Astrometry(relast=)` takes."""
function act_load_relastrom(payload)
    N    = _nereus()
    path = String(payload[:path])
    isfile(path) || error("relative astrometry file not found: $path")
    d = _as_dict(Base.invokelatest(N.load_orvara_relast, path))
    d["path"] = path
    d["n"] = length(get(d, "t", Float64[]))
    return d
end
DISPATCH["load_relastrom"] = act_load_relastrom

# The remaining exported loaders take different argument shapes, so each gets
# its own adapter rather than one stringly-typed `load(kind, ...)`.
function act_load_iad(payload)
    N = _nereus()
    path = String(payload[:path])
    isfile(path) || error("IAD file not found: $path")
    return _as_dict(Base.invokelatest(N.load_hip_iad, path))
end
DISPATCH["load_iad"] = act_load_iad

function act_load_gost(payload)
    N = _nereus()
    path = String(payload[:path])
    isfile(path) || error("GOST file not found: $path")
    return _as_dict(Base.invokelatest(N.load_gost, path))
end
DISPATCH["load_gost"] = act_load_gost

function act_load_gaia_dr3(payload)
    N = _nereus()
    spec = haskey(payload, :path) ? String(payload[:path]) : payload[:source_id]
    return _as_dict(Base.invokelatest(N.load_gaia_dr3, spec))
end
DISPATCH["load_gaia_dr3"] = act_load_gaia_dr3

function act_load_hgca(payload)
    N = _nereus()
    path = String(payload[:path])
    isfile(path) || error("HGCA FITS not found: $path")
    return _as_dict(Base.invokelatest(N.load_hgca_row, path, Int(payload[:hip])))
end
DISPATCH["load_hgca"] = act_load_hgca


# --- shipped datasets ---------------------------------------------------------
# Nereus ships the data these examples use, so a user should never have to
# build a path to it or know which reader it needs. The registry is the one
# place that knows both. It lives here (pip-shipped) rather than in Nereus so
# a column-name correction does not need a runtime release.

_data_dir() = joinpath(pkgdir(_nereus()), "test", "data")

const DATASETS = Dict{String,Any}(
    "gaia4" => Dict{String,Any}(
        "target" => "Gaia-4",
        "ref"    => "Stefansson et al. 2025, AJ 169, 107 (arXiv:2410.05654) table 5",
        "rv"     => Dict{String,Any}("path" => "gaia4_rv.dat", "format" => "labeled")),
    "hd114762" => Dict{String,Any}(
        "target" => "HD 114762",
        "ref"    => "Rosenthal et al. 2021, ApJS 255, 8 (California Legacy)",
        "rv"     => Dict{String,Any}("path" => "hd114762_rv.dat", "format" => "labeled",
                                     "rename" => Dict{String,Any}("j" => "HIRES",
                                                                  "lick" => "Lick"))),
    "hd159062" => Dict{String,Any}(
        "target" => "HD 159062",
        "ref"    => "Hirsch et al. 2019, ApJ 878, 50",
        "rv"     => Dict{String,Any}("path" => "hd159062_rv.dat", "format" => "orvara",
                                     "instrument_names" => ["HIRES"]),
        "relast" => Dict{String,Any}("path" => "hd159062_relast.txt")),
    "hd4747" => Dict{String,Any}(
        "target" => "HD 4747",
        "ref"    => "Brandt et al. 2019, AJ 158, 140",
        "rv"     => Dict{String,Any}("path" => "hd4747_rv.dat", "format" => "orvara",
                                     "instrument_names" => ["HIRES"]),
        "relast" => Dict{String,Any}("path" => "hd4747_relast.txt")),
    # The CSVs predate load_vizier_rv's defaults (`rv_err`, `inst`) and use
    # `rv_error`/`instrument`, so the package's own reader does not read the
    # package's own tables without these overrides. That is what the registry
    # is for.
    "51peg" => Dict{String,Any}(
        "target" => "51 Peg", "ref" => "Mayor & Queloz 1995 + archival",
        "rv" => Dict{String,Any}("path" => "51peg.csv", "format" => "vizier",
                                 "err_col" => "rv_error", "inst_col" => "instrument")),
    "gj876" => Dict{String,Any}(
        "target" => "GJ 876", "ref" => "archival",
        "rv" => Dict{String,Any}("path" => "gj876.csv", "format" => "vizier",
                                 "err_col" => "rv_err", "inst_col" => "instrument")),
    "eps_eri" => Dict{String,Any}(
        "target" => "eps Eri", "ref" => "archival",
        "rv" => Dict{String,Any}("path" => "eps_eri.csv", "format" => "vizier",
                                 "err_col" => "rv_error", "inst_col" => "instrument")),
    "hd18599" => Dict{String,Any}(
        "target" => "HD 18599", "ref" => "archival",
        "rv" => Dict{String,Any}("path" => "hd18599.csv", "format" => "vizier",
                                 "err_col" => "rv_error", "inst_col" => "instrument")),
    "hd33636" => Dict{String,Any}(
        "target" => "HD 33636", "ref" => "archival",
        "rv" => Dict{String,Any}("path" => "hd33636.csv", "format" => "vizier",
                                 "err_col" => "rv_error", "inst_col" => "instrument")),
    "hd33636_bean" => Dict{String,Any}(
        "target" => "HD 33636", "ref" => "Bean et al. 2007, AJ 134, 749",
        "rv" => Dict{String,Any}("path" => "hd33636_bean.csv", "format" => "vizier",
                                 "err_col" => "rv_error", "inst_col" => "instrument")),
    "hd38529" => Dict{String,Any}(
        "target" => "HD 38529", "ref" => "archival",
        "rv" => Dict{String,Any}("path" => "hd38529.csv", "format" => "vizier",
                                 "err_col" => "rv_error", "inst_col" => "instrument")),
)

function act_datasets(_)
    dir = _data_dir()
    out = Any[]
    for (name, spec) in sort!(collect(DATASETS), by = first)
        chans = String[k for k in ("rv", "photometry", "relast") if haskey(spec, k)]
        push!(out, Dict{String,Any}(
            "name" => name,
            "target" => spec["target"],
            "ref" => spec["ref"],
            "channels" => chans,
            "available" => all(isfile(joinpath(dir, spec[c]["path"])) for c in chans)))
    end
    return Dict{String,Any}("data_dir" => dir, "datasets" => out)
end
DISPATCH["datasets"] = act_datasets

function act_dataset(payload)
    name = String(payload[:name])
    haskey(DATASETS, name) || error("unknown dataset $(name); known: " *
                                    join(sort!(collect(keys(DATASETS))), ", "))
    spec = DATASETS[name]
    dir  = _data_dir()
    out  = Dict{String,Any}("name" => name, "target" => spec["target"],
                            "ref" => spec["ref"])
    if haskey(spec, "rv")
        p = copy(spec["rv"]); p["path"] = joinpath(dir, p["path"])
        out["rv"] = act_load_rv(Dict{Symbol,Any}(Symbol(k) => v for (k, v) in p))
    end
    if haskey(spec, "photometry")
        p = copy(spec["photometry"]); p["path"] = joinpath(dir, p["path"])
        out["photometry"] = act_load_photometry(Dict{Symbol,Any}(Symbol(k) => v for (k, v) in p))
    end
    if haskey(spec, "relast")
        p = copy(spec["relast"]); p["path"] = joinpath(dir, p["path"])
        out["relast"] = act_load_relastrom(Dict{Symbol,Any}(Symbol(k) => v for (k, v) in p))
    end
    return out
end
DISPATCH["dataset"] = act_dataset

# Serialises job execution across connections; see the dispatch loop.
const JOB_LOCK = ReentrantLock()

# --- liveness --------------------------------------------------------------
# A daemon MUST NOT outlive its owner. Python's atexit cannot run on SIGKILL,
# on a kernel OOM kill, or on a hard crash — measured 2026-08-08: killing the
# parent with -9 left a 530 MB Julia process running indefinitely. So the
# daemon polices itself with two independent guards:
#
#   1. parent-death: when the owner dies we are reparented (to launchd/init),
#      so getppid() changing away from the pid we were handed means: exit.
#   2. idle timeout: nothing has talked to us in `idle` seconds, so exit even
#      if the parent is somehow still around but has forgotten us.
#
# Either alone would leak in some scenario; both together bound the lifetime.

const LAST_ACTIVITY = Ref(time())
touch_activity!() = (LAST_ACTIVITY[] = time())

function start_watchdog(parent_pid::Int, idle::Float64)
    @async while true
        sleep(2)
        if parent_pid > 0 && ccall(:getppid, Cint, ()) != parent_pid
            println(stderr, "nereus daemon: owner $parent_pid gone — exiting")
            flush(stderr); exit(0)
        end
        if idle > 0 && (time() - LAST_ACTIVITY[]) > idle
            println(stderr, "nereus daemon: idle $(round(Int, idle))s — exiting")
            flush(stderr); exit(0)
        end
    end
end

function serve(sockpath::String, readyfile::Union{Nothing,String} = nothing;
               parent_pid::Int = 0, idle::Float64 = 1800.0)
    ispath(sockpath) && rm(sockpath; force = true)
    server = listen(sockpath)
    readyfile === nothing || write(readyfile, "ready")
    touch_activity!()
    start_watchdog(parent_pid, idle)
    println(READY_SENTINEL); flush(stdout)

    while true
        conn = accept(server)
        touch_activity!()
        @async begin
            try
                while true
                    req = read_msg(conn)
                    req === nothing && break
                    touch_activity!()
                    action = get(req, :action, "")
                    if action == "shutdown"
                        write_msg(conn, Dict("ok" => true, "result" => "bye"))
                        close(conn); exit(0)
                    end
                    fn = get(DISPATCH, String(action), nothing)
                    if fn === nothing
                        write_msg(conn, Dict("ok" => false,
                            "error" => "unknown action $(action)",
                            "known" => sort!(collect(keys(DISPATCH)))))
                        continue
                    end
                    try
                        # ONE JOB AT A TIME. Each connection is served by its
                        # own @async task, so two requests could execute
                        # concurrently -- and `sample_pt_emcee` uses
                        # `@threads :static`, which Julia refuses to run
                        # concurrently or nested (its per-thread RNGs are
                        # indexed by threadid, so the static schedule is load
                        # bearing). The reachable path is cancel-then-rerun:
                        # interrupting a cell drops the client socket but does
                        # NOT stop the Julia task, so the next request raced
                        # the one still unwinding and died with
                        #   `@threads :static` cannot be used concurrently or nested
                        # Serialising is right regardless: a fit saturates
                        # every thread, so two at once would only thrash.
                        got = trylock(JOB_LOCK)
                        if !got
                            # Do not block forever on a job that may never
                            # finish; say what is happening and what fixes it.
                            write_msg(conn, Dict("ok" => false, "error" =>
                                "this runtime is already running a job. If you " *
                                "cancelled a cell, the Julia side keeps going " *
                                "until it notices -- wait for it, or call " *
                                "s.stop() and start a fresh session."))
                            continue
                        end
                        try
                            write_msg(conn, Dict("ok" => true,
                                                 "result" => fn(get(req, :payload, Dict()))))
                        finally
                            unlock(JOB_LOCK)
                        end
                    catch err
                        write_msg(conn, Dict("ok" => false,
                            "error" => sprint(showerror, err),
                            "backtrace" => sprint(Base.show_backtrace, catch_backtrace())))
                    end
                end
            catch err
                @error "nereus daemon: connection aborted" exception=(err, catch_backtrace())
            finally
                close(conn)
            end
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) >= 1 || (println(stderr, "usage: daemon.jl <sock> [parent_pid] [idle_s]"); exit(2))
    ppid = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 0
    idle = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 1800.0
    serve(ARGS[1]; parent_pid = ppid, idle = idle)
end
