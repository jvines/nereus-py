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
    Dict("pong" => true,
         "julia" => string(VERSION),
         "cpu" => Sys.CPU_NAME,
         "threads" => Threads.nthreads())
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

_engine_of(payload) = _j(get(payload, :engine, Dict("engine" => "pt")))

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
    priors = _j(get(payload, :priors, Dict{String,Any}()))
    outdir = outdir === nothing ? nothing : String(outdir)

    f = getfield(mod, Symbol(op))
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
        Base.invokelatest(f, chs...; planets, engine = eng, output_dir = outdir, priors = priors)
    elseif op == "fit_astrometry"
        c = isempty(chs) ? Dict{String,Any}() : chs[1]
        Base.invokelatest(f; iad = get(c, "iad", nothing),
                          hgca = get(c, "hgca", nothing),
                          gost = get(c, "gost", nothing),
                          relast = get(c, "relast", nothing),
                          parallax = get(c, "parallax", nothing),
                          m_pri = get(c, "m_pri", nothing),
                          planets, engine = eng, output_dir = outdir, priors = priors)
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
                          planets, engine = eng, output_dir = outdir, priors = priors)
    elseif op == "fit_ttv"
        c = isempty(chs) ? Dict{String,Any}() : chs[1]
        Base.invokelatest(f; transit_times = get(c, "transit_times", nothing),
                          nbody = Bool(get(c, "nbody", false)),
                          planets, engine = eng, output_dir = outdir, priors = priors)
    elseif op == "fit_binary"
        # fit_binary takes NO `planets` keyword: the companion count is fixed
        # by the SB2/BINARY_RV model itself.
        c = isempty(chs) ? Dict{String,Any}() : chs[1]
        Base.invokelatest(f; rv = get(c, "primary", nothing),
                          secondary = get(c, "secondary", nothing),
                          engine = eng, output_dir = outdir, priors = priors)
    else
        isempty(chs) && error("$op: no data channel in payload")
        Base.invokelatest(f, get(chs[1], "data", chs[1]);
                          planets, engine = eng, output_dir = outdir, priors = priors)
    end
    return r.summary
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
                        write_msg(conn, Dict("ok" => true,
                                             "result" => fn(get(req, :payload, Dict()))))
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
