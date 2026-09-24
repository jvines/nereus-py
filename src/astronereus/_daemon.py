"""Supervise a warm Julia daemon and talk to it over a unix socket.

Deliberately NOT juliacall: embedding deadlocks under Julia's `@threads`, which
every sampler uses. A separate process also means a crashing job cannot take the
Python interpreter with it.
"""

from __future__ import annotations

import atexit
import json
import os
import re
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

from . import _runtime

_HDR = struct.Struct(">Q")


def _enc(o):
    """JSON fallback for the types a scientific payload actually carries.

    Channels are built with `dataclasses.asdict`, which leaves numpy arrays
    untouched, so every realistic call ships ndarrays. Non-finite floats pass
    through as NaN/Infinity — `daemon.jl` reads with `allow_inf = true`.
    """
    if hasattr(o, "tolist"):            # ndarray, numpy scalar
        return o.tolist()
    if hasattr(o, "item"):              # any remaining 0-d numpy scalar
        return o.item()
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    raise TypeError(
        f"cannot send {type(o).__name__} to Julia. Channel values must be "
        "JSON-able or expose .tolist()/.item() -- numpy arrays, pandas Series, "
        "astropy columns and plain lists all qualify. "
        "If you meant to pass numpy: pip install astronereus[arrays]")


class FitInterrupted(RuntimeError):
    """A fit stopped because the caller cancelled it.

    Raised instead of DaemonError's 50-line Julia backtrace, which only ever
    named whichever kernel the signal happened to land in -- for one Gaia
    astrometry fit, `_iad_normal_equations!` -- and read like a crash in the
    model rather than the cancellation it was.
    """


class DaemonError(RuntimeError):
    pass


# The left bracket Nereus's ProgressBar draws (src/progress.jl). Present on
# every progress line and on nothing else, so it identifies one unambiguously.
_BAR_GLYPH = "\u2595"

# On a host that cannot redraw, progress is forwarded as whole lines, throttled
# on BOTH axes: every _BAR_STEP of progress, and at least every _BAR_EVERY
# seconds so a slow fit still shows it is alive. Time alone does not work -- at
# one line per 5s a 30-minute fit buries the cell in 360 copies of the bar.
_BAR_STEP = 5.0      # percent
_BAR_EVERY = 60.0    # seconds

# How often the calling thread surfaces to redraw a native progress bar.
_POLL = 0.25

# A call must outlast this before it earns a bar, so ping and the other fast
# actions draw nothing at all.
_BAR_DELAY = 1.5
_BAR_PCT = re.compile(r"\(([0-9.]+)%\)")


def _parse_bar(line: str) -> dict:
    """Split one Julia progress line into title, percent and labelled fields.

    Shape: `pt_emcee | \u2595####....\u258f | 1900/4000 (47.5%) | acc=0.218 | ETA 3s`
    The drawn bar is dropped -- a host that renders its own does not want it.
    """
    segs = [x.strip() for x in line.split("|")]
    title = segs[0] or "sampling"
    m = _BAR_PCT.search(line)
    pct = float(m.group(1)) if m else 0.0
    fields = []
    for seg in segs[1:]:
        if not seg or _BAR_GLYPH in seg:
            continue
        if "=" in seg:
            k, _, v = seg.partition("=")
            fields.append((k.strip(), v.strip()))
        elif "(" in seg and "/" in seg:                 # "1900/4000 (47.5%)"
            fields.append(("step", seg.split("(")[0].strip()))
        else:                                           # "ETA 3s", "elapsed 4s"
            k, _, v = seg.partition(" ")
            fields.append((k.strip(), v.strip()))
    return {"title": title, "pct": pct, "fields": fields,
            "status": " | ".join(segs[2:])}


# Discrete colours are sampled from `cool` and pastelised (strength 0.45), the
# same rule the plotting code follows, so the bar belongs to the same family as
# the figures the fit produces. cool runs cyan -> magenta; pastelising is
# rgb + (1-rgb)*0.45, giving #73FFFF -> #FF73FF.
_COOL_LO, _COOL_HI = "#73FFFF", "#FF73FF"

_BAR_CSS = (
 ".nereus-pb{width:100%;font:13px/1.45 ui-sans-serif,system-ui,-apple-system,sans-serif;"
 "color:currentColor;margin:.35rem 0}"
 ".nereus-pb-h{display:flex;justify-content:space-between;align-items:baseline;"
 "gap:1rem;margin-bottom:.4rem}"
 ".nereus-pb-t{font-weight:600;letter-spacing:.02em}"
 ".nereus-pb-p{font-variant-numeric:tabular-nums;opacity:.75}"
 ".nereus-pb-track{width:100%;height:10px;border-radius:999px;overflow:hidden;"
 "background:color-mix(in srgb,currentColor 12%,transparent)}"
 ".nereus-pb-fill{height:100%;border-radius:999px;transition:width .25s linear;"
 "background:linear-gradient(90deg," + _COOL_LO + " 0%," + _COOL_HI + " 100%)}"
 ".nereus-pb-fill.ind{width:35%;animation:nereus-pb-slide 1.1s ease-in-out infinite}"
 "@keyframes nereus-pb-slide{0%{margin-left:-35%}100%{margin-left:100%}}"
 "@media (prefers-reduced-motion:reduce){.nereus-pb-fill.ind{animation:none;"
 "width:100%;opacity:.5}}"
 ".nereus-pb-s{display:flex;flex-wrap:wrap;gap:.35rem 1.25rem;margin-top:.5rem}"
 ".nereus-pb-s div{display:flex;flex-direction:column}"
 ".nereus-pb-k{font-size:10px;text-transform:uppercase;letter-spacing:.07em;opacity:.55}"
 ".nereus-pb-v{font-variant-numeric:tabular-nums}")

_PRETTY = {"acc": "accept", "nevals": "evals", "Rhat": "R-hat", "step": "step"}


def _bar_html(title: str, pct, fields, note: str = "") -> str:
    """Full-width progress bar. `pct=None` renders the indeterminate state."""
    import html as _h
    ind = pct is None
    head = _h.escape(note or "preparing") if ind else f"{pct:.1f}%"
    fill = ('<div class="nereus-pb-fill ind"></div>' if ind else
            '<div class="nereus-pb-fill" style="width:'
            f'{max(0.0, min(100.0, pct)):.1f}%"></div>')
    stats = "".join(
        f'<div><span class="nereus-pb-k">{_h.escape(_PRETTY.get(k, k))}</span>'
        f'<span class="nereus-pb-v">{_h.escape(str(v))}</span></div>'
        for k, v in (fields or []))
    return ('<div class="nereus-pb"><div class="nereus-pb-h">'
            f'<span class="nereus-pb-t">{_h.escape(title)}</span>'
            f'<span class="nereus-pb-p">{head}</span></div>'
            f'<div class="nereus-pb-track">{fill}</div>'
            f'<div class="nereus-pb-s">{stats}</div></div>')


def _bars_html(bodies) -> str:
    """Stack one or more bars under a single copy of the stylesheet.

    A fit emits SEVERAL bars in sequence -- the sampler's, then one per plot
    phase -- and each `mo.output.replace` swaps the whole cell output, so a
    later bar erased the finished one before it. The sampler's final frame
    carries R-hat, ESS and min_swap: the numbers that say whether the run can
    be trusted, gone the moment plotting started. Completed bars are now kept
    and the live one is appended beneath them.
    """
    return f"<style>{_BAR_CSS}</style>" + "".join(bodies)


class JuliaDaemon:
    def __init__(self, version: str = _runtime.JULIA_VERSION,
                 project: str | os.PathLike | None = None,
                 threads: str | int = "auto",
                 preload: str | None = "Nereus",
                 api_file: str | list[str] | None = None,
                 startup_timeout: float = 300.0,
                 idle_timeout: float = 1800.0):
        self.version = version
        self.project = str(project) if project else None
        self.threads = threads
        self.preload = preload
        # Extra Julia file to include after `using` — the public API surface.
        # In the shipped package this lives inside Nereus itself; during
        # bring-up it is loaded from a path so it can be iterated on.
        self.api_file = api_file
        self.startup_timeout = startup_timeout
        # The daemon exits on its own if we die abnormally (atexit cannot run
        # on SIGKILL/OOM) or if nothing talks to it for this long. Verified:
        # without this a -9'd parent leaves a ~530 MB Julia process behind.
        self.idle_timeout = idle_timeout
        self._proc: subprocess.Popen | None = None
        self._sock: socket.socket | None = None
        self._lock = threading.Lock()
        # set up by start(); declared here so stop() is safe before it runs
        self._log_fh = None
        self._forward: threading.Event | None = None
        # Set for the duration of a call on a host that draws its own progress
        # bar; the pump feeds it parsed updates instead of writing to stderr.
        self._progress_sink = None
        self._resp = None
        self._pump_thread: threading.Thread | None = None
        self._tmp = Path(tempfile.mkdtemp(prefix="nereus-daemon-"))
        self.sock_path = self._tmp / "daemon.sock"
        self.log_path = self._tmp / "daemon.log"

    # -- lifecycle ---------------------------------------------------------
    def _reconnect(self) -> bool:
        """Re-open the socket to a daemon that is still running.

        `_drop()` closes the connection whenever framing could have been left
        desynchronised -- most often because a cancelled fit's response was
        never read. The Julia process survives that and stays warm, so the
        next call must reattach to it rather than boot a new one. Nothing did:
        `start()` returns early when `_proc` is alive, so `_sock` stayed None
        and every later call died on `NoneType.settimeout`. Reaching that took
        a cancel, which until now never dropped the socket at all, so the two
        bugs hid each other.
        """
        if self._proc is None or self._proc.poll() is not None:
            return False
        if not self.sock_path.exists():
            return False
        try:
            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            s.connect(str(self.sock_path))
        except OSError:
            return False
        self._sock = s
        return True

    def start(self) -> "JuliaDaemon":
        if self._proc is not None:
            # Alive but disconnected (see _reconnect). If it cannot be
            # reattached, fall through and treat it as gone.
            if self._sock is None and not self._reconnect():
                if self._proc.poll() is not None:
                    self._proc = None
                else:
                    raise DaemonError(
                        "the daemon is running but its socket cannot be "
                        f"reopened ({self.sock_path}). Call stop() and retry.")
            return self
        julia, env = _runtime.julia_env(self.version)
        entry = Path(__file__).parent / "julia" / "daemon.jl"

        cmd = [str(julia), f"-t{self.threads}", "--startup-file=no"]
        if self.project:
            cmd.append(f"--project={self.project}")
        # Preload inside the daemon so the ~20 s `using` cost is paid once, at
        # boot, rather than on the first job.
        ppid = os.getpid()
        if self.preload:
            _files = ([self.api_file] if isinstance(self.api_file, str)
                      else list(self.api_file or []))
            api_inc = "".join(f'include("{p}"); ' for p in _files)
            cmd += ["-e", f"@eval using {self.preload}; "
                          f"{api_inc}"
                          f"include(\"{entry}\"); "
                          f"serve(ARGS[1]; parent_pid=parse(Int,ARGS[2]), "
                          f"idle=parse(Float64,ARGS[3]))",
                    str(self.sock_path), str(ppid), str(self.idle_timeout)]
        else:
            cmd += [str(entry), str(self.sock_path), str(ppid),
                    str(self.idle_timeout)]

        # Julia's startup output goes to YOUR TERMINAL until the daemon is
        # ready, and to the log always.
        #
        # It used to go only to a log inside an unannounced mkdtemp. Starting
        # the daemon can take minutes when a package needs recompiling, so that
        # was minutes of silence with the only diagnostic somewhere the user
        # was never told about. "Tail this file from another terminal" is not
        # an answer; the output belongs where the person is looking.
        #
        # Forwarding stops once the socket answers: after that the daemon is
        # long-lived and its chatter would interleave with the caller's own
        # output. The log keeps everything either way.
        self._log_fh = self.log_path.open("w", buffering=1, errors="replace")
        self._forward = threading.Event()
        self._forward.set()
        # SAME PROCESS GROUP AS THE CALLER, deliberately. A SIGINT meant for
        # the caller reaches Julia too, so cancelling the cell or hitting
        # Ctrl-C stops the sampler -- which is the point of cancelling. 0.4.7
        # put the daemon in its own session to stop a marimo interrupt killing
        # a fit, which was backwards: it left the fit running with the cell
        # already gone and no way to reach it short of stop().
        #
        # The daemon itself survives the signal; it catches the interrupt, ends
        # that one job and keeps serving, so the next fit reuses the same warm
        # runtime. Only the reporting needed fixing, see FitInterrupted.
        self._proc = subprocess.Popen(
            cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            bufsize=0)

        def _pump(stream, fh, forward):
            # CHUNKS, not lines. Julia's progress bars redraw with \r and emit
            # no newline until they finish, so a line-based reader buffers the
            # entire bar until the fit is over -- which is precisely when it
            # stops being useful. Reading raw bytes also passes \r through
            # untouched, so the bar redraws in place as intended.
            #
            # No prefix for the same reason: anything prepended to a \r-updated
            # line corrupts the redraw.
            #
            # The daemon's stdout is a PIPE, so Julia's ProgressBar decides it
            # is off-tty and prints a FULL LINE per update instead of redrawing
            # -- grep-able in the log, but ~70 stacked copies of the same bar in
            # a notebook cell. Those lines are collapsed back into one in-place
            # bar here: the pipe is an implementation detail of how the client
            # talks to Julia, not a statement about where the user is looking.
            # Padding to the previous width rather than an ANSI erase, because
            # \r is the one cursor control every notebook front-end honours.
            fd = stream.fileno()
            buf = ""
            bar_len = 0                      # width of the bar currently shown
            last_bar, last_pct = 0.0, -1e9   # throttle, line mode only
            style = _runtime.progress_style()
            inplace = style == "inplace"
            while True:
                try:
                    data = os.read(fd, 8192)
                except (OSError, ValueError):
                    break
                if not data:
                    break
                text = data.decode("utf-8", errors="replace")
                fh.write(text)
                if not forward.is_set():
                    buf = ""
                    continue
                buf += text
                out = []
                while True:
                    nl = buf.find("\n")
                    if nl < 0:
                        break
                    line, buf = buf[:nl], buf[nl + 1:]
                    line = line.rstrip("\r")
                    if _BAR_GLYPH in line:                 # a progress update
                        sink = self._progress_sink
                        if sink is not None:
                            # Native-bar host (marimo). The bar is driven by
                            # the CALLING thread, because marimo's runtime
                            # context is thread-local -- output emitted from
                            # this pump thread would not attach to the cell.
                            try:
                                sink(_parse_bar(line))
                            except Exception:
                                pass
                        elif inplace:
                            out.append("\r" + line.ljust(bar_len))
                            bar_len = len(line)
                        else:
                            # A host that does not honour \r would run every
                            # update into one unreadable line, so it gets whole
                            # lines -- throttled to roughly one per _BAR_EVERY
                            # seconds, plus the final one, which is a dozen
                            # legible lines instead of seventy.
                            now = time.monotonic()
                            m = _BAR_PCT.search(line)
                            pct = float(m.group(1)) if m else 100.0
                            if (pct >= 100.0 or pct - last_pct >= _BAR_STEP
                                    or now - last_bar >= _BAR_EVERY):
                                last_bar, last_pct = now, pct
                                out.append(line + "\n")
                    else:
                        if bar_len:                        # close the bar first
                            out.append("\n")
                            bar_len = 0
                        out.append(line + "\n")
                if out:
                    sys.stderr.write("".join(out))
                    sys.stderr.flush()

        self._pump_thread = threading.Thread(
            target=_pump, args=(self._proc.stdout, self._log_fh, self._forward),
            daemon=True)
        self._pump_thread.start()
        atexit.register(self.stop)

        # Julia's output goes to the log, not the terminal -- it is noisy and
        # would interleave with the caller's own. But startup can take MINUTES
        # when a package needs recompiling, and a silent multi-minute wait with
        # the log in an unannounced mkdtemp is indistinguishable from a hang.
        # So: say where the log is, and tick while waiting.
        tty = _runtime.show_progress()
        t0 = time.time()
        announced = False
        deadline = t0 + self.startup_timeout
        while time.time() < deadline:
            if self._proc.poll() is not None:
                raise DaemonError(
                    f"daemon exited with {self._proc.returncode}\n"
                    f"  log: {self.log_path}\n"
                    f"--- log ---\n{self.log_path.read_text(errors='replace')[-4000:]}")
            if self.sock_path.exists():
                try:
                    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                    s.connect(str(self.sock_path))
                    self._sock = s
                    self._forward.clear()      # daemon is live; stop echoing
                    if announced and tty:
                        print(f"astronereus: daemon ready "
                              f"({time.time() - t0:.0f}s)",
                              file=sys.stderr, flush=True)
                    return self
                except OSError:
                    pass
            if tty and not announced and time.time() - t0 > 3.0:
                announced = True
                print(f"astronereus: starting the Julia daemon "
                      f"(up to {self.startup_timeout:.0f}s; log: {self.log_path})",
                      file=sys.stderr, flush=True)
            time.sleep(0.1)
        self.stop()
        raise DaemonError(
            f"daemon did not become ready within {self.startup_timeout}s.\n"
            "  If it was still precompiling, that is not a failure, just a\n"
            "  longer wait than the default allows -- raise it with\n"
            "  astronereus.session(startup_timeout=1200).\n"
            f"  log: {self.log_path}\n"
            f"--- log ---\n{self.log_path.read_text(errors='replace')[-4000:]}")

    def stop(self) -> None:
        if self._sock is not None:
            try:
                self._send({"action": "shutdown"})
            except Exception:
                pass
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None
        if self._forward is not None:
            self._forward.clear()      # never echo a dying daemon's output
        if self._proc is not None:
            try:
                self._proc.wait(timeout=10)
            except Exception:
                self._proc.kill()
            self._proc = None
        # After the process is gone the pump drains and exits on EOF. Join
        # briefly so the log is complete before anyone reads it for an error
        # message; it is a daemon thread, so a hung read cannot block exit.
        if self._pump_thread is not None:
            self._pump_thread.join(timeout=2.0)
            self._pump_thread = None
        if self._log_fh is not None:
            try:
                self._log_fh.close()
            except Exception:
                pass
            self._log_fh = None

    # -- transport ---------------------------------------------------------
    def _send(self, obj) -> None:
        body = json.dumps(obj, default=_enc).encode()
        self._sock.sendall(_HDR.pack(len(body)) + body)

    def _recvn(self, n: int) -> bytes:
        buf = bytearray()
        while len(buf) < n:
            chunk = self._sock.recv(n - len(buf))
            if not chunk:
                raise DaemonError("daemon closed the connection")
            buf += chunk
        return bytes(buf)

    def _recvn_polling(self, n: int, deadline, on_tick) -> bytes:
        """`_recvn`, but it surfaces every `_POLL` seconds so the CALLING
        thread can redraw a native progress bar.

        The partial buffer survives each timeout, which a plain
        `settimeout(_POLL)` around `_recvn` would not: a socket timeout
        mid-message discards nothing here, so framing stays intact.
        """
        buf = bytearray()
        self._sock.settimeout(_POLL)
        while len(buf) < n:
            try:
                chunk = self._sock.recv(n - len(buf))
            except socket.timeout:
                on_tick()
                if deadline is not None and time.monotonic() > deadline:
                    raise
                continue
            if not chunk:
                raise DaemonError("daemon closed the connection")
            buf += chunk
            on_tick()
        return bytes(buf)

    def _send_and_wait(self, action, payload, timeout) -> None:
        """Send one request and read its response into `self._resp`.

        On a host that draws its own progress bar the read polls, so this
        thread -- the one marimo's thread-local runtime context belongs to --
        can drive that bar between reads. Everywhere else the pump writes the
        bar to stderr and this just blocks.
        """
        native = (_runtime.progress_style() == "native"
                  and _runtime.show_progress())
        if not native:
            self._sock.settimeout(timeout)
            self._send({"action": action, "payload": payload or {}})
            (n,) = _HDR.unpack(self._recvn(8))
            self._resp = json.loads(self._recvn(n))
            return

        import queue as _queue
        q: "_queue.Queue[dict]" = _queue.Queue()
        self._progress_sink = q.put_nowait
        deadline = None if timeout is None else time.monotonic() + timeout

        import marimo as mo
        t_start = time.monotonic()
        drew = live = False
        done_bars: list[str] = []     # finished bars, kept above the live one
        cur_title = cur_body = None

        def tick():
            nonlocal drew, live, cur_title, cur_body
            last = None
            while True:
                try:
                    last = q.get_nowait()
                except _queue.Empty:
                    break
            if last is None:
                # Nothing from the sampler yet. Julia JIT-compiles the
                # likelihood for each target and data shape on its first call,
                # and the sampler has not started, so there is no percentage to
                # report -- measured on a small RV fit, 14.6s of a 26.4s cold
                # run, 55% of the wait. An indeterminate bar covers it.
                #
                # But only after _BAR_DELAY. EVERY call comes through here,
                # including ping and the feature ops, which compile nothing and
                # return in milliseconds: drawing immediately left a bar
                # spinning forever under `s.ping()`, announcing a compile that
                # was not happening. A delay fixes the whole class rather than
                # blacklisting one action, because "did this call take long
                # enough to be worth a bar" is the actual question.
                if not drew and time.monotonic() - t_start > _BAR_DELAY:
                    drew = True
                    mo.output.replace(mo.Html(_bars_html(done_bars + [_bar_html(
                        "nereus", None, [], "compiling the model for this data")])))
                return
            # A new title means a new PHASE -- sampling handed over to
            # plotting. Keep the finished bar; the indeterminate placeholder is
            # not a phase and is simply replaced.
            if cur_title is not None and last["title"] != cur_title:
                done_bars.append(cur_body)
            drew = live = True
            cur_title = last["title"]
            cur_body = _bar_html(last["title"], last["pct"], last["fields"])
            mo.output.replace(mo.Html(_bars_html(done_bars + [cur_body])))

        try:
            self._sock.settimeout(_POLL)
            self._send({"action": action, "payload": payload or {}})
            (n,) = _HDR.unpack(self._recvn_polling(8, deadline, tick))
            self._resp = json.loads(self._recvn_polling(n, deadline, tick))
        finally:
            self._progress_sink = None
            # A bar that only ever showed the indeterminate state belongs to a
            # call that never sampled; leaving it up would keep claiming a
            # compile after the call returned. One that reached a real
            # percentage is left at its final frame.
            if drew and not live:
                try:
                    mo.output.replace(mo.Html(""))
                except Exception:
                    pass

    def call(self, action: str, payload=None, timeout: float | None = None):
        with self._lock:
            # start() INSIDE the lock: two threads that both saw _sock is None
            # would otherwise each spawn a Julia process.
            if self._sock is None:
                self.start()
            # Echo Julia's output for the duration of the call. A fit prints a
            # live progress bar, and it is no use in a log file: the whole
            # point of a progress bar is to be seen while you wait. Forwarding
            # is off between calls so the idle daemon stays quiet.
            if self._forward is not None and _runtime.show_progress():
                self._forward.set()
            try:
                self._send_and_wait(action, payload, timeout)
                resp = self._resp
            except BaseException:
                # BaseException, NOT Exception: cancelling a notebook cell
                # raises KeyboardInterrupt, which is a BaseException and so
                # sailed straight past an `except Exception` without dropping
                # the socket. Julia -- which got the same SIGINT -- then wrote
                # its interrupt response into a connection nobody was reading,
                # and the NEXT fit sent its request and read that stale
                # response instead of its own. Cancel, re-run, DaemonError;
                # re-run again, fine, because by then the queue had shifted by
                # one. That is the "stop the cell, re-run, it fails once" bug.
                #
                # Framing is length-prefixed, so a half-read response leaves
                # the stream desynchronised — the next call would read the
                # tail of this one as an 8-byte header. There is no
                # resynchronisation point, so drop the connection; the next
                # call reconnects.
                #
                # NOTE: a timeout here does NOT stop the Julia job. It keeps
                # running in the daemon. Call stop() if you need it dead.
                self._drop()
                raise
            finally:
                if self._forward is not None:
                    self._forward.clear()
        if not resp.get("ok"):
            err = resp.get("error", "unknown error")
            bt = resp.get("backtrace", "")
            # An interrupted fit is not a bug in the model and its Julia
            # backtrace says nothing a caller can act on -- it just names
            # whichever kernel happened to be executing when the signal
            # landed. Say what happened in one line instead.
            if "InterruptException" in err or "InterruptException" in bt:
                raise FitInterrupted(
                    "the fit was cancelled before it finished, so nothing was "
                    "written to output_dir. The daemon is still up and warm -- "
                    "just call the fit again.")
            raise DaemonError(err + "\n" + bt)
        return resp.get("result")

    def _drop(self) -> None:
        """Close the socket and forget it, so the next call reconnects."""
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()
