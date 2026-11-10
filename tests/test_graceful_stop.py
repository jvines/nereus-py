"""Stopping a fit without losing it: the client side. No Julia.

A Ctrl-C during a call must not drop the connection: the first one asks the
sampler to stop and keep its draws, and the call returns what Julia sends back;
only an error reply, or a second Ctrl-C, ends it as before. A fake daemon on a
socketpair plays Julia.
"""
from __future__ import annotations

import json
import os
import signal
import socket
import struct
import threading
import time

import pytest

from astronereus import engines
from astronereus._daemon import FitInterrupted, JuliaDaemon
from astronereus._fit import fit_joint, fit_rv
from astronereus.stopping import Stopping

_HDR = struct.Struct(">Q")


def _recv_msg(sock):
    buf = b""
    while len(buf) < 8:
        buf += sock.recv(8 - len(buf))
    (n,) = _HDR.unpack(buf)
    body = b""
    while len(body) < n:
        body += sock.recv(n - len(body))
    return json.loads(body)


def _send_msg(sock, obj):
    body = json.dumps(obj).encode()
    sock.sendall(_HDR.pack(len(body)) + body)


class FakeJulia:
    """Reads one request, waits for `release`, then answers with `reply`."""

    def __init__(self, sock, reply):
        self.sock = sock
        self.reply = reply
        self.got_request = threading.Event()
        self.release = threading.Event()
        self.request = None
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()

    def _run(self):
        try:
            self.request = _recv_msg(self.sock)
            self.got_request.set()
            self.release.wait(10)
            _send_msg(self.sock, self.reply)
        except OSError:
            pass


@pytest.fixture
def daemon():
    a, b = socket.socketpair()
    d = JuliaDaemon()
    d._sock = a            # connected: call() will not start a Julia process
    d._proc = None
    yield d, b
    for s in (a, b):
        try:
            s.close()
        except OSError:
            pass


def _sigint_later(fake, n=1, gap=0.15):
    """Deliver n SIGINTs to this process once the request is in, then release."""
    def go():
        fake.got_request.wait(5)
        for _ in range(n):
            os.kill(os.getpid(), signal.SIGINT)
            time.sleep(gap)
        if fake.reply is not None:
            fake.release.set()
    t = threading.Thread(target=go, daemon=True)
    t.start()
    return t


def test_first_ctrl_c_keeps_waiting_and_returns_the_graceful_result(daemon, capsys):
    d, peer = daemon
    result = {"status": "ok", "stopped": "interrupted", "stopped_at_step": 812,
              "n_steps": 3000, "n_draws_kept": 8120}
    fake = FakeJulia(peer, {"ok": True, "result": result})
    before = signal.getsignal(signal.SIGINT)
    _sigint_later(fake)
    with pytest.warns(RuntimeWarning, match=r"step 812/3000.*8120 draws"):
        out = d.call("fit_rv", {})
    assert out == result
    assert d._sock is not None                    # the connection survived
    assert signal.getsignal(signal.SIGINT) is before     # and the handler is restored
    assert "Ctrl-C again to abort" in capsys.readouterr().err


def test_an_interrupt_error_reply_is_still_fit_interrupted(daemon):
    d, peer = daemon
    fake = FakeJulia(peer, {"ok": False, "error": "InterruptException:",
                            "backtrace": ""})
    _sigint_later(fake)
    with pytest.raises(FitInterrupted, match="burn-in"):
        d.call("fit_rv", {})


def test_a_second_ctrl_c_aborts_and_drops_the_connection(daemon):
    d, peer = daemon
    fake = FakeJulia(peer, {"ok": True, "result": {}})
    _sigint_later(fake, n=2)
    fake.release.clear()
    t0 = time.monotonic()
    with pytest.raises(KeyboardInterrupt):
        d.call("fit_rv", {})
    assert time.monotonic() - t0 < 5              # did not wait for the reply
    assert d._sock is None                        # next call reconnects
    fake.release.set()


def test_a_call_without_ctrl_c_is_untouched(daemon):
    d, peer = daemon
    fake = FakeJulia(peer, {"ok": True, "result": {"stopped": "steps"}})
    fake.release.set()
    assert d.call("fit_rv", {}) == {"stopped": "steps"}


def test_a_graceful_result_without_ctrl_c_does_not_warn(daemon, recwarn):
    d, peer = daemon
    fake = FakeJulia(peer, {"ok": True, "result": {"stopped": "interrupted"}})
    fake.release.set()
    d.call("fit_rv", {})
    assert not [w for w in recwarn if issubclass(w.category, RuntimeWarning)]


def test_off_the_main_thread_the_old_behaviour_stands(daemon):
    d, peer = daemon
    fake = FakeJulia(peer, {"ok": True, "result": {"x": 1}})
    fake.release.set()
    box = {}
    t = threading.Thread(target=lambda: box.update(r=d.call("ping", {})))
    t.start(); t.join(5)
    assert box["r"] == {"x": 1}


# -- what crosses the wire ----------------------------------------------------

def test_stopping_serialises_only_what_is_set():
    assert Stopping().to_wire() == {}             # check_every no longer defaults to 500
    w = Stopping(max_seconds=3600, rhat_max=1.01, ess_min=400,
                 check_every=100).to_wire()
    assert w == {"max_seconds": 3600, "rhat_max": 1.01, "ess_min": 400,
                 "check_every": 100}


def test_transdim_engine_carries_the_convergence_options():
    e = engines.TransdimPTEmcee(convergence_stop=True, rhat_threshold=1.02,
                                tail_ess_threshold=500, n_converged_checks=2,
                                min_steps=800, diag_every=100,
                                science_params=["P_k1"], max_seconds=900,
                                max_evals=10**7, stop_file="/x/STOP")
    o = e.to_wire()
    assert o["engine"] == "transdim_pt_emcee"
    assert o["options"] == {
        "convergence_stop": True, "rhat_threshold": 1.02, "tail_ess_threshold": 500,
        "n_converged_checks": 2, "min_steps": 800, "diag_every": 100,
        "science_params": ["P_k1"], "max_seconds": 900, "max_evals": 10**7,
        "stop_file": "/x/STOP"}
    assert engines.TransdimPTEmcee().options() == {}
    p = engines.PTEmcee(max_seconds=60.0, max_evals=1000, stop_file="/x/STOP").options()
    assert p == {"max_seconds": 60.0, "max_evals": 1000, "stop_file": "/x/STOP"}


class Capture:
    def __init__(self):
        self.calls = []

    def raw(self, action, payload=None, **kw):
        self.calls.append((action, payload))
        return {"status": "ok"}


RV = {"HARPS": {"t": [0.0, 1.0, 2.0], "rv": [0.0, 1.0, 0.0], "rv_err": [1.0, 1.0, 1.0]}}


def test_fit_rv_sends_stopping_and_stop_file():
    s = Capture()
    fit_rv(RV, session=s, stopping=Stopping(max_seconds=60, rhat_max=1.02),
           stop_file="/work/out/STOP", output_dir="/work/out")
    (_, payload), = s.calls
    assert payload["stopping"] == {"max_seconds": 60, "rhat_max": 1.02}
    assert payload["stop_file"] == "/work/out/STOP"


def test_stop_file_is_left_out_when_not_given():
    s = Capture()
    fit_rv(RV, session=s)
    (_, payload), = s.calls
    assert "stop_file" not in payload and payload["stopping"] is None


def test_run_job_stop_file_goes_into_the_config():
    from astronereus._api import Session
    s = Session.__new__(Session)
    seen = {}
    s.raw = lambda action, payload=None, **kw: seen.update(a=action, p=payload) or {
        "status": "ok", "stopped": "requested"}
    r = s.run_job({"output_dir": "/o", "x": 1}, stop_file="/o/halt")
    assert seen["p"]["stop_file"] == "/o/halt" and seen["p"]["x"] == 1
    assert r.stopped == "requested"
    with pytest.raises(ValueError):
        s.run_job("job.json", stop_file="/o/halt")
