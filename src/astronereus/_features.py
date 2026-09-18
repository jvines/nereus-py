"""Non-inference features. Each is its own operation, not a mode of `fit`."""
from __future__ import annotations
from typing import Any, Literal, Sequence


class _Group:
    def __init__(self, session_getter, prefix: str, actions: dict[str, str]):
        self._get, self._prefix, self._actions = session_getter, prefix, actions

    def __dir__(self):
        return sorted(self._actions)

    def __getattr__(self, name):
        if name not in self._actions:
            raise AttributeError(f"{self._prefix}.{name} is not a Nereus "
                                 f"operation. Available: {', '.join(sorted(self._actions))}")
        action = self._actions[name]

        def _call(**payload):
            return self._get().raw(action, payload)
        _call.__name__ = name
        _call.__doc__ = f"Nereus operation {action!r}."
        return _call


DETECT = {"transits": "detect.transits", "rv_planets": "detect.rv_planets",
          "rotation": "detect.rotation", "segments": "detect.segments",
          "rv_periodogram": "detect.rv_periodogram",
          "lc_periodogram": "detect.lc_periodogram"}
DETREND = {"savgol": "detrend.savgol", "gp": "detrend.gp",
           "notch": "detrend.notch", "locor": "detrend.locor"}
TOMOGRAM = {"matched_filter": "tomogram.matched_filter",
            "null_distribution": "tomogram.null_distribution",
            "injection_test": "tomogram.injection_test",
            "shadow_track": "tomogram.shadow_track",
            "residuals": "tomogram.residuals",
            "ccf_profile": "tomogram.ccf_profile"}
DIAGNOSTICS = {"ess_rhat": "diagnostics.ess_rhat", "loo": "diagnostics.loo",
               "ppc": "diagnostics.ppc",
               "label_switching": "diagnostics.label_switching",
               "fit_health": "diagnostics.fit_health"}
