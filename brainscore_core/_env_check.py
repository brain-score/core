"""Runtime check that installed dependency versions match UMI's pins.

Drift off these pins silently breaks scoring: ``transformers>=5`` removes
``DynamicCache.to_legacy_cache`` (the language KV-cache path AttributeErrors),
and ``scikit-learn>=1.6`` changes ``LogisticRegression(multi_class=...)``
semantics (behavioral-readout scores change without error). This module warns
loudly at import rather than letting a scored run be silently wrong.

Uses only ``importlib.metadata`` so ``brainscore_core`` stays free of heavy
dependencies (no torch/transformers import).
"""
from importlib.metadata import version, PackageNotFoundError

# (distribution name, ok(version_tuple) -> bool, human-readable expectation)
_PINS = [
    ("transformers", lambda v: v[0] < 5, "<5"),
    ("scikit-learn", lambda v: (v[0], v[1]) < (1, 6), "<1.6"),
    ("numpy", lambda v: v[0] < 2, "<2"),
    ("xarray", lambda v: (v[0], v[1], v[2]) == (2022, 3, 0), "==2022.3.0"),
]


def _parse(s):
    """Leading-numeric version tuple, padded to 3 fields. '5.5.0' -> (5, 5, 0)."""
    parts = []
    for field in s.split(".")[:3]:
        num = ""
        for ch in field:
            if ch.isdigit():
                num += ch
            else:
                break
        parts.append(int(num) if num else 0)
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts)


def check_env_pins():
    """Return a list of human-readable drift messages (empty if all pins hold)."""
    drift = []
    for pkg, ok, expected in _PINS:
        try:
            installed = version(pkg)
        except PackageNotFoundError:
            continue  # package not installed in this (e.g. core-only) env
        try:
            if not ok(_parse(installed)):
                drift.append(f"{pkg}=={installed} violates the UMI pin ({expected})")
        except Exception:
            continue  # unparseable version string: don't block import
    return drift


def warn_on_drift():
    """Emit a RuntimeWarning if any dependency pin is violated."""
    drift = check_env_pins()
    if drift:
        import warnings
        warnings.warn(
            "UMI dependency pins are violated by the installed environment:\n  - "
            + "\n  - ".join(drift)
            + "\nScoring may be SILENTLY WRONG (transformers>=5 breaks the language "
            "KV-cache path; scikit-learn>=1.6 changes behavioral-readout semantics). "
            "Install the pinned environment (environment-unified.yml) to fix.",
            RuntimeWarning,
            stacklevel=2,
        )
