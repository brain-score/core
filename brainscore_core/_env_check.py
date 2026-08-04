"""Runtime check that installed dependencies fall within UMI's compatibility bounds.

These are the version ranges scoring is *verified against* — deliberately looser
than any single install pin (that's the environment file's job). Drift outside
them silently breaks scoring: ``scikit-learn>=1.6`` changes
``LogisticRegression(multi_class=...)`` semantics, so behavioral-readout scores
change without error. This module warns loudly at import rather than letting a
scored run be silently wrong.

``transformers`` 5 is now supported. The two things that blocked it are handled:
``DynamicCache.to_legacy_cache`` was removed, so the KV sliding window selects a
cache API by shape (``slice_kv_cache``), and the image-processor class names were
rebound so the default moved from PIL to torchvision, which registrations pin
explicitly (``hf_compat.pin_image_processor``). Both were checked against 5.14:
GPT-2 features are bit-identical across the two majors, and the one model whose
processor default actually moved shifts its score by 0.038%.

Uses only ``importlib.metadata`` so ``brainscore_core`` stays free of heavy
dependencies (no torch/transformers import).
"""
from importlib.metadata import version, PackageNotFoundError

# (distribution name, ok(version_tuple) -> bool, human-readable bound)
_BOUNDS = [
    ("transformers", lambda v: (4, 57) <= (v[0], v[1]) < (6, 0), ">=4.57,<6"),
    ("scikit-learn", lambda v: (1, 5) <= (v[0], v[1]) < (1, 6), ">=1.5,<1.6"),
    ("numpy", lambda v: (1, 21) <= (v[0], v[1]) < (2, 0), ">=1.21,<2"),
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


def check_env_bounds():
    """Return human-readable drift messages (empty if all bounds hold)."""
    drift = []
    for pkg, ok, expected in _BOUNDS:
        try:
            installed = version(pkg)
        except PackageNotFoundError:
            continue  # package not installed in this (e.g. core-only) env
        try:
            if not ok(_parse(installed)):
                drift.append(
                    f"{pkg}=={installed} is outside the UMI runtime "
                    f"compatibility bound ({expected})")
        except Exception:
            continue  # unparseable version string: don't block import
    return drift


def warn_on_drift():
    """Emit a RuntimeWarning if any dependency is outside its compatibility bound."""
    drift = check_env_bounds()
    if drift:
        import warnings
        warnings.warn(
            "Installed dependencies are outside UMI's runtime compatibility "
            "bounds (these are the versions scoring is verified against; the exact "
            "install pin is the environment file's job):\n  - "
            + "\n  - ".join(drift)
            + "\nScoring may be SILENTLY WRONG (scikit-learn>=1.6 changes "
            "LogisticRegression semantics, so behavioral-readout scores move "
            "without erroring). Install the pinned environment "
            "(environment-unified.yml) to fix.",
            RuntimeWarning,
            stacklevel=2,
        )
