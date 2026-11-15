"""Runtime check that installed dependencies fall within UMI's compatibility bounds.

Public API: ``check_env_bounds`` (drift messages only), ``describe_env`` (one
row per checked package, passing or not) and ``warn_on_drift`` (the warning
``import brainscore_core`` emits). ``python -m brainscore.doctor`` prints
``describe_env`` as a table.

These are the version ranges scoring is *verified against* — deliberately looser
than any single install pin (that's the environment file's job). Drift outside
them silently breaks scoring. This module warns loudly at import rather than
letting a scored run be silently wrong.

``scikit-learn`` moved from 1.5 to 1.7.2 on 2026-09-10, matching the version
``language/environment_lock.yml`` already locked. The 1.5 pin had made four
registered benchmarks unrunnable — ``Pereira2018.{243,384}sentences-ridge``,
``Blank2014-ridge`` and ``Fedorenko2016-ridge`` all route through
``kfold='group'``, and upstream calls ``GroupKFold(shuffle=..., random_state=...)``,
parameters that only exist from 1.6.

The bump is score-neutral, which was checked rather than assumed:
``LogisticRegression(multi_class='multinomial')`` is still honoured in 1.7.2 and
returns bit-identical probabilities to 1.5.1 for both binary and multiclass
fits. sklearn's own warning says removal was planned for 1.7; it actually
slipped to **1.8**, which is why the upper bound is ``<1.8``. At 1.8 the
argument disappears and binary problems become proper binary logistic fits
(as if ``multi_class='ovr'``), which does move probabilities — measured on a
seeded 400x30 binary fit, mid-range values shift from 0.0019 to 0.0076.
Multiclass is unaffected. Crossing into 1.8 therefore needs the behavioral
readouts re-measured, and is a deliberate step rather than a version bump.

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
import platform
import sys
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# (distribution name, ok(version_tuple) -> bool, human-readable bound)
_BOUNDS = [
    ("transformers", lambda v: (4, 57) <= (v[0], v[1]) < (6, 0), ">=4.57,<6"),
    ("scikit-learn", lambda v: (1, 7) <= (v[0], v[1]) < (1, 8), ">=1.7,<1.8"),
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


def check_env_bounds() -> List[str]:
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


def warn_on_drift() -> None:
    """Emit a RuntimeWarning if any dependency is outside its compatibility bound."""
    drift = check_env_bounds()
    if drift:
        import warnings
        warnings.warn(
            "Installed dependencies are outside UMI's runtime compatibility "
            "bounds (these are the versions scoring is verified against; the exact "
            "install pin is the environment file's job):\n  - "
            + "\n  - ".join(drift)
            + "\nScoring may be SILENTLY WRONG: for example, scikit-learn 1.8 drops "
            "LogisticRegression(multi_class=...), so binary behavioral-readout "
            "scores move without erroring. To fix, install with the pinned "
            "versions: run unified/install/setup.sh, or pip install with "
            "-c unified/install/v2-constraints.txt (see the unified README).",
            RuntimeWarning,
            stacklevel=2,
        )


def describe_env() -> List[Tuple[str, Optional[str], str, Optional[bool]]]:
    """Per-dependency status, including the ones that are fine.

    ``check_env_bounds`` reports only problems, which is what a warning needs.
    A user asking "is my environment right?" needs to see the passing rows too,
    otherwise silence is indistinguishable from nothing having been checked.

    :return: rows of ``(package, installed or None, expected, ok)`` where ``ok``
        is None for a package that is absent or has an unparseable version
    """
    rows = []
    for pkg, ok, expected in _BOUNDS:
        try:
            installed = version(pkg)
        except PackageNotFoundError:
            rows.append((pkg, None, expected, None))
            continue
        try:
            rows.append((pkg, installed, expected, bool(ok(_parse(installed)))))
        except Exception:
            rows.append((pkg, installed, expected, None))
    return rows


# The libraries whose versions can change the numbers of a score, and the four
# Brain-Score packages. Every key of ``software`` is the installed distribution
# name, as the package metadata spells it ("scikit-learn", "brainscore-core",
# "brainscore_vision"), never the import name. Read from the installed metadata:
# nothing is imported, so torch is never loaded here.
_SCORE_LIBRARIES = ("numpy", "torch", "transformers", "scikit-learn", "xarray")
_BRAINSCORE_PACKAGES = (("brainscore_core", "brainscore-core"), ("brainscore", "brainscore"),
                        ("brainscore_vision", "brainscore_vision"),
                        ("brainscore_language", "brainscore_language"))


def _installed(distribution: str) -> Optional[str]:
    """The installed version, or None for any failure (absent, or a broken ``.dist-info``)."""
    try:
        return version(distribution)
    except Exception:
        return None


def _project_folder(directory: Path) -> Optional[Path]:
    """The folder that holds the package's ``pyproject.toml`` or ``setup.py``: the package folder or its parent."""
    for folder in (directory, directory.parent):
        if (folder / "pyproject.toml").is_file() or (folder / "setup.py").is_file():
            return folder
    return None


def _git_commit(directory: Path) -> Optional[str]:
    """The commit of the checkout that holds the package in ``directory``; None if there is none.

    Only the package's own project folder is looked at, never a folder above it:
    a package installed inside a virtual environment that sits in an unrelated
    repository must not report that repository's commit. A copy without
    ``.git`` (an archive) states its commit in a file ``COMMIT`` in the project
    folder, written when the archive is made (``git rev-parse HEAD > COMMIT``);
    only a 40-character hexadecimal commit counts. Any failure gives None, so a
    stamp never loses a finished score.
    """
    try:
        folder = _project_folder(Path(directory))
        if folder is None:
            return None
        dot_git = folder / ".git"
        if dot_git.is_file():           # a worktree: ".git" names the git folder, maybe relative
            target = Path(dot_git.read_text().split("gitdir:", 1)[-1].strip())
            git_dir = target if target.is_absolute() else (dot_git.parent / target).resolve()
        elif dot_git.is_dir():
            git_dir = dot_git
        else:
            stated = folder / "COMMIT"
            commit = stated.read_text().strip() if stated.is_file() else ""
            return commit if len(commit) == 40 and all(c in "0123456789abcdef" for c in commit) else None
        head = (git_dir / "HEAD").read_text().strip()
        if not head.startswith("ref:"):
            return head
        ref = head[4:].strip()
        common = git_dir / "commondir"
        roots = [git_dir] + ([(git_dir / common.read_text().strip()).resolve()]
                             if common.is_file() else [])
        for root in roots:
            if (root / ref).is_file():
                return (root / ref).read_text().strip()
            packed = root / "packed-refs"
            if packed.is_file():
                for line in packed.read_text().splitlines():
                    if line.endswith(" " + ref):
                        return line.split(" ", 1)[0]
    except Exception:
        return None
    return None


def _score_environment() -> Dict[str, Dict[str, Optional[str]]]:
    """``{'software': {name: version}, 'code_commits': {package: commit}}`` for a score's attrs.

    Versions come from the installed metadata (None when absent). A commit is
    read only for a Brain-Score package already imported, from the checkout its
    files are in (None outside a git checkout); the installed version of an
    editable package does not say which checkout runs.
    """
    software = {"python": platform.python_version()}
    software.update({name: _installed(name) for name in _SCORE_LIBRARIES})
    software.update({distribution: _installed(distribution)
                     for _module, distribution in _BRAINSCORE_PACKAGES})
    commits = {}
    for module_name, _distribution in _BRAINSCORE_PACKAGES:
        module = sys.modules.get(module_name)
        path = getattr(module, "__file__", None)
        commits[module_name] = _git_commit(Path(path).resolve().parent) if path else None
    return {"software": software, "code_commits": commits}
