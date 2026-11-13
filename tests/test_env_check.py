"""Tests for the dependency-pin drift check."""
from brainscore_core import environment_check


def test_parse_leading_numeric():
    assert environment_check._parse("5.5.0") == (5, 5, 0)
    assert environment_check._parse("1.7.2") == (1, 7, 2)
    assert environment_check._parse("2022.3.0") == (2022, 3, 0)
    assert environment_check._parse("4.57.1") == (4, 57, 1)
    # trailing non-numeric / short strings pad safely
    assert environment_check._parse("2") == (2, 0, 0)
    assert environment_check._parse("1.26.4rc1") == (1, 26, 4)


def test_pins_hold_at_targets(monkeypatch):
    ok = {"transformers": "4.57.1", "scikit-learn": "1.7.2",
          "numpy": "1.26.4", "xarray": "2022.3.0"}
    monkeypatch.setattr(environment_check, "version", lambda p: ok[p])
    assert environment_check.check_env_bounds() == []


def test_pins_report_drift(monkeypatch):
    # transformers 5 is supported now, so 6 is the first version that drifts.
    # scikit-learn 1.8 is the first that drifts: it drops
    # LogisticRegression(multi_class=...), which moves binary readout scores.
    drifted = {"transformers": "6.0.0", "scikit-learn": "1.8.0",
               "numpy": "1.26.4", "xarray": "2022.3.0"}
    monkeypatch.setattr(environment_check, "version", lambda p: drifted[p])
    drift = environment_check.check_env_bounds()
    assert any("transformers" in d for d in drift)
    assert any("scikit-learn" in d for d in drift)
    assert len(drift) == 2  # numpy + xarray are fine


def test_transformers_5_is_within_bounds(monkeypatch):
    """The KV-cache and image-processor changes in 5 are handled, so it passes.

    Guards the bound itself: this was <5 until the sliding window stopped
    depending on to_legacy_cache and registrations pinned their image processor.
    """
    ok = {"transformers": "5.14.1", "scikit-learn": "1.7.2",
          "numpy": "1.26.4", "xarray": "2022.3.0"}
    monkeypatch.setattr(environment_check, "version", lambda p: ok[p])
    assert environment_check.check_env_bounds() == []


def test_too_old_versions_also_report_drift(monkeypatch):
    old = {"transformers": "4.0.0", "scikit-learn": "1.0.0",
           "numpy": "1.26.4", "xarray": "2022.3.0"}
    monkeypatch.setattr(environment_check, "version", lambda p: old[p])
    drift = environment_check.check_env_bounds()
    assert any("transformers" in d for d in drift)
    assert any("scikit-learn" in d for d in drift)


def test_missing_package_is_skipped(monkeypatch):
    def _raise(pkg):
        from importlib.metadata import PackageNotFoundError
        raise PackageNotFoundError(pkg)
    monkeypatch.setattr(environment_check, "version", _raise)
    assert environment_check.check_env_bounds() == []  # core-only env: nothing to check


def test_transformers_just_below_shipped_floor_flagged(monkeypatch):
    v = {"transformers": "4.56.2", "scikit-learn": "1.7.2",
         "numpy": "1.26.4", "xarray": "2022.3.0"}
    monkeypatch.setattr(environment_check, "version", lambda p: v[p])
    assert any("transformers" in d for d in environment_check.check_env_bounds())


def test_describe_env_reports_passing_absent_and_drifted_rows(monkeypatch):
    from importlib.metadata import PackageNotFoundError
    from brainscore_core import environment_check
    v = {"transformers": "4.57.1", "numpy": "2.0.0", "xarray": "2022.3.0"}

    def _version(package):
        if package not in v:
            raise PackageNotFoundError(package)
        return v[package]
    monkeypatch.setattr(environment_check, "version", _version)
    assert environment_check.describe_env() == [
        ("transformers", "4.57.1", ">=4.57,<6", True),
        ("scikit-learn", None, ">=1.7,<1.8", None),
        ("numpy", "2.0.0", ">=1.21,<2", False),
        ("xarray", "2022.3.0", "==2022.3.0", True),
    ]


def test_warn_on_drift_warns_with_the_drift(monkeypatch):
    import pytest
    from brainscore_core import environment_check
    monkeypatch.setattr(environment_check, "check_env_bounds",
                        lambda: ["numpy==2.0.0 is outside"])
    with pytest.warns(RuntimeWarning, match="numpy==2.0.0 is outside"):
        environment_check.warn_on_drift()


def test_environment_specs_module_is_separate():
    """brainscore_core.environment keeps observation/action specs only."""
    from brainscore_core import environment, environment_check
    assert environment is not environment_check
    assert not hasattr(environment, "describe_env")


def test_core_imports_without_torch_or_transformers():
    """Startup runs the drift check without importing heavy dependencies."""
    import subprocess
    import sys
    code = (
        "import sys\n"
        "class _Block:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name.split('.')[0] in ('torch', 'transformers'):\n"
        "            raise ImportError('blocked ' + name)\n"
        "sys.meta_path.insert(0, _Block())\n"
        "import brainscore_core, brainscore_core.environment_check\n"
        "import brainscore_core.streaming_helpers\n"
        "assert 'torch' not in sys.modules and 'transformers' not in sys.modules\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
