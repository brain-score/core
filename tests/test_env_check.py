"""Tests for the dependency-pin drift check (_env_check)."""
from brainscore_core import _env_check


def test_parse_leading_numeric():
    assert _env_check._parse("5.5.0") == (5, 5, 0)
    assert _env_check._parse("1.7.2") == (1, 7, 2)
    assert _env_check._parse("2022.3.0") == (2022, 3, 0)
    assert _env_check._parse("4.57.1") == (4, 57, 1)
    # trailing non-numeric / short strings pad safely
    assert _env_check._parse("2") == (2, 0, 0)
    assert _env_check._parse("1.26.4rc1") == (1, 26, 4)


def test_pins_hold_at_targets(monkeypatch):
    ok = {"transformers": "4.57.1", "scikit-learn": "1.5.2",
          "numpy": "1.26.4", "xarray": "2022.3.0"}
    monkeypatch.setattr(_env_check, "version", lambda p: ok[p])
    assert _env_check.check_env_pins() == []


def test_pins_report_drift(monkeypatch):
    drifted = {"transformers": "5.5.0", "scikit-learn": "1.7.2",
               "numpy": "1.26.4", "xarray": "2022.3.0"}
    monkeypatch.setattr(_env_check, "version", lambda p: drifted[p])
    drift = _env_check.check_env_pins()
    assert any("transformers" in d for d in drift)
    assert any("scikit-learn" in d for d in drift)
    assert len(drift) == 2  # numpy + xarray are fine


def test_too_old_versions_also_report_drift(monkeypatch):
    old = {"transformers": "4.0.0", "scikit-learn": "1.0.0",
           "numpy": "1.26.4", "xarray": "2022.3.0"}
    monkeypatch.setattr(_env_check, "version", lambda p: old[p])
    drift = _env_check.check_env_pins()
    assert any("transformers" in d for d in drift)
    assert any("scikit-learn" in d for d in drift)


def test_missing_package_is_skipped(monkeypatch):
    def _raise(pkg):
        from importlib.metadata import PackageNotFoundError
        raise PackageNotFoundError(pkg)
    monkeypatch.setattr(_env_check, "version", _raise)
    assert _env_check.check_env_pins() == []  # core-only env: nothing to check
