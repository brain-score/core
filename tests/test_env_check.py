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
    ok = {"transformers": "4.57.1", "scikit-learn": "1.7.2",
          "numpy": "1.26.4", "xarray": "2022.3.0"}
    monkeypatch.setattr(_env_check, "version", lambda p: ok[p])
    assert _env_check.check_env_bounds() == []


def test_pins_report_drift(monkeypatch):
    # transformers 5 is supported now, so 6 is the first version that drifts.
    # scikit-learn 1.8 is the first that drifts: it drops
    # LogisticRegression(multi_class=...), which moves binary readout scores.
    drifted = {"transformers": "6.0.0", "scikit-learn": "1.8.0",
               "numpy": "1.26.4", "xarray": "2022.3.0"}
    monkeypatch.setattr(_env_check, "version", lambda p: drifted[p])
    drift = _env_check.check_env_bounds()
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
    monkeypatch.setattr(_env_check, "version", lambda p: ok[p])
    assert _env_check.check_env_bounds() == []


def test_too_old_versions_also_report_drift(monkeypatch):
    old = {"transformers": "4.0.0", "scikit-learn": "1.0.0",
           "numpy": "1.26.4", "xarray": "2022.3.0"}
    monkeypatch.setattr(_env_check, "version", lambda p: old[p])
    drift = _env_check.check_env_bounds()
    assert any("transformers" in d for d in drift)
    assert any("scikit-learn" in d for d in drift)


def test_missing_package_is_skipped(monkeypatch):
    def _raise(pkg):
        from importlib.metadata import PackageNotFoundError
        raise PackageNotFoundError(pkg)
    monkeypatch.setattr(_env_check, "version", _raise)
    assert _env_check.check_env_bounds() == []  # core-only env: nothing to check


def test_transformers_just_below_shipped_floor_flagged(monkeypatch):
    v = {"transformers": "4.56.2", "scikit-learn": "1.7.2",
         "numpy": "1.26.4", "xarray": "2022.3.0"}
    monkeypatch.setattr(_env_check, "version", lambda p: v[p])
    assert any("transformers" in d for d in _env_check.check_env_bounds())
