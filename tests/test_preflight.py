"""Storage failures should be detected before scoring consumes resources."""
from types import SimpleNamespace

import pytest

from brainscore_core.compatibility import ensure_legacy_benchmark_modalities
from brainscore_core.preflight import check_cache_directory


@pytest.mark.parametrize('kind', ['file', 'dangling_link'])
def test_invalid_cache_has_actionable_error(tmp_path, monkeypatch, kind):
    path = tmp_path / 'cache'
    if kind == 'file':
        path.write_text('not a directory')
    else:
        path.symlink_to(tmp_path / 'unmounted' / 'cache')
    monkeypatch.setenv('RESULTCACHING_DISABLE', '0')
    monkeypatch.setenv('RESULTCACHING_HOME', str(path))
    with pytest.raises(OSError, match='RESULTCACHING_HOME'):
        check_cache_directory()
    assert path.is_symlink() if kind == 'dangling_link' else path.is_file()


def test_cache_probe_leaves_no_files(tmp_path, monkeypatch):
    path = tmp_path / 'cache'
    monkeypatch.setenv('RESULTCACHING_HOME', str(path))
    monkeypatch.setenv('RESULTCACHING_DISABLE', '0')
    assert check_cache_directory() == path
    assert list(path.iterdir()) == []


def test_disabled_cache_does_not_touch_storage(tmp_path, monkeypatch):
    monkeypatch.setenv('RESULTCACHING_HOME', str(tmp_path / 'absent'))
    monkeypatch.setenv('RESULTCACHING_DISABLE', '1')
    assert check_cache_directory() is None
    assert not (tmp_path / 'absent').exists()


@pytest.mark.parametrize('declaration', [
    {'required_modalities': {'audio'}},
    {'accepted_modalities': {'audio', 'vision'}},
    {'required_input_channels': set()},
    {'required_input_channels': {'text'}},
])
def test_legacy_default_preserves_explicit_contract(declaration):
    benchmark = SimpleNamespace(**declaration)
    assert ensure_legacy_benchmark_modalities(benchmark, {'vision'}) is benchmark
    assert vars(benchmark) == declaration


@pytest.mark.parametrize('domain', ['vision', 'text'])
def test_legacy_default_supplies_domain(domain):
    benchmark = SimpleNamespace()
    ensure_legacy_benchmark_modalities(benchmark, {domain})
    assert benchmark.required_modalities == {domain}
