"""The environment fingerprint and the accelerator in a score's ``software``."""
import sys
from types import SimpleNamespace

import pytest

from brainscore_core import environment_check
from brainscore_core.environment_check import _accelerator, _score_environment, environment_fingerprint


class _Distribution:
    def __init__(self, name, version):
        self.metadata, self.version = {'Name': name}, version


def _with(monkeypatch, packages):
    import importlib.metadata
    monkeypatch.setattr(importlib.metadata, 'distributions',
                        lambda: [_Distribution(name, version) for name, version in packages])
    environment_fingerprint.cache_clear()


@pytest.fixture(autouse=True)
def _fresh():
    environment_fingerprint.cache_clear()
    yield
    environment_fingerprint.cache_clear()


BASE = [('numpy', '1.26.4'), ('scikit-learn', '1.7.2'), ('torch', '2.6.0+cu124')]


def test_the_same_packages_give_the_same_value_in_any_order(monkeypatch):
    _with(monkeypatch, BASE)
    first = environment_fingerprint()
    _with(monkeypatch, list(reversed(BASE)))
    assert environment_fingerprint() == first
    assert len(first) == 64 and all(c in '0123456789abcdef' for c in first)


def test_another_version_gives_another_value(monkeypatch):
    _with(monkeypatch, BASE)
    first = environment_fingerprint()
    _with(monkeypatch, [('numpy', '1.26.3')] + BASE[1:])
    assert environment_fingerprint() != first


def test_an_added_package_gives_another_value(monkeypatch):
    _with(monkeypatch, BASE)
    first = environment_fingerprint()
    _with(monkeypatch, BASE + [('pandas', '2.3.3')])
    assert environment_fingerprint() != first


def test_names_are_normalized(monkeypatch):
    _with(monkeypatch, BASE)
    first = environment_fingerprint()
    _with(monkeypatch, [('NumPy', '1.26.4'), ('scikit_learn', '1.7.2'), ('torch', '2.6.0+cu124')])
    assert environment_fingerprint() == first


def test_the_brainscore_packages_do_not_change_it(monkeypatch):
    _with(monkeypatch, BASE)
    first = environment_fingerprint()
    _with(monkeypatch, BASE + [('brainscore-core', '9.9'), ('brainscore', '0.1'),
                               ('brainscore_vision', '1'), ('brainscore-language', '2')])
    assert environment_fingerprint() == first


def test_it_is_computed_once_per_process(monkeypatch):
    calls = []
    import importlib.metadata
    monkeypatch.setattr(importlib.metadata, 'distributions',
                        lambda: calls.append(1) or [_Distribution(*p) for p in BASE])
    environment_fingerprint.cache_clear()
    environment_fingerprint(), environment_fingerprint(), _score_environment()
    assert calls == [1]


def test_the_score_software_carries_the_fingerprint_and_the_accelerator():
    software = _score_environment()['software']
    assert software['environment_fingerprint'] == environment_fingerprint()
    assert software['accelerator'] == _accelerator()


def test_without_torch_imported_the_accelerator_is_cpu(monkeypatch):
    monkeypatch.delitem(sys.modules, 'torch', raising=False)
    assert _accelerator() == 'cpu'


def test_a_cuda_device_is_named(monkeypatch):
    cuda = SimpleNamespace(is_available=lambda: True, current_device=lambda: 0,
                           get_device_name=lambda index: 'NVIDIA A10G')
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(cuda=cuda))
    assert _accelerator() == 'NVIDIA A10G'


def test_mps_is_named(monkeypatch):
    torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False),
                            backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True)))
    monkeypatch.setitem(sys.modules, 'torch', torch)
    assert _accelerator() == 'mps'


def test_a_failing_query_is_not_stated(monkeypatch):
    def fail():
        raise RuntimeError('driver')
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(cuda=SimpleNamespace(is_available=fail)))
    assert _accelerator() == 'not stated'
