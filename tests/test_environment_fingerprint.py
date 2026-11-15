"""The environment fingerprint and the accelerator in a score's ``software``."""
import json
import os
import platform
import sys
from types import SimpleNamespace

import pytest

from brainscore_core import environment_check
from brainscore_core.environment_check import _accelerator, _score_environment, environment_fingerprint


class _Distribution:
    def __init__(self, name, version, direct_url=None):
        self.metadata, self.version, self._direct_url = {'Name': name}, version, direct_url

    def read_text(self, filename):
        return json.dumps(self._direct_url) if filename == 'direct_url.json' and self._direct_url else None

    def locate_file(self, path):
        return '/site-packages/' + path


def _with(monkeypatch, packages):
    import importlib.metadata
    monkeypatch.setattr(importlib.metadata, 'distributions',
                        lambda **kwargs: [p if isinstance(p, _Distribution) else _Distribution(*p) for p in packages])
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
                        lambda **kwargs: calls.append(1) or [_Distribution(*p) for p in BASE])
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


def _git(commit):
    return {'url': 'https://example.org/quickbrain.git', 'vcs_info': {'vcs': 'git', 'commit_id': commit}}


def test_another_commit_of_a_repository_install_gives_another_value(monkeypatch):
    _with(monkeypatch, BASE + [_Distribution('quickbrain', '0.1', _git('a' * 40))])
    first = environment_fingerprint()
    _with(monkeypatch, BASE + [_Distribution('quickbrain', '0.1', _git('b' * 40))])
    assert environment_fingerprint() != first


def test_another_location_of_an_editable_install_gives_another_value(monkeypatch):
    editable = lambda url: {'url': url, 'dir_info': {'editable': True}}
    _with(monkeypatch, BASE + [_Distribution('helper', '1.0', editable('file:///one/helper'))])
    first = environment_fingerprint()
    _with(monkeypatch, BASE + [_Distribution('helper', '1.0', editable('file:///two/helper'))])
    assert environment_fingerprint() != first


def test_a_wheel_file_records_no_source(monkeypatch):
    _with(monkeypatch, BASE + [_Distribution('helper', '1.0')])
    first = environment_fingerprint()
    _with(monkeypatch, BASE + [_Distribution('helper', '1.0', {'url': 'file:///w.whl', 'archive_info': {}})])
    assert environment_fingerprint() == first


@pytest.mark.parametrize('name, value', [('python_version', '0.0.1'), ('system', 'TestSystem'), ('machine', 'test-machine')])
def test_the_python_version_and_the_platform_count(monkeypatch, name, value):
    _with(monkeypatch, BASE)
    first = environment_fingerprint()
    monkeypatch.setattr(platform, name, lambda: value)
    environment_fingerprint.cache_clear()
    assert environment_fingerprint() != first


class _Broken(_Distribution):
    @property
    def metadata(self):
        raise ValueError('bad METADATA file')

    @metadata.setter
    def metadata(self, value):
        pass


def test_metadata_that_cannot_be_read_gives_an_unavailable_value(monkeypatch):
    _with(monkeypatch, BASE + [_Broken('broken', '1.0')])
    value = environment_fingerprint()
    assert value == 'unavailable: the metadata of a distribution cannot be read (ValueError: bad METADATA file)'
    assert _score_environment()['software']['environment_fingerprint'] == value      # the stamp keeps running


def test_a_distribution_without_a_name_gives_an_unavailable_value(monkeypatch):
    _with(monkeypatch, BASE + [_Distribution(None, '1.0')])
    assert environment_fingerprint() == 'unavailable: a distribution has no name (at /site-packages/)'



# The folders the fingerprint reads: those known when the interpreter starts (repair after the cloud check).

def _dist_info(folder, name, version):
    """A made-up installed distribution: ``<folder>/<name>-<version>.dist-info/METADATA``."""
    info = folder / f'{name}-{version}.dist-info'
    info.mkdir(parents=True)
    (info / 'METADATA').write_text(f'Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n')


def test_the_value_does_not_change_when_sys_path_gains_a_folder_with_a_distribution(tmp_path, monkeypatch):
    before = environment_fingerprint()
    environment_fingerprint.cache_clear()
    _dist_info(tmp_path / 'added', 'made-up-added', '1.0')
    monkeypatch.syspath_prepend(str(tmp_path / 'added'))
    assert environment_fingerprint() == before


def test_a_vendor_folder_inside_a_package_is_never_read(tmp_path, monkeypatch):
    """As setuptools/_vendor joins sys.path when setuptools is imported."""
    from brainscore_core.environment_check import _environment_folders
    site_packages = tmp_path / 'site-packages'
    _dist_info(site_packages, 'made-up-installed', '1.0')
    _dist_info(site_packages / 'tool' / '_vendor', 'made-up-vendored', '9.9')
    monkeypatch.setattr(environment_check, '_environment_folders', lambda: [str(site_packages)])
    before = environment_fingerprint()
    environment_fingerprint.cache_clear()
    monkeypatch.setattr(sys, 'path', sys.path + [str(site_packages / 'tool' / '_vendor')])
    assert environment_fingerprint() == before
    assert str(site_packages / 'tool' / '_vendor') not in _environment_folders()


def test_a_name_found_twice_counts_once_the_first_in_path_order(tmp_path, monkeypatch):
    first, second = tmp_path / 'first', tmp_path / 'second'
    _dist_info(first, 'made-up', '1.0')
    _dist_info(second, 'made-up', '2.0')
    alone_first = tmp_path / 'alone'
    _dist_info(alone_first, 'made-up', '1.0')
    values = {}
    for label, folders in (('both', [first, second]), ('first only', [alone_first]),
                           ('second first', [second, first])):
        monkeypatch.setattr(environment_check, '_environment_folders', lambda f=folders: [str(x) for x in f])
        environment_fingerprint.cache_clear()
        values[label] = environment_fingerprint()
    assert values['both'] == values['first only']           # the later 2.0 adds nothing
    assert values['second first'] != values['both']         # the first in order decides


def test_a_folder_on_pythonpath_with_a_distribution_is_another_environment(tmp_path, monkeypatch):
    from brainscore_core.environment_check import _environment_folders
    before = environment_fingerprint()
    _dist_info(tmp_path / 'extlibs', 'made-up-extlib', '1.0')
    monkeypatch.setenv('PYTHONPATH', os.pathsep.join([str(tmp_path / 'extlibs'), os.environ.get('PYTHONPATH', '')]))
    environment_fingerprint.cache_clear()
    assert _environment_folders()[0] == str(tmp_path / 'extlibs')
    assert environment_fingerprint() != before


def test_the_doctor_a_score_and_another_process_give_one_value():
    import subprocess
    value = environment_fingerprint()
    assert _score_environment()['software']['environment_fingerprint'] == value
    other = subprocess.run(
        [sys.executable, '-W', 'ignore', '-c',
         'import sys; sys.path.append("/nonexistent/vendor"); import json, xml.dom.minidom; '
         'from brainscore_core.environment_check import environment_fingerprint; print(environment_fingerprint())'],
        capture_output=True, text=True, env=dict(os.environ), check=True)
    assert other.stdout.strip().splitlines()[-1] == value
