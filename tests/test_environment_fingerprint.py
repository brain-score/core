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



# The locations the fingerprint reads: the sys.path entries known at start.

def _dist_info(folder, name, version):
    """A made-up installed distribution: ``<folder>/<name>-<version>.dist-info/METADATA``."""
    import shutil
    for old in folder.glob(f'{name}-*.dist-info'):
        shutil.rmtree(old)
    info = folder / f'{name}-{version}.dist-info'
    info.mkdir(parents=True)
    (info / 'METADATA').write_text(f'Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n')


def _startup(monkeypatch, pythonpath=(), sites=(), user=None, path=None):
    """Stand in for what the interpreter knew at start, and for its ``sys.path``."""
    import site
    pythonpath, sites = [str(p) for p in pythonpath], [str(p) for p in sites]
    monkeypatch.setattr(environment_check, '_STARTUP_PYTHONPATH', pythonpath)
    monkeypatch.setattr(site, 'getsitepackages', lambda: sites)
    monkeypatch.setattr(site, 'ENABLE_USER_SITE', user is not None)
    monkeypatch.setattr(site, 'getusersitepackages', lambda: str(user))
    order = path if path is not None else pythonpath + ([str(user)] if user else []) + sites
    monkeypatch.setattr(sys, 'path', [str(p) for p in order])
    environment_fingerprint.cache_clear()


_CORE = os.path.dirname(os.path.dirname(environment_check.__file__))


def _fresh_value(pythonpath, cwd=None, before='', env=None):
    """The fingerprint a fresh interpreter gives; ``before`` runs after the import, before the call."""
    import subprocess
    code = (f'from brainscore_core.environment_check import environment_fingerprint\n{before}\n'
            'print(environment_fingerprint())')
    environment = {**os.environ, **(env or {})}
    if pythonpath is not None:
        environment['PYTHONPATH'] = os.pathsep.join([*pythonpath, _CORE])
    out = subprocess.run([sys.executable, '-W', 'ignore', '-c', code], cwd=cwd, env=environment,
                         capture_output=True, text=True, check=True).stdout
    return out.strip().splitlines()[-1]


def test_the_value_does_not_change_when_sys_path_gains_a_folder_with_a_distribution(tmp_path, monkeypatch):
    before = environment_fingerprint()
    environment_fingerprint.cache_clear()
    _dist_info(tmp_path / 'added', 'made-up-added', '1.0')
    monkeypatch.syspath_prepend(str(tmp_path / 'added'))
    assert environment_fingerprint() == before


def test_a_vendor_folder_inside_a_package_is_never_read(tmp_path, monkeypatch):
    """As setuptools/_vendor joins sys.path when setuptools is imported."""
    site_packages = tmp_path / 'site-packages'
    _dist_info(site_packages, 'made-up-installed', '1.0')
    _dist_info(site_packages / 'tool' / '_vendor', 'made-up-vendored', '9.9')
    _startup(monkeypatch, sites=[site_packages])
    before = environment_fingerprint()
    environment_fingerprint.cache_clear()
    monkeypatch.setattr(sys, 'path', sys.path + [str(site_packages / 'tool' / '_vendor')])
    assert environment_fingerprint() == before
    assert environment_check._environment_locations() == [str(site_packages)]


def test_a_name_found_twice_counts_once_the_first_in_path_order(tmp_path, monkeypatch):
    first, second, alone = tmp_path / 'first', tmp_path / 'second', tmp_path / 'alone'
    _dist_info(first, 'made-up', '1.0')
    _dist_info(second, 'made-up', '2.0')
    _dist_info(alone, 'made-up', '1.0')
    values = {}
    for label, folders in (('both', [first, second]), ('first only', [alone]), ('second first', [second, first])):
        _startup(monkeypatch, pythonpath=folders)
        values[label] = environment_fingerprint()
    assert values['both'] == values['first only']           # the later 2.0 adds nothing
    assert values['second first'] != values['both']         # the first in order decides


def test_a_folder_on_pythonpath_with_a_distribution_is_another_environment(tmp_path):
    _dist_info(tmp_path / 'extlibs', 'made-up-extlib', '1.0')
    assert _fresh_value([str(tmp_path / 'extlibs')]) != _fresh_value([])


def test_a_relative_pythonpath_entry_keeps_the_folder_it_named_at_start(tmp_path):
    """Two processes with one environment; one changes directory before its first call."""
    _dist_info(tmp_path / 'a' / 'deps', 'fp-probe', '1.0')
    _dist_info(tmp_path / 'b' / 'deps', 'fp-probe', '2.0')
    moved = _fresh_value(['deps'], cwd=tmp_path / 'a', before=f'import os; os.chdir({str(tmp_path / "b")!r})')
    assert moved == _fresh_value(['deps'], cwd=tmp_path / 'a')


def test_the_folders_named_by_pth_files_of_a_site_folder_count(tmp_path, monkeypatch):
    site_packages, external = tmp_path / 'site-packages', tmp_path / 'external'
    site_packages.mkdir()
    (site_packages / 'external.pth').write_text(f'# a comment\n{external}\nimport os\n')
    values = []
    for version in ('1.0', '2.0'):
        _dist_info(external, 'fp-probe', version)
        _startup(monkeypatch, sites=[site_packages], path=[site_packages, external])
        values.append(environment_fingerprint())
    assert values[0] != values[1]


def test_an_import_line_of_a_pth_file_is_not_read(tmp_path):
    (tmp_path / 'one.pth').write_text(f'import sys; sys.path.append({str(tmp_path / "x")!r})\nimport\tos\nplain\n')
    assert environment_check._pth_folders(str(tmp_path)) == [str(tmp_path / 'plain')]


def test_the_order_of_sys_path_decides_a_venv_before_the_user_site(tmp_path, monkeypatch):
    """A virtual environment with system site packages: Python imports from the venv first."""
    venv, user, base = tmp_path / 'venv', tmp_path / 'user', tmp_path / 'base'
    _dist_info(user, 'fp-venv', '9.0')
    base.mkdir()
    values = []
    for version in ('1.0', '2.0'):
        _dist_info(venv, 'fp-venv', version)
        _startup(monkeypatch, sites=[venv, base], user=user, path=[venv, user, base])
        values.append(environment_fingerprint())
    assert values[0] != values[1]


def test_an_archive_on_pythonpath_is_read(tmp_path, monkeypatch):
    import zipfile
    values = []
    for version in ('1.0', '2.0'):
        egg = tmp_path / f'probe-{version}' / 'probe.egg'
        egg.parent.mkdir()
        with zipfile.ZipFile(egg, 'w') as archive:
            archive.writestr('fp_probe.py', 'VALUE = 1\n')
            archive.writestr(f'fp_probe-{version}.dist-info/METADATA',
                             f'Metadata-Version: 2.1\nName: fp-probe\nVersion: {version}\n')
        _startup(monkeypatch, pythonpath=[egg])
        values.append(environment_fingerprint())
    assert values[0] != values[1]


def test_a_module_with_no_metadata_that_shadows_a_distribution_is_not_seen(tmp_path, monkeypatch):
    """The limit of an identity made of distributions: the earlier loose fp_probe.py is what an
    import uses, but it has no metadata, so the value is that of the later distribution. Equal
    fingerprints are necessary for a bit-for-bit comparison, not a proof that every file is equal."""
    shadow, installed = tmp_path / 'shadow', tmp_path / 'installed'
    shadow.mkdir()
    (shadow / 'fp_probe.py').write_text('VALUE = "shadow"\n')
    _dist_info(installed, 'fp-probe', '1.0')
    _startup(monkeypatch, pythonpath=[installed])
    without = environment_fingerprint()
    _startup(monkeypatch, pythonpath=[shadow, installed])
    assert environment_fingerprint() == without


def test_the_doctor_a_score_and_another_process_give_one_value():
    value = environment_fingerprint()
    assert _score_environment()['software']['environment_fingerprint'] == value
    assert _fresh_value(None, before='import sys, json, xml.dom.minidom; sys.path.append("/nonexistent/vendor")') == value
