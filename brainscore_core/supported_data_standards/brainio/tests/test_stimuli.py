import os

import imageio
import numpy as np
import pandas as pd
import pytest

from brainscore_core.supported_data_standards import brainio
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet
from brainscore_core.supported_data_standards.brainio.tests.conftest import get_csv_path, get_dir_path
from brainscore_core.supported_data_standards.brainio.s3 import load_stimulus_set_from_s3, load_assembly_from_s3

MJ2015_STIM = load_stimulus_set_from_s3(identifier="hvm-public",bucket="brainscore-storage/brainio-brainscore",
                                              csv_sha1="5ca7a3da00d8e9c694a9cd725df5ba0ad6d735af",
                                              zip_sha1="8aa44e038d7b551efa8077467622f9d48d72e473",
                                              csv_version_id="null",
                                              zip_version_id="null"
                                            )   


class TestPreservation:
    def test_subselection(self):
        stimulus_set = StimulusSet([{'stimulus_id': i} for i in range(100)])
        stimulus_set.stimulus_paths = {i: f'/dummy/path/{i}' for i in range(100)}
        stimulus_set = stimulus_set[stimulus_set['stimulus_id'].isin(stimulus_set['stimulus_id'].values[:3])]
        assert stimulus_set.get_stimulus(0) is not None

    def test_pd_concat(self):
        s1 = StimulusSet([{'stimulus_id': i} for i in range(10)])
        s1.stimulus_paths = {i: f'/dummy/path/{i}' for i in range(10)}
        s2 = StimulusSet([{'stimulus_id': i} for i in range(10, 20)])
        s2.stimulus_paths = {i: f'/dummy/path/{i}' for i in range(10, 20)}
        s = pd.concat((s1, s2))
        s.stimulus_paths = {**s1.stimulus_paths, **s2.stimulus_paths}
        assert s.get_stimulus(1) is not None
        assert s.get_stimulus(11) is not None


def test_get_stimulus_set(brainio_home):
    stimulus_set = MJ2015_STIM                           
    assert "image_id" in stimulus_set.columns
    assert set(stimulus_set.columns).issuperset({'image_id', 'object_name', 'variation', 'category_name',
                                                 'image_file_name', 'background_id', 'ty', 'tz',
                                                 'size', 'id', 's', 'rxz', 'ryz', 'ryz_semantic',
                                                 'rxy', 'rxy_semantic', 'rxz_semantic'})
    assert len(stimulus_set) == 3200
    assert stimulus_set.identifier == 'hvm-public'
    for stimulus_id in stimulus_set['image_id']:
        stimulus_path = stimulus_set.get_stimulus(stimulus_id)
        assert os.path.exists(stimulus_path)
        extension = os.path.splitext(stimulus_path)[1]
        assert extension in ['.png', '.PNG', '.jpg', '.jpeg', '.JPG', '.JPEG']


def test_loadname_dicarlo_hvm(brainio_home_session):
    assert MJ2015_STIM is not None


class TestLoadImage:
    def test_dicarlohvm(self, brainio_home_session):
        stimulus_set = MJ2015_STIM
        paths = stimulus_set.stimulus_paths.values()
        for path in paths:
            image = imageio.imread(path)
            assert isinstance(image, np.ndarray)
            assert image.size > 0


@pytest.mark.parametrize('stimulus_set_identifier,bucket,csv_sha1,zip_sha1', [
    pytest.param('hvm-public', "brainscore-storage/brainio-brainscore", "5ca7a3da00d8e9c694a9cd725df5ba0ad6d735af", "8aa44e038d7b551efa8077467622f9d48d72e473", marks=[]),
    pytest.param('Ferguson2024_circle_line', "brainscore-storage/brainio-brainscore", "fc59d23ccfb41b4f98cf02865fc335439d2ad222", "1f0065910b01a1a0e12611fe61252eafb9c534c3", marks=[pytest.mark.private_access]),
])
def test_existence(stimulus_set_identifier, bucket, csv_sha1, zip_sha1):
    stimulus_set = load_stimulus_set_from_s3(identifier=stimulus_set_identifier,bucket=bucket,
                                              csv_sha1=csv_sha1,
                                              zip_sha1=zip_sha1,
                                              csv_version_id="null", # hardcoded null 
                                              zip_version_id="null"  # hardcoded null 
                                            )   
    assert stimulus_set is not None


def test_from_files():
    s = brainio.stimuli.StimulusSet.from_files(get_csv_path(), get_dir_path())
    assert "stimulus_id" in s.columns
    return s


class TestFromFiles:
    def test_basic(self):
        test_from_files()

