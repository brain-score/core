from pathlib import Path
import os

import pytest
from unittest.mock import patch, MagicMock
from moto import mock_aws
import boto3
import numpy as np
from PIL import Image

from brainscore_core.supported_data_standards import brainio
from pandas import DataFrame

from brainscore_core.supported_data_standards.brainio import fetch
from brainscore_core.supported_data_standards.brainio.assemblies import DataAssembly, NeuroidAssembly, get_levels
from brainscore_core.supported_data_standards.brainio.s3 import load_assembly_from_s3, load_stimulus_set_from_s3
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet
from brainscore_core.supported_data_standards.brainio.packaging import write_netcdf, check_stimulus_numbers, check_stimulus_naming_convention, \
    package_stimulus_set, package_data_assembly, get_user_info, upload_to_s3
# Note: lookup module functionality is not available in brainio-lite (catalog system removed)
from brainscore_core.supported_data_standards.brainio.tests.conftest import make_stimulus_set_df, make_spk_assembly, make_meta_assembly, BUCKET_NAME


def test_write_netcdf(test_write_netcdf_path):
    assy = DataAssembly(
        data=[[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15], [16, 17, 18]],
        coords={
            'up': ("a", ['alpha', 'alpha', 'beta', 'beta', 'beta', 'beta']),
            'down': ("a", [1, 1, 1, 1, 2, 2]),
            'sideways': ('b', ['x', 'y', 'z'])
        },
        dims=['a', 'b']
    )
    netcdf_sha1 = write_netcdf(assy, str(test_write_netcdf_path))
    assert test_write_netcdf_path.exists()


def test_reset_index():
    assy = DataAssembly(
        data=[[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15], [16, 17, 18]],
        coords={
            'up': ("a", ['alpha', 'alpha', 'beta', 'beta', 'beta', 'beta']),
            'down': ("a", [1, 1, 1, 1, 2, 2]),
            'sideways': ('b', ['x', 'y', 'z'])
        },
        dims=['a', 'b']
    )
    assert assy["a"].variable.level_names == ["up", "down"]
    assert list(assy.indexes) == ["a", "b"]
    assy = assy.reset_index(list(assy.indexes))
    assert assy["a"].variable.level_names is None
    assert get_levels(assy) == []
    assert list(assy.indexes) == []


def test_reset_index_levels():
    assy = DataAssembly(
        data=[[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15], [16, 17, 18]],
        coords={
            'up': ("a", ['alpha', 'alpha', 'beta', 'beta', 'beta', 'beta']),
            'down': ("a", [1, 1, 1, 1, 2, 2]),
            'sideways': ('b', ['x', 'y', 'z'])
        },
        dims=['a', 'b']
    )
    assert assy["a"].variable.level_names == ["up", "down"]
    assy = assy.reset_index(["up", "down"])
    assert get_levels(assy) == []


def test_stimulus_numbers():
    stimulus_set = StimulusSet(DataFrame({'stimulus_id': [0, 1]}))
    filenames = ['Nat300_1.png', 'Nat300_2.png']
    assert len(stimulus_set) == len(filenames)
    stimulus_set.stimulus_paths = {stimulus_set.at[idx, 'stimulus_id']: filenames[idx]
                                   for idx in range(len(stimulus_set))}

    check_stimulus_numbers(stimulus_set)


def test_stimulus_naming_convention():
    for name in ['image_1.png', 'Nat300_100.png', '1.png']:
        check_stimulus_naming_convention(name)


@pytest.mark.private_access
def test_package_stimulus_set(test_stimulus_set_identifier, brainio_home, restore_this_file):
    stimulus_set = StimulusSet(make_stimulus_set_df())
    stimulus_set.stimulus_paths = {row["stimulus_id"]: Path(__file__).parent / f'images/{row["filename"]}'
                                   for _, row in stimulus_set.iterrows()}
    del stimulus_set["filename"]
    identifier = test_stimulus_set_identifier
    result = package_stimulus_set(None, stimulus_set, identifier, bucket_name=BUCKET_NAME)
    # Catalog functionality has been removed - just verify packaging returns result
    assert result['identifier'] == identifier
    assert 'csv_sha1' in result
    assert 'zip_sha1' in result
    # Note: catalog lookup functionality (list_stimulus_sets, get_stimulus_set) has been deprecated


@pytest.mark.private_access
def test_package_data_assembly(test_stimulus_set_identifier, brainio_home, restore_this_file):
    # catalog = lookup.get_catalog(test_catalog_identifier)
    # assert 'lookup_source' not in catalog
    # assert 'source_catalog' not in catalog
    stimulus_set = StimulusSet(make_stimulus_set_df())
    stimulus_set.stimulus_paths = {row["stimulus_id"]: Path(__file__).parent / f'images/{row["filename"]}'
                                   for _, row in stimulus_set.iterrows()}
    del stimulus_set["filename"]
    identifier = test_stimulus_set_identifier
    package_stimulus_set(None, stimulus_set, identifier, bucket_name=BUCKET_NAME)
    assy = DataAssembly(
        data=[[[1], [2], [3]], [[4], [5], [6]], [[7], [8], [9]], [[10], [11], [12]], [[13], [14], [15]], [[16], [17], [18]]],
        coords={
            'stimulus_id': ("presentation", ["n"+str(i) for i in range(6)]),
            'stimulus_type': ("presentation", ["foo"]*6),
            'neuroid_id': ("neuroid", list("ABC")),
            'neuroid_type': ("neuroid", ["bar"]*3),
            'time_bin_start': ('time_bin', [0]),
            'time_bin_end': ('time_bin', [10]),
        },
        dims=['presentation', 'neuroid', 'time_bin']
    )
    identifier = "test.package_assembly"
    result = package_data_assembly(
        None, assy, identifier, test_stimulus_set_identifier, "DataAssembly", BUCKET_NAME)
    # Catalog functionality has been removed - just verify packaging returns result
    assert result['identifier'] == identifier
    assert 'sha1' in result
    # Note: catalog lookup functionality (list_assemblies, get_assembly) has been deprecated


@pytest.mark.private_access
def test_package_extras(test_stimulus_set_identifier, brainio_home):
    stimulus_set = StimulusSet(make_stimulus_set_df())
    stimulus_set.stimulus_paths = {row["stimulus_id"]: Path(__file__).parent / f'images/{row["filename"]}'
                                   for _, row in stimulus_set.iterrows()}
    del stimulus_set["filename"]
    identifier = test_stimulus_set_identifier
    package_stimulus_set(None, stimulus_set, identifier, bucket_name=BUCKET_NAME)
    assy = make_spk_assembly()
    identifier = "test.package_assembly_extras"
    assy_extra = make_meta_assembly()
    assy_extra.name = "test"
    extras = {assy_extra.name: assy_extra}
    result = package_data_assembly(None, assy, identifier, test_stimulus_set_identifier,
                          "SpikeTimesAssembly", BUCKET_NAME, extras)
    # Catalog functionality has been removed - just verify packaging returns result
    assert result['identifier'] == identifier
    assert 'sha1' in result
    # Note: catalog lookup functionality (list_assemblies, get_assembly) has been deprecated


def test_compression(test_write_netcdf_path):
    write_netcdf(make_spk_assembly(6), test_write_netcdf_path, compress=False)
    uncompressed = test_write_netcdf_path.stat().st_size
    write_netcdf(make_spk_assembly(6), test_write_netcdf_path, compress=True)
    compressed = test_write_netcdf_path.stat().st_size
    assert uncompressed > compressed


@mock_aws
def test_get_user_info():
    client = boto3.client('sts', region_name='us-east-1')

    # mocking the get_caller_identity method on the client instance
    with patch('brainscore_core.supported_data_standards.brainio.packaging.boto3.client') as mock_boto_client:
        mock_boto_client.return_value = client

        with patch.object(client, 'get_caller_identity', return_value={
            'Account': '123456789012',
            'Arn': 'arn:aws:iam::123456789012:user/testuser',
            'UserId': 'ABCDEFGHIJKLMNO:testuser'
        }):
            account_id, username = get_user_info(client)
            assert account_id == '123456789012'
            assert username == 'testuser'


# @mock_aws
# @patch('brainio.packaging.tqdm')
# @patch('os.path.getsize', return_value=1000)
# def test_upload_to_s3(mock_getsize, mock_tqdm):
#     # set up mocks
#     mock_tqdm_instance = MagicMock()
#     mock_tqdm.return_value = mock_tqdm_instance
#     s3_client = boto3.client('s3', region_name='us-east-1')
#     sts_client = boto3.client('sts', region_name='us-east-1')

#     # crreate test bucket and file
#     s3_client.create_bucket(Bucket='test-bucket')
#     source_file_path = Path('files/testfile.txt')
#     bucket_name = 'test-bucket'
#     target_s3_key = 'testfile.txt'
#     os.makedirs(source_file_path.parent, exist_ok=True)
#     with open(source_file_path, 'w') as f:
#         f.write("dummy content")

#     # mock STS response
#     with patch.object(sts_client, 'get_caller_identity', return_value={
#         'Account': '123456789012',
#         'Arn': 'arn:aws:iam::123456789012:user/testuser',
#         'UserId': 'ABCDEFGHIJKLMNO:testuser'
#     }):
#         # patch get_user_info to return the expected user
#         with patch('brainio.packaging.get_user_info', return_value=('123456789012', 'testuser')):
#             object_properties = upload_to_s3(source_file_path, bucket_name, target_s3_key)

#             # after upload, check that tagging worked
#             tag_set = s3_client.get_object_tagging(Bucket=bucket_name, Key=target_s3_key)['TagSet']
#             expected_tag = {'Key': 'uploadedBy', 'Value': 'testuser'}
#             assert expected_tag in tag_set
#             mock_tqdm.assert_called_once_with(total=1000, unit='B', unit_scale=True, desc='upload to s3')

#     # clean up dummy file
#     os.remove(source_file_path)


@pytest.fixture
def s3_test_bucket(monkeypatch):
    for key, value in {"AWS_ACCESS_KEY_ID": "testing", "AWS_SECRET_ACCESS_KEY": "testing",
                       "AWS_SESSION_TOKEN": "testing", "AWS_DEFAULT_REGION": "us-east-1"}.items():
        monkeypatch.setenv(key, value)
    with mock_aws():
        client = boto3.client("s3")
        client.create_bucket(Bucket="test-bucket")
        yield client


@pytest.fixture
def upload_source_file(tmp_path):
    path = tmp_path / "stimulus_dummy.csv"
    path.write_text("stimulus_id,filename\n0,0.png\n")
    return path


def test_upload_to_bucket(s3_test_bucket, upload_source_file):
    upload_to_s3(upload_source_file, "test-bucket", target_s3_key="stimulus_dummy.csv")
    keys = [obj["Key"] for obj in s3_test_bucket.list_objects_v2(Bucket="test-bucket")["Contents"]]
    assert keys == ["stimulus_dummy.csv"]


@pytest.mark.parametrize("bucket, expected_key", [
    ("test-bucket/some/folder", "some/folder/stimulus_dummy.csv"),
    ("test-bucket/some/folder/", "some/folder/stimulus_dummy.csv"),
    ("test-bucket//some//folder", "some/folder/stimulus_dummy.csv"),
    ("test-bucket/", "stimulus_dummy.csv"),
])
def test_upload_to_bucket_folder(s3_test_bucket, upload_source_file, bucket, expected_key):
    # `load_stimulus_set_from_s3` and `load_assembly_from_s3` accept "<bucket>/<folder>", so uploads should too
    properties = upload_to_s3(upload_source_file, bucket, target_s3_key="stimulus_dummy.csv")
    keys = [obj["Key"] for obj in s3_test_bucket.list_objects_v2(Bucket="test-bucket")["Contents"]]
    assert keys == [expected_key]
    assert properties["ContentLength"] == upload_source_file.stat().st_size


@pytest.mark.parametrize("bucket", ["test-bucket", "test-bucket/some/folder", "test-bucket/some/folder/",
                                    "test-bucket//some//folder", "test-bucket/"])
def test_package_and_load_stimulus_set(s3_test_bucket, tmp_path, monkeypatch, bucket):
    s3_test_bucket.put_bucket_versioning(Bucket="test-bucket", VersioningConfiguration={"Status": "Enabled"})
    monkeypatch.setattr(fetch, "_local_data_path", str(tmp_path / "upload_home"))
    stimulus_paths = {}
    for stimulus_id in ["0", "1"]:
        path = tmp_path / f"{stimulus_id}.png"
        Image.fromarray(np.full((4, 4), int(stimulus_id) * 100, dtype=np.uint8)).save(path)
        stimulus_paths[stimulus_id] = str(path)
    stimulus_set = StimulusSet([{"stimulus_id": stimulus_id} for stimulus_id in stimulus_paths])
    stimulus_set.stimulus_paths = stimulus_paths
    stimulus_set.name = "dummy.Test2026"

    info = package_stimulus_set(None, stimulus_set, "dummy.Test2026", bucket_name=bucket)

    monkeypatch.setattr(fetch, "_local_data_path", str(tmp_path / "download_home"))
    loaded = load_stimulus_set_from_s3("dummy.Test2026", bucket=info["bucket"],
                                       csv_sha1=info["csv_sha1"], zip_sha1=info["zip_sha1"],
                                       csv_version_id=info["csv_version_id"],
                                       zip_version_id=info["zip_version_id"])
    assert len(loaded) == 2
    assert all(path.exists() for path in loaded.stimulus_paths.values())


@pytest.mark.parametrize("bucket", ["test-bucket", "test-bucket/some/folder"])
def test_package_and_load_data_assembly(s3_test_bucket, tmp_path, monkeypatch, bucket):
    s3_test_bucket.put_bucket_versioning(Bucket="test-bucket", VersioningConfiguration={"Status": "Enabled"})
    monkeypatch.setattr(fetch, "_local_data_path", str(tmp_path / "upload_home"))
    assembly = NeuroidAssembly(np.arange(6, dtype=float).reshape(3, 2),
                               coords={"stimulus_id": ("presentation", [0, 1, 2]),
                                       "neuroid_id": ("neuroid", [0, 1])},
                               dims=["presentation", "neuroid"])

    info = package_data_assembly(None, assembly, "dummy.Test2026", "dummy.Test2026", bucket_name=bucket)

    monkeypatch.setattr(fetch, "_local_data_path", str(tmp_path / "download_home"))
    loaded = load_assembly_from_s3("dummy.Test2026", version_id=info["version_id"], sha1=info["sha1"],
                                   bucket=info["bucket"], cls=NeuroidAssembly)
    np.testing.assert_array_equal(loaded.values, assembly.values)


@pytest.mark.parametrize("bucket", ["brainio.requested", "brainio.requested/some/folder"])
def test_requested_bucket_checks_stimulus_set(bucket, mocker):
    check = mocker.patch("brainscore_core.supported_data_standards.brainio.packaging.check_experiment_stimulus_set",
                         side_effect=RuntimeError("checked"))
    stimulus_set = StimulusSet([{"stimulus_id": "0"}])
    with pytest.raises(RuntimeError, match="checked"):
        package_stimulus_set(None, stimulus_set, "dummy.Test2026", bucket_name=bucket)
    check.assert_called_once()
