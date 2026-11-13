import boto3
import numpy as np
import pytest
from moto import mock_aws
from PIL import Image

from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly
from brainscore_core.supported_data_standards.brainio.packaging import (package_data_assembly, package_stimulus_set,
                                                                        upload_to_s3)
from brainscore_core.supported_data_standards.brainio.s3 import load_assembly_from_s3, load_stimulus_set_from_s3
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet


@pytest.fixture
def s3_bucket(monkeypatch):
    for key, value in {"AWS_ACCESS_KEY_ID": "testing", "AWS_SECRET_ACCESS_KEY": "testing",
                       "AWS_SESSION_TOKEN": "testing", "AWS_DEFAULT_REGION": "us-east-1"}.items():
        monkeypatch.setenv(key, value)
    with mock_aws():
        client = boto3.client("s3")
        client.create_bucket(Bucket="test-bucket")
        yield client


@pytest.fixture
def source_file(tmp_path):
    path = tmp_path / "stimulus_dummy.csv"
    path.write_text("stimulus_id,filename\n0,0.png\n")
    return path


def test_upload_to_bucket(s3_bucket, source_file):
    upload_to_s3(source_file, "test-bucket", target_s3_key="stimulus_dummy.csv")
    keys = [obj["Key"] for obj in s3_bucket.list_objects_v2(Bucket="test-bucket")["Contents"]]
    assert keys == ["stimulus_dummy.csv"]


@pytest.mark.parametrize("bucket, expected_key", [
    ("test-bucket/some/folder", "some/folder/stimulus_dummy.csv"),
    ("test-bucket/", "stimulus_dummy.csv"),
])
def test_upload_to_bucket_folder(s3_bucket, source_file, bucket, expected_key):
    # `load_stimulus_set_from_s3` and `load_assembly_from_s3` accept "<bucket>/<folder>", so uploads should too
    properties = upload_to_s3(source_file, bucket, target_s3_key="stimulus_dummy.csv")
    keys = [obj["Key"] for obj in s3_bucket.list_objects_v2(Bucket="test-bucket")["Contents"]]
    assert keys == [expected_key]
    assert properties["ContentLength"] == source_file.stat().st_size


@pytest.mark.parametrize("bucket", ["test-bucket", "test-bucket/some/folder", "test-bucket/some/folder/",
                                    "test-bucket/"])
def test_package_and_load_stimulus_set(s3_bucket, tmp_path, monkeypatch, bucket):
    s3_bucket.put_bucket_versioning(Bucket="test-bucket", VersioningConfiguration={"Status": "Enabled"})
    monkeypatch.setenv("BRAINIO_HOME", str(tmp_path / "upload_home"))
    stimulus_paths = {}
    for stimulus_id in ["0", "1"]:
        path = tmp_path / f"{stimulus_id}.png"
        Image.fromarray(np.full((4, 4), int(stimulus_id) * 100, dtype=np.uint8)).save(path)
        stimulus_paths[stimulus_id] = str(path)
    stimulus_set = StimulusSet([{"stimulus_id": stimulus_id} for stimulus_id in stimulus_paths])
    stimulus_set.stimulus_paths = stimulus_paths
    stimulus_set.name = "dummy.Test2026"

    info = package_stimulus_set(None, stimulus_set, "dummy.Test2026", bucket_name=bucket)

    monkeypatch.setenv("BRAINIO_HOME", str(tmp_path / "download_home"))
    loaded = load_stimulus_set_from_s3("dummy.Test2026", bucket=info["bucket"],
                                       csv_sha1=info["csv_sha1"], zip_sha1=info["zip_sha1"],
                                       csv_version_id=info["csv_version_id"],
                                       zip_version_id=info["zip_version_id"])
    assert len(loaded) == 2
    assert all(path.exists() for path in loaded.stimulus_paths.values())


@pytest.mark.parametrize("bucket", ["test-bucket", "test-bucket/some/folder"])
def test_package_and_load_data_assembly(s3_bucket, tmp_path, monkeypatch, bucket):
    s3_bucket.put_bucket_versioning(Bucket="test-bucket", VersioningConfiguration={"Status": "Enabled"})
    monkeypatch.setenv("BRAINIO_HOME", str(tmp_path / "upload_home"))
    assembly = NeuroidAssembly(np.arange(6, dtype=float).reshape(3, 2),
                               coords={"stimulus_id": ("presentation", [0, 1, 2]),
                                       "neuroid_id": ("neuroid", [0, 1])},
                               dims=["presentation", "neuroid"])

    info = package_data_assembly(None, assembly, "dummy.Test2026", "dummy.Test2026", bucket_name=bucket)

    monkeypatch.setenv("BRAINIO_HOME", str(tmp_path / "download_home"))
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
