"""
S3 MODULE - AWS S3 utilities for Brain-Score

PURPOSE:
--------
This module provides S3 file operations specifically for Brain-Score's needs.
It handles downloading model weights, assemblies, and stimulus sets from
AWS S3 buckets with proper caching and integrity verification.

KEY FUNCTIONS:
-------------
- load_file_from_s3(): Download any file from S3 to local path
- load_weight_file(): Download model weights (backwards compatibility)
- load_folder_from_s3(): Batch download multiple files
- load_assembly_from_s3(): Load data assemblies with stimulus set merging
- load_stimulus_set_from_s3(): Load stimulus sets from S3 (CSV + ZIP)
- get_path(): Resolve S3 paths for assemblies and stimulus sets
- download_file_if_not_exists(): Conditional download utility
- sha1_hash(): Calculate file integrity hashes

CORE FUNCTIONALITY:
------------------
- Downloads files from S3 with version ID support
- Caches files locally to avoid re-downloading
- Handles Brain-Score's standard directory structure
- Integrates with BotoFetcher for authenticated/unauthenticated access
- Provides backwards compatibility for existing model loading code

This module combines S3 functionality from multiple sources:
- Original brainscore_vision.model_helpers.s3 (model weights, general S3 ops)
- brainscore_vision.data_helpers.s3 (assembly/stimulus set loading)
All moved to core since S3 operations are domain-agnostic infrastructure
that should be shared across all Brain-Score domains (vision, language, etc.).
"""

import os
from pathlib import Path
from os.path import expanduser
import logging
import functools
from typing import Tuple, List, Union, Callable

import boto3
from botocore import UNSIGNED
from botocore.config import Config

from .fetch import BotoFetcher, verify_sha1, fetch_file, unzip, resolve_stimulus_set_class
from .assemblies import DataAssembly, AssemblyLoader, StimulusMergeAssemblyLoader, StimulusReferenceAssemblyLoader
from .stimuli import StimulusSetLoader, StimulusSet

_logger = logging.getLogger(__name__)


def get_path(identifier: str, file_type: str, bucket: str, version_id: str, sha1: str, filename_prefix: str = None, folder_name: str = None):
    """
    Finds path of desired file (for .csvs, .zips, and .ncs).
    """
    if filename_prefix is None:
        filename_prefix = 'stimulus_' if file_type in ('csv', 'zip') else 'assy_'
    
    filename = f"{filename_prefix}{identifier.replace('.', '_')}.{file_type}"
    if folder_name:
        remote_path = f"{folder_name}/{filename}"
    else:
        remote_path = filename
    file_path = fetch_file(location_type="S3",
                           location=f"https://{bucket}.s3.amazonaws.com/{remote_path}",
                           version_id=version_id,
                           sha1=sha1)
    return file_path


def load_assembly_from_s3(identifier: str, version_id: str, sha1: str, bucket: str, cls: type,
                          stimulus_set_loader: Callable[[], StimulusSet] = None,
                          merge_stimulus_set_meta: bool = True) -> DataAssembly:
    """
    Load a data assembly from S3, optionally within a specific folder.
    """
    # Parse bucket name and folder name
    if '/' in bucket:
        parts = bucket.split('/', 1)  # Split only on first '/', preserving the rest as folder path
        bucket = parts[0]
        folder_name = parts[1]
    else:
        folder_name = None
    file_path = get_path(identifier, 'nc', bucket, version_id, sha1, folder_name=folder_name)
    if stimulus_set_loader:  # merge stimulus set meta into assembly if `stimulus_set_loader` is passed
        stimulus_set = stimulus_set_loader()
        loader_base_class = StimulusMergeAssemblyLoader if merge_stimulus_set_meta else StimulusReferenceAssemblyLoader
        loader_class = functools.partial(loader_base_class,
                                         stimulus_set_identifier=stimulus_set.identifier, stimulus_set=stimulus_set)
    else:  # if no `stimulus_set_loader` passed, just load assembly
        loader_class = AssemblyLoader
    loader = loader_class(cls=cls, file_path=file_path)
    assembly = loader.load()
    assembly.attrs['identifier'] = identifier
    return assembly


def load_stimulus_set_from_s3(identifier: str, bucket: str, csv_sha1: str, zip_sha1: str,
                              csv_version_id: str, zip_version_id: str, filename_prefix: str = None):
    # Parse bucket name and folder name
    if '/' in bucket:
        parts = bucket.split('/', 1)  # Split only on first '/', preserving the rest as folder path
        bucket = parts[0]
        folder_name = parts[1]
    else:
        folder_name = None
    csv_path = get_path(identifier, 'csv', bucket, csv_version_id, csv_sha1, filename_prefix=filename_prefix, folder_name=folder_name)
    zip_path = get_path(identifier, 'zip', bucket, zip_version_id, zip_sha1, filename_prefix=filename_prefix, folder_name=folder_name)
    stimuli_directory = unzip(zip_path)
    loader = StimulusSetLoader(
        csv_path=csv_path,
        stimuli_directory=stimuli_directory,
        cls=resolve_stimulus_set_class('StimulusSet')
    )
    stimulus_set = loader.load()
    stimulus_set.identifier = identifier
    # ensure perfect overlap
    stimuli_paths = [Path(stimuli_directory) / local_path for local_path in os.listdir(stimuli_directory)
                     if not local_path.endswith('.zip') and not local_path.endswith('.csv')]
    assert set(stimulus_set.stimulus_paths.values()) == set(stimuli_paths), \
        "Inconsistency: unzipped stimuli paths do not match csv paths"
    return stimulus_set


def download_file_if_not_exists(local_path: Union[str, Path], bucket: str, remote_filepath: str):
    if local_path.is_file():
        return  # nothing to do, file already exists
    unsigned_config = Config(signature_version=UNSIGNED)  # do not attempt to look up credentials
    s3 = boto3.client('s3', config=unsigned_config)
    s3.download_file(bucket, remote_filepath, str(local_path))


def load_folder_from_s3(bucket: str, folder_path: str, filename_version_sha: List[Tuple[str, str, str]],
                        save_directory: Union[Path, str]):
    for filename, version_id, sha1 in filename_version_sha:
        load_file_from_s3(bucket=bucket, path=f"{folder_path}/{filename}", version_id=version_id, sha1=sha1,
                          local_filepath=Path(save_directory) / filename)


def load_file_from_s3(bucket: str, path: str, local_filepath: Union[Path, str],
                      sha1: Union[str, None] = None, version_id: Union[str, None] = None):
    """
    Load a file from AWS S3 and validate its contents.
    :param bucket: The name of the S3 bucket
    :param path: The path of the file inside the S3 bucket
    :param local_filepath: The local path of the file to be saved to
    :param sha1: The SHA1 hash of the file. If you are not sure of this, use the `sha1_hash` function in this same file
    :param version_id: Which version of the file on S3 to use.
        Optional but strongly encouraged to avoid accidental overwrites.
        If you use Brain-Score functionality to upload files to S3, the version id will be printed to the console.
        You can also find this on the S3 user interface by opening the file and then clicking on the versions tab.
    """
    fetcher = BotoFetcher(location=f"https://{bucket}.s3.amazonaws.com/{path}", version_id=version_id,
                          # this is a bit hacky: don't tell BotoFetcher the full path because it will make a directory
                          # where there should be a file
                          local_filename=Path(local_filepath).parent)
    fetcher.output_filename = str(local_filepath)  # force using this local path instead of folder structure
    fetcher.fetch()


def load_file(bucket: str, relative_path: str, version_id: str, folder_name: str = None) -> Path:
    """
    :param bucket: bucket to download file from. Usually is brainscore-storage
    :param relative_path: The path of the file inside the S3 bucket, relative to the `{folder_name}/` directory.
    :param version_id: version_id of the object to download, found in AWS under object properties
    :param folder_name: name of the folder inside the bucket to download from, i.e. 'models'
    :return: Posix Path for the file.
    """
    brainscore_cache = os.getenv("BRAINSCORE_HOME", expanduser("~/.brain-score"))
    s3_weight_folder = folder_name if folder_name is not None else os.getenv("BRAINSCORE_S3_WEIGHT_FOLDER", "models")
    local_path = Path(brainscore_cache) / s3_weight_folder / relative_path
    local_path.parent.mkdir(parents=True, exist_ok=True)
    load_file_from_s3(bucket=bucket, path=f"{s3_weight_folder}/{relative_path}", version_id=version_id,
                          local_filepath=local_path)
    return local_path


def load_weight_file(bucket: str, relative_path: str, version_id: str, sha1: str, folder_name: str = None) -> Path:
    """
    Wrapper function for backwards compatibility, post large-file upload

    :param bucket: bucket to download file from. Usually is 'brainscore-storage
    :param relative_path: The path of the file inside the S3 bucket, relative to the `{folder_name}/` directory.
    :param version_id: version_id of the object to download, found in AWS under object properties
    :param sha1: string SHA1 hash of the file to download. Deprecated as of 5/14/25.
    :param folder_name: :param folder_name: name of the folder inside the bucket to download from, i.e. 'models'
    :return: Posix Path for the file.
    """
    return load_file(bucket=bucket, relative_path=relative_path, version_id=version_id, folder_name=folder_name)


def sha1_hash(file_path):
    """Calculate SHA1 hash of a file."""
    import hashlib
    sha1 = hashlib.sha1()
    with open(file_path, 'rb') as f:
        while chunk := f.read(8192):
            sha1.update(chunk)
    return sha1.hexdigest()
