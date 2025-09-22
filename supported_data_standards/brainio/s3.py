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
- sha1_hash(): Calculate file integrity hashes

CORE FUNCTIONALITY:
------------------
- Downloads files from S3 with version ID support
- Caches files locally to avoid re-downloading
- Handles Brain-Score's standard directory structure
- Integrates with BotoFetcher for authenticated/unauthenticated access
- Provides backwards compatibility for existing model loading code

This was originally in brainscore_vision.model_helpers.s3 but moved to core
since S3 operations are domain-agnostic infrastructure that should be shared
across all Brain-Score domains (vision, language, etc.).
"""

import os
from pathlib import Path
from os.path import expanduser
import logging
from typing import Tuple, List, Union

from .fetch import BotoFetcher, verify_sha1

_logger = logging.getLogger(__name__)


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
