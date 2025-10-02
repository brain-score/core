"""
FETCH MODULE - File downloading and data loading for Brain-Score

PURPOSE:
--------
This module handles downloading and loading of Brain-Score data from remote
sources (primarily AWS S3). It provides the core infrastructure for fetching
assemblies, stimulus sets, and model weights without requiring the BrainIO
catalog system.

KEY CLASSES:
-----------
- BotoFetcher: Downloads files from AWS S3 with progress tracking
- Assembly/StimulusSet resolvers: Map class names to actual classes

CORE FUNCTIONALITY:
------------------
- S3 file downloading with authentication fallback
- ZIP file extraction for stimulus sets
- SHA1 hash verification for data integrity
- Local caching to avoid re-downloading files
- Progress bars for large file transfers

CATALOG SYSTEM REMOVED:
----------------------
The original BrainIO get_assembly() and get_stimulus_set() functions that
relied on CSV catalog lookups are replaced with direct S3 parameter functions
in the brainscore_core.supported_data_standards.brainio.s3 module.
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import os
import zipfile
from pathlib import Path

import boto3
from botocore import UNSIGNED
from botocore.config import Config
from six.moves.urllib.parse import urlparse
from tqdm import tqdm

from . import assemblies
from . import stimuli
from .stimuli import StimulusSetLoader

BRAINIO_HOME = 'BRAINIO_HOME'

_logger = logging.getLogger(__name__)
_local_data_path = None


def get_local_data_path():
    # This makes it easier to mock.
    global _local_data_path
    if _local_data_path is None:
        _local_data_path = os.path.expanduser(os.getenv(BRAINIO_HOME, '~/.brainio'))
    return _local_data_path


class Fetcher(object):
    """A Fetcher obtains data with which to populate a DataAssembly.  """

    def __init__(self, location, local_filename):
        self.location = location
        self.local_filename = local_filename
        self.local_dir_path = os.path.join(get_local_data_path(), self.local_filename)
        os.makedirs(self.local_dir_path, exist_ok=True)

    def fetch(self):
        """
        Fetches the resource identified by location.
        :return: a full local file path
        """
        raise NotImplementedError("The base Fetcher class does not implement .fetch().  Use a subclass of Fetcher.")


class BotoFetcher(Fetcher):
    """A Fetcher that retrieves files from Amazon Web Services' S3 data storage.  """

    def __init__(self, location, local_filename, version_id=None):
        super(BotoFetcher, self).__init__(location, local_filename)
        parsed_url = urlparse(self.location)
        split_path = parsed_url.path.lstrip('/').split("/")
        # http://docs.aws.amazon.com/AmazonS3/latest/dev/UsingBucket.html#access-bucket-intro
        virtual_hosted_style = 's3.' in parsed_url.hostname  # s3. for virtual hosted style; s3- for older AWS
        if virtual_hosted_style:
            self.bucketname = parsed_url.hostname.split(".s3.")[0]
            self.relative_path = os.path.join(*(split_path))
        else:
            self.bucketname = split_path[0]
            self.relative_path = os.path.join(*(split_path[1:]))
        self.extra_args = {"VersionId": version_id} if version_id else None
        self.output_filename = os.path.join(self.local_dir_path, os.path.basename(self.relative_path))
        # Ensure the directory exists
        os.makedirs(os.path.dirname(self.output_filename), exist_ok=True)
        self._logger = logging.getLogger(fullname(self))

    def fetch(self):
        # Ensure the directory path for output_filename exists
        if not os.path.exists(self.output_filename):
            self.download_boto()
        return self.output_filename

    def download_boto(self):
        """Downloads file from S3 via boto at `url` and writes it in `self.output_filename`."""
        try:  # try with authentication
            self._logger.debug("attempting default download (signed)")
            self.download_boto_config(config=None)
        except Exception as e_signed:  # try without authentication
            self._logger.debug("default download failed, trying unsigned")
            # disable signing requests. see https://stackoverflow.com/a/34866092/2225200
            unsigned_config = Config(signature_version=UNSIGNED)
            try:
                self.download_boto_config(config=unsigned_config)
            except Exception as e_unsigned:
                # when unsigned download also fails, raise both exceptions
                # raise Exception instead of specific type to avoid missing __init__ arguments
                raise Exception([e_signed, e_unsigned])

    def download_boto_config(self, config):
        s3 = boto3.resource('s3', config=config)
        obj = s3.Object(self.bucketname, self.relative_path)
        # show progress. see https://gist.github.com/wy193777/e7607d12fad13459e8992d4f69b53586
        with tqdm(total=obj.content_length, unit='B', unit_scale=True,
                  desc=self.bucketname + "/" + self.relative_path) as progress_bar:
            def progress_hook(bytes_amount):
                if bytes_amount > 0:  # at the end, this sometimes passes a negative byte amount which tqdm can't handle
                    progress_bar.update(bytes_amount)

            obj.download_file(self.output_filename, Callback=progress_hook, ExtraArgs=self.extra_args)


def fetch_file(location_type, location, sha1=None, version_id=None):
    if location_type == 'S3':
        fetcher = BotoFetcher(location=location, local_filename=sha1, version_id=version_id)
    else:
        raise NotImplementedError(f"Unknown location type {location_type}")
    
    local_path = fetcher.fetch()
    if sha1:
        verify_sha1(local_path, sha1)
    return local_path


def verify_sha1(file_path, expected_sha1):
    """Verify the SHA1 hash of a file."""
    import hashlib
    sha1_hash = hashlib.sha1()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha1_hash.update(byte_block)
    actual_sha1 = sha1_hash.hexdigest()
    if actual_sha1 != expected_sha1:
        raise ValueError(f"SHA1 mismatch. Expected: {expected_sha1}, Actual: {actual_sha1}")


def unzip(zip_path):
    """Unzip a file and return the directory containing the extracted files."""
    extract_dir = os.path.splitext(zip_path)[0]
    if not os.path.exists(extract_dir):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
    return extract_dir


def resolve_assembly_class(class_name):
    """Resolve assembly class by name."""
    assembly_classes = {
        'DataAssembly': assemblies.DataAssembly,
        'NeuroidAssembly': assemblies.NeuroidAssembly,
        'BehavioralAssembly': assemblies.BehavioralAssembly,
        'PropertyAssembly': assemblies.PropertyAssembly,
        'MetadataAssembly': assemblies.MetadataAssembly,
        'SpikeTimesAssembly': assemblies.SpikeTimesAssembly,
    }
    if class_name in assembly_classes:
        return assembly_classes[class_name]
    else:
        raise ValueError(f"Unknown assembly class: {class_name}")


def resolve_stimulus_set_class(class_name):
    """Resolve stimulus set class by name."""
    if class_name == 'StimulusSet' or class_name is None:
        return stimuli.StimulusSet
    else:
        raise ValueError(f"Unknown stimulus set class: {class_name}")


def get_assembly(identifier):
    """
    Simplified get_assembly function - requires direct S3 parameters.
    This replaces the catalog-based lookup system.
    """
    raise NotImplementedError(
        "get_assembly requires direct S3 parameters in brainio-lite. "
        "Use the load_assembly_from_s3 function from brainscore_core.supported_data_standards.brainio.s3 instead."
    )


def get_stimulus_set(identifier):
    """
    Simplified get_stimulus_set function - requires direct S3 parameters.
    This replaces the catalog-based lookup system.
    """
    raise NotImplementedError(
        "get_stimulus_set requires direct S3 parameters in brainio-lite. "
        "Use the load_stimulus_set_from_s3 function from brainscore_core.supported_data_standards.brainio.s3 instead."
    )


def fullname(obj):
    return obj.__module__ + "." + obj.__class__.__name__
