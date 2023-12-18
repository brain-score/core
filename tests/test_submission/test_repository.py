import logging
import os
import tempfile
from pathlib import Path

import pytest

from brainscore_core.submission.database import connect_db
from brainscore_core.submission.database_models import clear_schema
from brainscore_core.submission.repository import extract_zip_file, find_submission_directory
from tests.test_submission import init_users

logger = logging.getLogger(__name__)


@pytest.mark.memory_intense
class TestRepository:
    working_dir = None
    config_dir = str(os.path.join(os.path.dirname(__file__), 'configs/'))

    @classmethod
    def setup_class(cls):
        connect_db(db_secret='sqlite3.db')
        clear_schema()
        init_users()

    @classmethod
    def tear_down_class(cls):
        clear_schema()

    def setup_method(self):
        tmpdir = tempfile.mkdtemp()
        TestRepository.working_dir = tmpdir

    def tear_down_method(self):
        os.rmdir(TestRepository.working_dir)

    def test_extract_zip_file(self):
        path = extract_zip_file(33, TestRepository.config_dir, TestRepository.working_dir)
        assert str(path) == f'{TestRepository.working_dir}/candidate_models'

    def test_find_correct_dir(self):
        Path(f'{TestRepository.working_dir}/.temp').touch()
        Path(f'{TestRepository.working_dir}/_MACOS').touch()
        Path(f'{TestRepository.working_dir}/candidate_models').touch()
        dir = find_submission_directory(TestRepository.working_dir)
        assert dir == 'candidate_models'
        with pytest.raises(Exception):
            Path(f'{TestRepository.working_dir}/candidate_models2').touch()
            find_submission_directory(TestRepository.working_dir)
