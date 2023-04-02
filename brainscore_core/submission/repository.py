import logging
import os
import zipfile
from pathlib import Path

logger = logging.getLogger(__name__)


def extract_zip_file(submission_id, config_path, work_dir):
    logger.info(f'Unpack zip file')
    zip_file = Path(f'{config_path}/submission_{submission_id}.zip')
    with zipfile.ZipFile(zip_file, 'r') as model_repo:
        model_repo.extractall(path=str(work_dir))
    # Use the single directory in the zip file
    full_path = Path(work_dir).absolute()
    submission_directory = find_submission_directory(work_dir)
    return full_path / submission_directory


def find_submission_directory(work_dir):
    """
    Find the single directory inside a directory that corresponds to the submission file.
    Ignores hidden directories, e.g. those prefixed with `.` and `_`
    """
    path_list = os.listdir(work_dir)
    candidates = []
    for item in path_list:
        if not item.startswith('.') and not item.startswith('_'):
            candidates.append(item)
    if len(candidates) is 1:
        return candidates[0]
    logger.error('The zip file structure is not correct, we try to detect the correct directory')
    if 'sample-model-submission' in candidates:
        return 'sample-model-submission'
    logger.error('The submission file contains too many entries and can therefore not be installed')
    raise Exception('The submission file contains too many entries and can therefore not be installed')
