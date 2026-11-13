"""Cheap shared checks to run before constructing an expensive model."""
import os
from pathlib import Path
import tempfile


def check_cache_directory():
    """Check enabled result storage with a real temporary write; never alter links.

    Configure RESULTCACHING_HOME before importing scoring packages, because the
    result-caching decorators capture that location when they are imported.
    """
    if os.environ.get('RESULTCACHING_DISABLE') == '1':
        return None
    path = Path(os.environ.get('RESULTCACHING_HOME', '~/.result_caching')).expanduser()
    try:
        path.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryFile(dir=path) as probe:
            probe.write(b'cache preflight')
            probe.flush()
    except OSError as error:
        raise OSError(
            f'Result cache is not writable: {path}. Check the directory or mounted '
            'drive, or set RESULTCACHING_HOME to a writable directory before '
            'starting Python. Set RESULTCACHING_DISABLE=1 to run without caching.'
        ) from error
    return path
