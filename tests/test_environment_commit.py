from pathlib import Path

import pytest

from brainscore_core import environment_check
from brainscore_core.environment_check import _git_commit, _installed

COMMIT = "0123456789abcdef0123456789abcdef01234567"


def _checkout(root: Path, package: str = "pkg") -> Path:
    """A project folder with a pyproject.toml, a package folder and a git folder at COMMIT."""
    (root / package).mkdir(parents=True)
    (root / "pyproject.toml").write_text("[project]\nname='x'\n")
    git = root / ".git"
    (git / "refs" / "heads").mkdir(parents=True)
    (git / "HEAD").write_text("ref: refs/heads/main\n")
    (git / "refs" / "heads" / "main").write_text(COMMIT + "\n")
    return root / package


def test_an_editable_checkout_gives_its_commit(tmp_path):
    package = _checkout(tmp_path / "project")
    assert _git_commit(package) == COMMIT


def test_a_package_in_a_venv_inside_an_unrelated_repository_gives_none(tmp_path):
    _checkout(tmp_path / "unrelated")       # an unrelated repository with its own commit
    site = tmp_path / "unrelated" / ".venv" / "lib" / "site-packages"
    (site / "pkg").mkdir(parents=True)
    assert _git_commit(site / "pkg") is None


def test_a_worktree_git_file_with_a_relative_gitdir_resolves(tmp_path):
    main = tmp_path / "main"
    _checkout(main)
    git_dir = main / ".git" / "worktrees" / "w"
    git_dir.mkdir(parents=True)
    (git_dir / "HEAD").write_text("ref: refs/heads/feature\n")
    (git_dir / "commondir").write_text("../..\n")
    (main / ".git" / "refs" / "heads" / "feature").write_text(COMMIT[::-1] + "\n")
    worktree = tmp_path / "elsewhere" / "w"
    (worktree / "pkg").mkdir(parents=True)
    (worktree / "pyproject.toml").write_text("[project]\nname='x'\n")
    relative = Path(*([".."] * 2), "main", ".git", "worktrees", "w")
    (worktree / ".git").write_text(f"gitdir: {relative}\n")
    assert (worktree / relative / "HEAD").is_file()
    assert _git_commit(worktree / "pkg") == COMMIT[::-1]


def test_a_head_file_that_is_not_utf8_gives_none(tmp_path):
    package = _checkout(tmp_path / "project")
    (tmp_path / "project" / ".git" / "HEAD").write_bytes(b"\xff\xfe\x00bad")
    assert _git_commit(package) is None


def test_a_broken_dist_info_gives_none(monkeypatch):
    def broken(name):
        raise ValueError("broken metadata")
    monkeypatch.setattr(environment_check, "version", broken)
    assert _installed("numpy") is None
    assert environment_check._score_environment()["software"]["numpy"] is None
