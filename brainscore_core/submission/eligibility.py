"""Keep local-only evaluations out of shared score storage."""


def require_leaderboard_eligible(flag: bool) -> None:
    """Reject an explicit exclusion or malformed publication flag.

    Callers default a missing flag to True for existing benchmarks. This guard
    applies to submission storage and uploads, not local benchmark execution.
    """
    if type(flag) is not bool:
        raise TypeError('leaderboard_eligible must be a boolean when declared')
    if not flag:
        raise ValueError(
            'leaderboard_eligible=False: this evaluation is for local use only; '
            'it cannot be stored or uploaded by the submission pipeline'
        )
