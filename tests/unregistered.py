"""A channel family name that the catalog does not hold when the test runs.

Unknown-family tests call this inside the test body, so a family registered
later (by a plugin or another test) never makes them pass or fail by accident.
"""

from brainscore_core import io_catalog


def unregistered_family() -> str:
    names = {entry.name for entry in io_catalog.all_entries()}
    return next(f'unregistered{index}' for index in range(len(names) + 1)
                if f'unregistered{index}' not in names)
