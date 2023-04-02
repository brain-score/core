# Endpoints for plugin submissions

All entry points are defined in `endpoints.py`.
The code in this package provides helper functions, 
it is never itself directly called from the command line.
Instead, the `run_scoring` endpoint will be called in its domain-specific library such as *language*,
which in turn uses the generic endpoint helpers here in *core*.
