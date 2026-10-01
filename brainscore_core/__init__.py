"""Public scoring interfaces, loaded only when requested."""

from importlib import import_module

__all__ = ["Metric", "Score", "Benchmark"]


def __getattr__(name):
    modules = {"Metric": ".metrics", "Score": ".metrics", "Benchmark": ".benchmarks"}
    if name not in modules:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(modules[name], __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
