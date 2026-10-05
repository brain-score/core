# Minimal Subject contract

The public `Subject` must support implementations that only declare identity,
input/output channels, optional hard requirements, and `interact(session)`.
`reset()` remains a default no-op lifecycle hook for stateless subjects.

Keep the legacy typed interface on `UnifiedModel`, a compatibility subclass.
This makes the contract explicit without adding legacy requirements or automatic
method injection to native subjects. Existing domain plugins keep their adapters;
existing typed UMI implementations use the compatibility base. Computation,
preprocessing, and metric paths are unchanged.

Native scoring preflight checks channel declarations, without requiring layer
maps or modality names. Wrapper-specific session probes remain optional for
native subjects and retain their current behavior for legacy wrappers. The
feature-width memory probe uses a neural session for native subjects.

Acceptance tests cover missing required declarations, native open-loop and
feedback sessions, reset, typed response helpers, early compatibility rejection,
and feature-width probing. Existing contract, wrapper, adapter, and offline
integration tests protect the compatibility paths. Heavy scoring and downloads
are outside this change; unit parity checks do not establish full scientific
qualification.

## Verification

Verified on September 21, 2026 using the existing Python 3.11 test environment
with scikit-learn 1.7.2. `PYTHONPATH` selected all four production worktrees.
Every pytest run set `RESULTCACHING_DISABLE=1`, `UMI_TEST_CPU_ONLY=1`,
`HF_HUB_OFFLINE=1`, and `TRANSFORMERS_OFFLINE=1`.

- Core top-level unit suites (`tests/test_*.py`): 628 passed.
- Vision adapter, preflight, and conformance: 44 passed.
- Language adapter, preflight, and conformance: 38 passed.
- Unified offline suites (`not slow and not private_access`): 766 passed,
  14 skipped, 64 deselected.
- Included native scoring integration tests: 3 passed, one per package
  (unified, vision, language). The loaders preserve the native subject and the
  scoring entry points reach a session-based synthetic benchmark.
- The Python example in `docs/UMI_MIGRATION.md` executes successfully.
- All four worktree diffs pass `git diff --check`.

These are local contract and compatibility checks, not real-model scientific
qualification. No models or data were downloaded,
no production scoring was run, and no cloud resources were started.
