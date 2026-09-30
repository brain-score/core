"""Protect provenance based on the approved document, never the proposal."""

from .contract import FIELD_SPECS, LIST_PATHS, get_path, source_kind

# Scoring configuration and identity are outside the descriptive editor.
FIXED_PATHS = {"/model/visual_degrees", "/model/visual_degrees_description"}


def related(a, b):
    return a == b or a.startswith(b + "/") or b.startswith(a + "/")


def evidence(entry, path):
    assertions = [a for a in entry.get("assertions", []) if related(a["path"], path)]
    sources = entry.get("sources", {})
    refs = {ref for a in assertions for ref in a.get("sources", [])}
    return assertions, {ref: sources[ref] for ref in refs}


def editability(entry, path):
    if path in FIXED_PATHS:
        return False, "Scoring configuration is managed separately."
    _, sources = evidence(entry, path)
    kinds = {source_kind(source) for source in sources.values()}
    if "paper" in kinds or "huggingface" in kinds:
        return False, "Backed by a paper or Hugging Face source."
    if "unreviewed" in kinds:
        return False, "Source review is needed before this field can be edited."
    value = get_path(entry, path)
    if value is None or value == "" or value == []:
        return True, "Undocumented; add a source with your proposal."
    if kinds and kinds <= {"other"}:
        return True, "Editable with a supporting source."
    return False, "Source review is needed before this field can be edited."


def protected_changes(before, after):
    changed = []
    for path in (*FIELD_SPECS, *LIST_PATHS):
        old_evidence = evidence(before, path)
        new_evidence = evidence(after, path)
        if (
            get_path(before, path) != get_path(after, path)
            or old_evidence != new_evidence
        ):
            if not editability(before, path)[0]:
                changed.append(path)
            # Self-certifying a field as verified also requires reviewer override.
            elif (
                any(a["status"] == "verified" for a in new_evidence[0])
                and old_evidence != new_evidence
            ):
                changed.append(path)
    if before.get("legacy", {}) != after.get("legacy", {}):
        changed.append("/legacy")
    return sorted(set(changed))
