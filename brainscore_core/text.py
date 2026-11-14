"""Explicit text presentation context shared by native and legacy wrappers.

Table rows are independent unless context_id groups them into ordered passage
parts. Context is scoped to one process call, never carried between calls.
"""

import re


def text_parts(stimuli):
    for column in ('sentence', 'text'):
        if column in stimuli.columns:
            parts = list(stimuli[column])
            if not all(isinstance(part, str) for part in parts):
                raise ValueError('Text presentation parts must be strings.')
            return parts
    raise ValueError('Text stimuli require a sentence or text column.')


def text_groups(stimuli):
    """Return groups of row positions, in first-appearance order."""
    if 'context_id' not in stimuli.columns:
        return [[i] for i in range(len(stimuli))]
    import pandas as pd
    groups = {}
    for i, group in enumerate(stimuli['context_id']):
        if not pd.api.types.is_scalar(group) or pd.isna(group):
            raise ValueError('context_id values must be non-missing scalar identifiers.')
        groups.setdefault(group, []).append(i)
    return list(groups.values())


def prepare_context(parts):
    """Join ordered English text parts using the legacy language convention."""
    return re.sub(r'\s+([.,!?;:])', r'\1', ' '.join(
        part for part in parts if part.strip()))


def contextualized_texts(stimuli):
    parts = text_parts(stimuli)
    if 'context_id' not in stimuli.columns:
        return parts
    result = list(parts)
    for positions in text_groups(stimuli):
        prefix = []
        for position in positions:
            prefix.append(parts[position])
            result[position] = prepare_context(prefix)
    return result
