"""One shared time axis is required before outputs are merged along ``neuroid``."""


def _presentation_keys(assembly):
    """``((level names, index rows), stimulus_ids)``; either part may be None."""
    import numpy as np
    if 'presentation' not in assembly.dims:
        return None
    index = assembly.indexes.get('presentation')
    # The index schema (level names) is part of a presentation's identity.
    full = (tuple(index.names), list(index)) if index is not None else None
    try:
        ids = list(np.asarray(assembly['stimulus_id'].values))
    except KeyError:
        ids = None
    return full, ids


def _alignment(keys_a, keys_b):
    """Row order of b that matches a's presentations, or None if it is unknown.

    Identical complete indexes need no reorder (repeated stimulus_ids are valid
    there). Otherwise unique stimulus_ids identify presentations; with repeated
    stimulus_ids the complete index (for example stimulus_id and trial) must.
    """
    if keys_a is None or keys_b is None:
        return None
    (full_a, ids_a), (full_b, ids_b) = keys_a, keys_b
    if full_a is not None and full_a == full_b:
        return list(range(len(full_a[1])))
    same_schema = full_a is not None and full_b is not None and full_a[0] == full_b[0]
    rows_a = full_a[1] if same_schema else None
    rows_b = full_b[1] if same_schema else None
    for a, b in ((ids_a, ids_b), (rows_a, rows_b)):
        if a is None or b is None or len(a) != len(b) or len(set(a)) != len(a) or set(a) != set(b):
            continue
        where = {identity: position for position, identity in enumerate(b)}
        return [where[identity] for identity in a]
    return None


def _time_axis(assembly):
    """What the ``time_bin`` axis means: its size, index and every time coordinate."""
    if 'time_bin' not in assembly.dims:
        return None
    index = assembly.indexes.get('time_bin')
    coords = {}
    for name in set(assembly.coords) | set(getattr(index, 'names', None) or ()):
        if not name or name == 'time_bin':
            continue
        coord = assembly[name]
        if 'time_bin' in coord.dims:
            coords[name] = coord
    return assembly.sizes['time_bin'], index, coords, _presentation_keys(assembly)


def _same_values(a, b) -> bool:
    import numpy as np
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return False
    if a.dtype.kind in 'fc' and b.dtype.kind in 'fc':
        return bool(np.array_equal(a, b, equal_nan=True))
    return bool(np.array_equal(a, b))


def _same_time_axis(a, b) -> bool:
    import numpy as np
    if a is None or b is None:
        return a is None and b is None
    (size_a, index_a, coords_a, ids_a), (size_b, index_b, coords_b, ids_b) = a, b
    if size_a != size_b or (index_a is None) != (index_b is None):
        return False
    if index_a is not None and (list(index_a.names) != list(index_b.names)
                                or not index_a.equals(index_b)):
        return False
    if set(coords_a) != set(coords_b):
        return False
    for name, coord_a in coords_a.items():
        coord_b = coords_b[name]
        if coord_a.dims != coord_b.dims:
            return False
        values_a, values_b = np.asarray(coord_a.values), np.asarray(coord_b.values)
        if 'presentation' in coord_a.dims:
            # Per-presentation bounds: compare the same presentation, not the same row.
            order = _alignment(ids_a, ids_b)
            if order is None:
                return False
            values_b = np.take(values_b, order, axis=coord_b.dims.index('presentation'))
        if not _same_values(values_a, values_b):
            return False
    return True


def require_one_time_axis(sub_assemblies, sources=None) -> None:
    """Refuse to merge outputs whose time axes differ.

    A merge along ``neuroid`` needs one shared ``time_bin`` axis: the same
    length, index and time coordinates, including per-presentation bounds
    compared stimulus by stimulus. Different axes would be padded, broadcast or
    aligned by position, which invents values (contract rule 3.6). ``sources``
    names the parts in errors.
    """
    axes = [_time_axis(assembly) for assembly in sub_assemblies]
    if all(_same_time_axis(axes[0], axis) for axis in axes[1:]):
        return
    if sources is None:
        sources = [str(assembly['modality'].values[0])
                   if 'modality' in assembly.coords and assembly.sizes['neuroid']
                   else '?' for assembly in sub_assemblies]
    raise ValueError(
        f"The outputs for {list(sources)} have different time axes, so they "
        "cannot be merged into one assembly without inventing values. Request "
        "each region as its own neural:<region> channel (each keeps its own "
        "time axis), or call process() once per input type.")
