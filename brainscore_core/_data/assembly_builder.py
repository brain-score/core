"""Shared helpers for constructing Brain-Score xarray assemblies."""

from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from brainscore_core.supported_data_standards.brainio.assemblies import (
    NeuroidAssembly, walk_coords,
)


def make_neuroid_ids(model_identifier: str, layer_name: str,
                     n_features: int) -> list[str]:
    """Return the standard ``model.layer.unit`` neuroid ids."""
    return [
        f"{model_identifier}.{layer_name}.{i}"
        for i in range(n_features)
    ]


def make_layer_neuroid_coords(
    model_identifier: str,
    layer_name: str,
    n_features: int,
    include: Sequence[str] = ('neuroid_id', 'neuroid_num', 'model', 'layer'),
) -> dict:
    """Build common neuroid-axis coords in caller-specified order."""
    coords: dict[str, tuple[str, list[Any]]] = {}
    for coord in include:
        if coord == 'neuroid_id':
            coords['neuroid_id'] = (
                'neuroid',
                make_neuroid_ids(model_identifier, layer_name, n_features),
            )
        elif coord == 'neuroid_num':
            coords['neuroid_num'] = ('neuroid', list(range(n_features)))
        elif coord == 'model':
            coords['model'] = ('neuroid', [model_identifier] * n_features)
        elif coord == 'layer':
            coords['layer'] = ('neuroid', [layer_name] * n_features)
        else:
            raise ValueError(f"Unknown neuroid coord {coord!r}.")
    return coords


def make_assembly(values, coords: Mapping, dims: Sequence[str],
                  assembly_cls=NeuroidAssembly):
    """Construct an assembly without changing coord insertion order."""
    return assembly_cls(values, coords=dict(coords), dims=list(dims))


def concat_neuroid_assemblies(
    assemblies: Sequence,
    *,
    strategy: str = 'manual',
    validate_non_neuroid: bool = False,
):
    """Concatenate assemblies along ``neuroid`` preserving existing coords.

    ``strategy='manual'`` matches the historic wrapper implementation that
    avoids ``xarray.concat`` for large 2-D activations. It joins rows by
    presentation identity, not by position: a block that lists the same
    presentations in another order is put in the first block's order, and a
    block whose rows cannot be matched to the first block's is refused. When
    the blocks have no presentation identity at all (no ``presentation``
    dimension, or equal presentation keys that do not tell rows apart, such as
    repeated ``stimulus_id`` values without an index), rows are joined by
    position, as before: nothing identifies them, and nothing contradicts the
    order.
    ``strategy='xarray'`` keeps wrappers that already relied on xarray concat
    byte-for-byte closer.
    """
    if len(assemblies) == 1:
        return assemblies[0]
    if any('time_bin' in a.dims for a in assemblies):
        from .time_axis import require_one_time_axis
        require_one_time_axis(assemblies, sources=[f'block{i}' for i in range(len(assemblies))])
    if strategy == 'xarray':
        import xarray as xr
        return xr.concat(list(assemblies), dim='neuroid')
    if strategy != 'manual':
        raise ValueError(f"Unknown concat strategy {strategy!r}.")

    first = assemblies[0]
    assemblies = [first] + [_in_presentation_order(first, assembly, i)
                            for i, assembly in enumerate(assemblies[1:], start=1)]
    merged = np.concatenate(
        [a.values for a in assemblies],
        axis=first.dims.index('neuroid'),
    )
    nonneuroid_coords = {
        coord: (dims, values)
        for coord, dims, values in walk_coords(first)
        if set(dims) != {'neuroid'}
    }
    neuroid_coords: dict[str, list[Any]] = {
        coord: [dims, values]
        for coord, dims, values in walk_coords(first)
        if set(dims) == {'neuroid'}
    }
    for assembly in assemblies[1:]:
        for coord in neuroid_coords:
            neuroid_coords[coord][1] = np.concatenate(
                (neuroid_coords[coord][1], assembly[coord].values)
            )
        if validate_non_neuroid:
            assert first.dims == assembly.dims
            for coord, dims, values in walk_coords(assembly):
                if set(dims) == {'neuroid'}:
                    continue
                assert (values == nonneuroid_coords[coord][1]).all()

    final_neuroid_coords = {
        coord: (dims_values[0], dims_values[1])
        for coord, dims_values in neuroid_coords.items()
    }
    return type(first)(
        merged,
        coords={**nonneuroid_coords, **final_neuroid_coords},
        dims=first.dims,
    )


def _in_presentation_order(first, assembly, position: int):
    """``assembly`` with its rows in ``first``'s presentation order.

    Presentations are matched with the rule the time-axis guard uses: an
    identical complete presentation index needs no reorder; otherwise unique
    ``stimulus_id`` values, or else the complete index with its level names,
    identify each row. Rows that cannot be matched are refused, because joining
    them by position would pair one presentation's values with another's.
    """
    if 'presentation' not in first.dims or 'presentation' not in assembly.dims:
        return assembly
    index_a = first.indexes.get('presentation')
    index_b = assembly.indexes.get('presentation')
    if (index_a is not None and index_b is not None
            and list(index_a.names) == list(index_b.names) and index_a.equals(index_b)):
        return assembly                     # the same rows in the same order
    from .time_axis import _alignment, _presentation_keys
    keys_a, keys_b = _presentation_keys(first), _presentation_keys(assembly)
    order = _alignment(keys_a, keys_b)
    if order is None:
        if keys_a == keys_b:
            return assembly                 # nothing tells the rows apart
        raise ValueError(
            f"Block {position} does not list the same presentations as block 0 "
            "in a way that can be matched row by row (identical presentation "
            "index, unique stimulus_id, or the complete presentation index), so "
            "its rows cannot be joined without pairing values from different "
            "presentations. Give every block the same presentations.")
    if order == list(range(len(order))):
        return assembly
    return assembly.isel(presentation=order)


def replace_presentation_coord(assembly, coord_name: str,
                               values: Iterable):
    """Return a fresh assembly with one presentation coord replaced."""
    new_coords = {}
    for name, dims, coord_values in walk_coords(assembly):
        if name == coord_name:
            continue
        new_coords[name] = (dims, coord_values)
    new_coords[coord_name] = ('presentation', list(values))
    return type(assembly)(assembly.values, coords=new_coords, dims=assembly.dims)


# Keep existing pickle references and public type names stable.
for _definition in (
    make_neuroid_ids,
    make_layer_neuroid_coords,
    make_assembly,
    concat_neuroid_assemblies,
    replace_presentation_coord,
):
    _definition.__module__ = 'brainscore_core.assembly_builder'
del _definition
