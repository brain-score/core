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
    avoids ``xarray.concat`` for large 2-D activations. ``strategy='xarray'``
    keeps wrappers that already relied on xarray concat byte-for-byte closer.
    """
    if len(assemblies) == 1:
        return assemblies[0]
    if strategy == 'xarray':
        import xarray as xr
        return xr.concat(list(assemblies), dim='neuroid')
    if strategy != 'manual':
        raise ValueError(f"Unknown concat strategy {strategy!r}.")

    first = assemblies[0]
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
