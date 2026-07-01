"""Unit-selection helpers for recording and perturbation."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union

from .events import Selection


class UnitSelector(ABC):
    """Forward-compat ABC for ``region_layer_map`` values.

    Today every submitter declares one layer per region with a bare string.
    The interface accepts ``Union[str, UnitSelector]`` so future selectors
    (multi-layer regions, indexed sub-populations, spike-level alignment)
    are additive subclasses, not an interface change. Bare strings are
    auto-promoted to ``LayerSelector(name=str)`` at registration time.

    Subclasses must expose a ``layer_path`` property returning the dotted
    module path the wrapper extracts from. Future selectors that span
    multiple layers will return the canonical or default one and supply
    additional metadata for the wrapper to consume.
    """

    @property
    @abstractmethod
    def layer_path(self) -> str:
        ...


@dataclass(frozen=True)
class LayerSelector(UnitSelector):
    """The single concrete ``UnitSelector`` for the initial release.

    One dotted module path per region — the value every existing
    registration uses.
    """
    name: str

    @property
    def layer_path(self) -> str:
        return self.name


@dataclass(frozen=True)
class CompositeSelector(UnitSelector):
    """Select units across multiple layers as one region.

    A composite region's neuroids are gathered from several layers, each
    optionally restricted to a subset of unit positions. Used when a brain
    region is best modeled by units drawn from more than one model layer
    (for example a whole-brain parcel that pools early and late features).

    :param layers: ordered tuple of ``(layer_path, indices)`` pairs, where
        ``indices`` is a tuple of unit positions to take from that layer
        (counted within that layer's own neuroids), or ``None`` to take every
        unit of the layer.
    """
    layers: Tuple[Tuple[str, Optional[Tuple[int, ...]]], ...]

    @property
    def layer_path(self) -> str:
        # Canonical (first) layer; the full set is in `layers`.
        return self.layers[0][0]

    @property
    def layer_paths(self) -> Tuple[str, ...]:
        """Every layer this selector draws from, in order."""
        return tuple(lp for lp, _ in self.layers)


def _promote_to_selector(value: Union[str, UnitSelector]) -> UnitSelector:
    """Normalize a ``region_layer_map`` value to a ``UnitSelector``."""
    if isinstance(value, UnitSelector):
        return value
    if isinstance(value, str):
        return LayerSelector(name=value)
    raise TypeError(
        f"region_layer_map values must be str or UnitSelector, "
        f"got {type(value).__name__}"
    )


class UnitSelection(ABC):
    """Resolves to a concrete :class:`Selection` by inspecting a model.

    ``resolve(model)`` uses only the public ``start_recording`` + ``process``
    interface, so the same selection runs against any :class:`Subject`. A
    :class:`StateChange` ``target`` may be either an already-resolved
    ``Selection`` or an unresolved ``UnitSelection``; ``process(StateChange)``
    resolves the latter (snapshotting and restoring the model's recording
    state) before handing off to ``state_change_fn``.
    """

    @abstractmethod
    def resolve(self, model) -> 'Selection':
        ...


@dataclass
class IndexSelection(UnitSelection):
    """Explicit unit indices — the trivial selection. ``resolve`` ignores the model."""
    layer: str
    indices: List[int]

    def resolve(self, model) -> 'Selection':
        return Selection(layer=self.layer, indices=list(self.indices),
                         metadata={'selector': 'index'})


@dataclass
class RandomSelection(UnitSelection):
    """``n_units`` units drawn uniformly at random from ``[0, n_total)`` — the
    causal control for any functional selection.

    Deterministic given ``seed``; needs no forward pass (``resolve`` ignores the
    model). Match ``layer`` / ``n_units`` / ``n_total`` to a
    :class:`FunctionalSelection` to get a same-size random lesion at the same
    layer — the standard "is the effect specific, or just damage?" control.
    ``n_total`` is the layer's unit count, available as
    ``FunctionalSelection`` result ``metadata['n_recorded']``.
    """
    layer: str
    n_units: int
    n_total: int
    seed: int = 0

    def resolve(self, model) -> 'Selection':
        import numpy as np
        k = min(self.n_units, self.n_total)
        rng = np.random.RandomState(self.seed)
        idx = sorted(int(i) for i in rng.choice(self.n_total, size=k, replace=False))
        return Selection(layer=self.layer, indices=idx,
                         metadata={'selector': 'random', 'seed': self.seed,
                                   'n_total': self.n_total})


@dataclass
class FunctionalSelection(UnitSelection):
    """Select units by a functional contrast — the fMRI-localizer analogue.

    Records the model's responses to ``localizer_stimuli`` at
    ``recording_target``, splits presentations into a positive and a negative
    group by their ``contrast_column`` label, scores each unit by the Cohen's d
    of positive vs negative, and keeps the top ``n_units`` (or every unit whose
    score clears ``threshold`` when ``n_units`` is ``None``). This is the
    word-vs-non-word VWFA localizer from the induced-dyslexia experiment, lifted
    out of the sweep scripts into a reusable, model-agnostic object.

    :param recording_target: region to record (a key in the model's
        ``region_layer_map``). The resolved ``Selection.layer`` is taken from
        the recorded neuroids' ``layer`` coord so it matches what the
        ``state_change_fn`` will hook.
    :param contrast: ``(positive_labels, negative_labels)`` matched against
        ``contrast_column`` on the assembly's presentation coords.
    :param sign: ``'positive'`` keeps units that respond MORE to the positive
        group (d > 0), ``'negative'`` the opposite, ``'abs'`` the most
        discriminating either way.
    """
    recording_target: str
    localizer_stimuli: Any
    contrast: Tuple[List[str], List[str]]
    contrast_column: str = 'label'
    n_units: Optional[int] = None
    threshold: float = 2.0
    sign: str = 'positive'

    def resolve(self, model) -> 'Selection':
        import numpy as np
        if self.sign not in ('positive', 'negative', 'abs'):
            raise ValueError(
                f"sign must be 'positive'|'negative'|'abs', got {self.sign!r}")
        model.start_recording(self.recording_target)
        asm = model.process(self.localizer_stimuli)
        if 'neuroid' not in asm.dims or 'presentation' not in asm.dims:
            raise ValueError(
                f"FunctionalSelection needs a (presentation, neuroid) assembly "
                f"from process(); got dims {tuple(asm.dims)}.")
        # collapse any extra dims (e.g. time_bin) to per-(presentation, neuroid)
        extra = [d for d in asm.dims if d not in ('presentation', 'neuroid')]
        if extra:
            asm = asm.mean(extra)
        labels = np.asarray(asm[self.contrast_column].values)
        pos_mask = np.isin(labels, list(self.contrast[0]))
        neg_mask = np.isin(labels, list(self.contrast[1]))
        if pos_mask.sum() == 0 or neg_mask.sum() == 0:
            raise ValueError(
                f"contrast {self.contrast} matched {int(pos_mask.sum())} positive "
                f"/ {int(neg_mask.sum())} negative presentations in column "
                f"{self.contrast_column!r}; both groups must be non-empty.")
        vals = asm.transpose('presentation', 'neuroid').values
        pos, neg = vals[pos_mask], vals[neg_mask]
        mp, mn = pos.mean(0), neg.mean(0)
        sp, sn = pos.std(0, ddof=1), neg.std(0, ddof=1)
        pooled = np.sqrt((sp ** 2 + sn ** 2) / 2.0)
        d = np.divide(mp - mn, pooled, out=np.zeros_like(mp, dtype=float),
                      where=pooled > 0)
        score = {'positive': d, 'negative': -d, 'abs': np.abs(d)}[self.sign]
        if self.n_units is not None:
            order = np.argsort(score)[::-1][:self.n_units]
        else:
            order = np.where(score >= self.threshold)[0]
        idx = sorted(int(i) for i in order)
        # the layer path the state_change_fn hooks — from the recorded coord
        layer_path = self.recording_target
        if 'layer' in asm.coords:
            uniq = list(dict.fromkeys(
                str(x) for x in np.asarray(asm['layer'].values).ravel()))
            if len(uniq) == 1:
                layer_path = uniq[0]
            elif len(uniq) > 1:
                raise ValueError(
                    f"recording {self.recording_target!r} spans {len(uniq)} layers "
                    f"{uniq}; functional localization across a composite region "
                    f"isn't supported yet — record a single-layer region.")
        return Selection(layer=layer_path, indices=idx,
                         metadata={'selector': 'functional', 'sign': self.sign,
                                   'contrast': self.contrast,
                                   'n_recorded': int(vals.shape[1]),
                                   'n_selected': len(idx)})
