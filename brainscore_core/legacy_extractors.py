"""Compatibility for deprecated preprocessor extraction callbacks.

Only old constructor arguments enter this module. The public extractor mapping
uses one calling convention and never inspects callable signatures.
"""

from .io_catalog import canonical_modality


def adapt_legacy_extractors(owner, preprocessors, activations_model):
    """Adapt registration forms once to ``extractor(stimuli, layers=...)``.

    Explicit extractors never use attribute-based capability discovery. Older
    registrations that put extraction functions in preprocessors are adapted by
    their callable signature at this compatibility boundary.
    """
    from collections.abc import Mapping
    import inspect

    if isinstance(activations_model, Mapping):
        extractors = dict(activations_model)
    elif activations_model is not None:
        if not preprocessors:
            raise ValueError('A shared activations_model requires declared preprocessors/modalities.')
        extractors = {key: activations_model for key in preprocessors}
    else:
        extractors = {}
    if any(not callable(value) for value in extractors.values()):
        raise TypeError('Every activations_model extractor must be callable.')
    layer_extractors = set(extractors)
    for modality, preprocessor in preprocessors.items():
        if modality in extractors:
            # Historically the single wrapper applied only to vision; an
            # extractor supplied in another modality takes precedence there.
            if isinstance(activations_model, Mapping) or canonical_modality(modality) == 'vision':
                continue
        if not callable(preprocessor):
            continue
        try:
            signature = inspect.signature(preprocessor)
        except (TypeError, ValueError) as error:
            if modality in extractors:
                continue
            raise TypeError('Declare this callable explicitly in activations_model.') from error
        variadic_wrapper = (not inspect.isfunction(preprocessor) and
                            bool(signature.parameters) and all(
                                p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
                                for p in signature.parameters.values()))
        if 'layers' in signature.parameters or variadic_wrapper:
            extractors[modality] = preprocessor
            layer_extractors.add(modality)
        else:
            try:
                signature.bind(owner._model, object(), recording_layer=None)
                accepts_legacy_call = True
            except TypeError:
                accepts_legacy_call = False
            if modality in extractors and not accepts_legacy_call:
                continue
            import warnings
            warnings.warn(
                'Extraction callbacks in preprocessors are deprecated. Use '
                'activations_model={modality: extractor}, where extractor accepts '
                '(stimuli, layers=...).',
                DeprecationWarning, stacklevel=3)
            def extract(stimuli, *, layers, function=preprocessor):
                if len(layers) > 1:
                    raise ValueError(
                        'This compatibility extraction callback accepts one layer. '
                        'For multiple layers, use activations_model={modality: '
                        'extractor} with an extractor accepting (stimuli, layers=...).')
                return function(owner._model, stimuli,
                                recording_layer=layers[0] if layers else None)
            extractors[modality] = extract
            layer_extractors.discard(modality)
    return extractors, layer_extractors

