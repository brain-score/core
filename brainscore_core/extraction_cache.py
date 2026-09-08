"""Deterministic provenance for activation extraction (no framework imports).

Wrappers declare their extraction settings; custom stateful providers can expose
``cache_config()`` returning immutable configuration, excluding counters and lazy
memoization. Unsupported or cyclic state bypasses caching, never falls back to a
name-only key. Checkpoint contents remain the responsibility of ``backbone_id``:
use a new identifier/revision when replacing weights.
"""

import dataclasses
import functools
import hashlib
import inspect
import json
import logging
import sys
import types
from collections.abc import Mapping
from pathlib import Path

import numpy as np
from result_caching import store_xarray as _store_xarray, is_enabled

logger = logging.getLogger(__name__)


class UncacheableConfiguration(ValueError):
    pass


def _name(value):
    cls = value if isinstance(value, type) else type(value)
    return f'{cls.__module__}.{cls.__qualname__}'


def _version(value):
    package = type(value).__module__.split('.')[0]
    return getattr(sys.modules.get(package), '__version__', None)


def _canonical(value, active):
    if len(active) > 64:
        raise UncacheableConfiguration('provider state is too deeply nested; define cache_config()')
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, np.generic):
        return {'numpy_scalar': str(value.dtype), 'value': _canonical(value.item(), active)}
    if isinstance(value, Path):
        return {'path': str(value)}
    if isinstance(value, np.dtype):
        return {'dtype': str(value)}
    if isinstance(value, bytes):
        return {'bytes': value.hex()}
    if value is Ellipsis:
        return {'ellipsis': True}
    if isinstance(value, types.ModuleType):
        return {'module': value.__name__, 'version': getattr(value, '__version__', None)}
    if isinstance(value, type):
        return {'class': _name(value)}
    if id(value) in active:
        raise UncacheableConfiguration(f'cyclic state in {_name(value)}; define cache_config()')
    active = active | {id(value)}
    convert = lambda item: _canonical(item, active)
    if dataclasses.is_dataclass(value):
        return {'class': _name(value), 'fields': convert({
            field.name: getattr(value, field.name) for field in dataclasses.fields(value)})}
    if isinstance(value, Mapping):
        pairs = [(convert(k), convert(v)) for k, v in value.items()]
        return {'mapping': sorted(pairs, key=lambda pair: json.dumps(pair[0], sort_keys=True))}
    if isinstance(value, (list, tuple)):
        return {type(value).__name__: [convert(item) for item in value]}
    if isinstance(value, (set, frozenset)):
        return {'set': sorted((convert(item) for item in value), key=lambda x: json.dumps(x, sort_keys=True))}
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            return {'shape': list(value.shape), 'dtype': str(value.dtype), 'values': convert(value.tolist())}
        return {'shape': list(value.shape), 'dtype': str(value.dtype),
                'sha256': hashlib.sha256(value.tobytes()).hexdigest()}
    if type(value).__name__ == 'AddedToken' and type(value).__module__ == 'tokenizers':
        return {'added_token': convert({key: getattr(value, key) for key in
                ('content', 'single_word', 'lstrip', 'rstrip', 'normalized', 'special')})}
    if isinstance(value, types.CodeType):
        # Exclude paths, line numbers, and debug tables, which vary by checkout.
        return convert((value.co_code, value.co_consts, value.co_names,
                        value.co_varnames, value.co_freevars, value.co_cellvars,
                        value.co_argcount, value.co_kwonlyargcount, value.co_flags))
    if isinstance(value, functools.partial):
        return {'partial': convert(value.func), 'args': convert(value.args),
                'kwargs': convert(value.keywords)}
    if inspect.ismethod(value):
        return {'method': convert(value.__func__), 'owner': convert(value.__self__)}
    if inspect.isfunction(value):
        closure = inspect.getclosurevars(value)
        return {'function': value.__qualname__, 'code': convert(value.__code__),
                'defaults': convert(value.__defaults__), 'kwdefaults': convert(value.__kwdefaults__),
                'closure': convert(closure.nonlocals), 'globals': convert(closure.globals)}
    if inspect.isbuiltin(value):
        return {'builtin': f'{value.__module__}.{value.__qualname__}'}
    if inspect.getattr_static(value, 'cache_config', None) is not None:
        return {'class': _name(value), 'config': convert(value.cache_config())}
    # Tokenizer backend padding/truncation are mutated by each encode call.
    # Their *requested* settings live on the tokenizer, not in that runtime state.
    if inspect.getattr_static(value, 'get_vocab', None) is not None:
        backend = getattr(value, 'backend_tokenizer', None)
        backend = json.loads(backend.to_str()) if backend is not None else None
        if backend is not None:
            backend.pop('padding', None)
            backend.pop('truncation', None)
        return {'class': _name(value), 'version': _version(value), 'vocab': convert(value.get_vocab()),
                'backend': convert(backend),
                'bpe_ranks': convert(getattr(value, 'bpe_ranks', None)),
                'settings': convert({key: getattr(value, key, None) for key in (
                    'init_kwargs', 'special_tokens_map', 'padding_side', 'truncation_side',
                    'model_max_length', 'clean_up_tokenization_spaces', 'split_special_tokens',
                    'add_prefix_space', 'do_lower_case', 'do_basic_tokenize', 'strip_accents', 'errors')})}
    if inspect.getattr_static(value, 'to_dict', None) is not None:
        # ProcessorMixin.to_dict() may omit its tokenizer/image processor.
        components = {key: getattr(value, key) for key in (
            'tokenizer', 'image_processor', 'feature_extractor', 'video_processor', 'chat_template')
                      if getattr(value, key, None) is not None}
        return {'class': _name(value), 'version': _version(value), 'config': convert(value.to_dict()),
                'components': convert(components)}
    if hasattr(value, '__dict__') and callable(value):
        return {'class': _name(value), 'call': convert(type(value).__call__),
                'state': convert(vars(value))}
    raise UncacheableConfiguration(f'unsupported {_name(value)}; define cache_config()')


def fingerprint(configuration):
    """Return a stable, versioned key; never use repr() or object addresses."""
    payload = json.dumps(_canonical(configuration, set()), sort_keys=True,
                         separators=(',', ':'), ensure_ascii=True)
    return 'v1-' + hashlib.sha256(payload.encode('utf-8')).hexdigest()


def extraction_fingerprint(configuration):
    """Fail closed for opaque third-party providers, leaving extraction runnable."""
    try:
        return fingerprint(configuration)
    except UncacheableConfiguration as error:
        logger.warning('Activation cache bypassed: %s', error)
        return None


def model_config(model):
    """Lightweight model metadata, including installed perturbation hooks.

    Never copy/hash tensors or load weights. Runtime eval/train flags are omitted:
    the extraction wrappers explicitly call eval() before every forward pass.
    """
    if model is None:
        return None
    modules = list(model.named_modules()) if hasattr(model, 'named_modules') else []
    tensors = (list(model.parameters()) + list(getattr(model, 'buffers', lambda: [])())
               if hasattr(model, 'parameters') else [])
    torch = sys.modules.get('torch')
    return {
        'class': _name(model),
        'config': getattr(model, 'config', None),
        'custom_config': (model.cache_config()
                          if inspect.getattr_static(model, 'cache_config', None) is not None else None),
        'structure': [(name, _name(module), module.extra_repr() if hasattr(module, 'extra_repr') else None)
                      for name, module in modules],
        'tensor_layout': [(tuple(tensor.shape), str(tensor.dtype), str(tensor.device)) for tensor in tensors],
        'dtypes': sorted({str(tensor.dtype) for tensor in tensors}),
        'devices': sorted({str(tensor.device) for tensor in tensors}),
        'torch_version': getattr(torch, '__version__', None),
        'implementation_version': _version(model),
        'matmul_precision': torch.get_float32_matmul_precision() if torch is not None else None,
        'hooks': [(name, list(getattr(module, kind, {}).values()))
                  for name, module in modules
                  for kind in ('_forward_pre_hooks', '_forward_hooks')
                  if getattr(module, kind, {})],
    }


def wrapper_config(wrapper, fields):
    """Read settings at lookup time, so post-construction edits also invalidate."""
    return {'implementation': implementation_config(type(wrapper)),
            'settings': {field: getattr(wrapper, field) for field in fields},
            'input_device': str(getattr(wrapper, '_device', None)),
            'model': model_config(wrapper._model)}


def implementation_config(cls):
    """Include inherited extraction code, including notebook-defined subclasses."""
    implementations = []
    for base in cls.__mro__:
        if base is object:
            continue
        try:
            code = inspect.getsource(base)
        except (OSError, TypeError):
            code = {name: member.__code__ for name, member in vars(base).items()
                    if inspect.isfunction(member)}
        implementations.append((_name(base), code))
    return implementations


class store_xarray(_store_xarray):
    """Existing xarray layer merging, with visible unversioned-cache misses."""

    def get_function_identifier(self, function, call_args):
        identifier = super().get_function_identifier(function, call_args)
        if (call_args.get('extraction_fingerprint') and is_enabled(identifier)
                and not self.is_stored(identifier)):
            old_args = {k: v for k, v in call_args.items() if k != 'extraction_fingerprint'}
            old_identifier = super().get_function_identifier(function, old_args)
            if self.is_stored(old_identifier):
                logger.info('Skipping unversioned activation cache; recomputing with configuration fingerprint: %s',
                            old_identifier)
            else:
                logger.info('Activation cache miss: %s; unversioned entries are retained but cannot be reused',
                            identifier)
        return identifier
