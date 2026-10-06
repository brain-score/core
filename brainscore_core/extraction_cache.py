"""Deterministic provenance for activation extraction (no framework imports).

Wrappers declare their extraction settings; custom stateful providers can expose
``cache_config()`` returning immutable configuration, excluding counters and lazy
memoization. Unsupported or cyclic state bypasses caching, never falls back to a
name-only key. File contents are checked at each lookup. Tensor hashes can be
reused within a scoring scope while tracked tensor state remains unchanged.
"""

import dataclasses
import functools
import hashlib
import inspect
import json
import logging
import sys
import types
import weakref
from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

import numpy as np
from result_caching import store_xarray as _store_xarray, is_enabled

logger = logging.getLogger(__name__)
_weight_hashes = ContextVar('brainscore_weight_hashes', default=None)
_HASH_CHUNK_BYTES = 4 * 1024 * 1024


@contextmanager
def weight_fingerprint_scope():
    """Reuse unchanged weight hashes within one scoring run, never across runs.

    Normal tensor edits and storage replacement invalidate entries. Writes via
    .data aliases or NumPy bypass version counters and must occur between runs.
    Outside this scope every lookup reads the full tensor contents.
    """
    token = _weight_hashes.set({})
    try:
        yield
    finally:
        _weight_hashes.reset(token)


class UncacheableConfiguration(ValueError):
    pass


@dataclasses.dataclass(frozen=True)
class FileContent:
    """An input file whose bytes, not only its path, affect extraction."""

    path: str


def file_inputs(paths):
    """Declare file-backed inputs without reading them when caching is disabled."""
    return [FileContent(str(path)) for path in paths]


def _tensor_chunks(value, max_elements):
    """Yield bounded, logical-order slices without flattening the whole tensor."""
    if value.numel() <= max_elements:
        yield value
    elif value.is_contiguous():
        flat = value.view(-1)
        for start in range(0, flat.numel(), max_elements):
            yield flat[start:start + max_elements]
    else:
        row_elements = value[0].numel()
        if row_elements > max_elements:
            for index in range(value.shape[0]):
                yield from _tensor_chunks(value[index], max_elements)
        else:
            rows = max(1, max_elements // max(1, row_elements))
            for start in range(0, value.shape[0], rows):
                yield value[start:start + rows]


def _hash_tensor(tensor):
    """Read tensor bytes in bounded chunks; no memoization in this function."""
    torch = sys.modules['torch']
    digest = hashlib.sha256()
    chunk = max(1, _HASH_CHUNK_BYTES // tensor.element_size())
    for part in _tensor_chunks(tensor.detach(), chunk):
        # Transfer before resolving views so temporary copies stay bounded.
        data = part.cpu().resolve_conj().resolve_neg().contiguous().reshape(-1)
        if data.stride(0) != 1:
            # A one-element view can be "contiguous" while retaining a stride.
            data = data.clone(memory_format=torch.contiguous_format)
        digest.update(data.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _tensor_content(tensor):
    torch = sys.modules['torch']
    if tensor.device.type == 'meta' or tensor.layout != torch.strided or tensor.is_quantized:
        raise UncacheableConfiguration('tensor storage cannot be fingerprinted safely')
    cache = _weight_hashes.get()
    try:
        version = tensor._version
    except RuntimeError:
        # Inference tensors do not have version counters: never memoize them.
        cache, version = None, None
    key = (tensor.data_ptr(), version, tuple(tensor.shape), tuple(tensor.stride()),
           str(tensor.dtype), str(tensor.device), tensor.storage_offset(),
           tensor.is_conj(), tensor.is_neg())
    storage = tensor.untyped_storage()
    entry = cache.get(id(tensor)) if cache is not None else None
    if (entry is not None and entry[0]() is tensor and entry[1]() is storage
            and entry[2] == key):
        digest = entry[3]
    else:
        digest = _hash_tensor(tensor)
        if cache is not None:
            cache[id(tensor)] = (weakref.ref(tensor), weakref.ref(storage), key, digest)
    return {'shape': list(tensor.shape), 'dtype': str(tensor.dtype),
            'sha256': digest}


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
    if isinstance(value, FileContent):
        digest = hashlib.sha256()
        try:
            with open(value.path, 'rb') as handle:
                for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b''):
                    digest.update(chunk)
        except OSError as error:
            raise UncacheableConfiguration(f'cannot read input file {value.path}: {error}') from error
        return {'path': value.path, 'sha256': digest.hexdigest()}
    torch = sys.modules.get('torch')
    if torch is not None and isinstance(value, torch.Tensor):
        return _tensor_content(value)
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
    return 'v2-' + hashlib.sha256(payload.encode('utf-8')).hexdigest()


def extraction_fingerprint(configuration, *, cache_identifier=None):
    """Fail closed for opaque third-party providers, leaving extraction runnable."""
    if cache_identifier is not None and not is_enabled(cache_identifier):
        return None
    try:
        return fingerprint(configuration)
    except UncacheableConfiguration as error:
        logger.warning('Activation cache bypassed: %s', error)
        return None


def model_config(model):
    """Model metadata and state, including installed perturbation hooks.

    Tensor contents are hashed during fingerprinting, not while building this
    description. Runtime eval/train flags are omitted: extraction calls eval().
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
        'tensor_contents': tensors,
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
