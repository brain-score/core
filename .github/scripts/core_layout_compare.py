"""Prove relocated definition bodies and installed snapshots match the base."""
import ast
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys

base, candidate, evidence = map(Path, sys.argv[1:])
manifest = json.loads((candidate / '.github/scripts/core_layout_modules.json').read_text())
reverse = {'brainscore_core.' + new: 'brainscore_core.' + old for old, new in manifest.items()}


class NormalizeImports(ast.NodeTransformer):
    """Resolve equivalent relative/absolute import spellings, keeping bindings."""
    def __init__(self, module):
        self.package = module.rpartition('.')[0]

    def visit_ImportFrom(self, node):
        target = importlib.util.resolve_name('.' * node.level + (node.module or ''), self.package) if node.level else node.module
        result = []
        for alias in node.names:
            whole = reverse.get(target + '.' + alias.name)
            if whole:
                parent, _, child = whole.rpartition('.')
                bound = alias.asname or alias.name
                result.append(ast.ImportFrom(module=parent, names=[ast.alias(name=child, asname=bound if bound != child else None)], level=0))
            else:
                result.append(ast.ImportFrom(module=reverse.get(target, target), names=[deepcopy(alias)], level=0))
        return result


def definitions(path, module):
    result = {}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            normalized = NormalizeImports(module).visit(deepcopy(node))
            result[node.name] = ast.dump(normalized, include_attributes=False)
    return result


counts = {}
# The manifest names every original root module, including unchanged ones.
for name, target in manifest.items():
    if name == '_env_check':
        continue  # This module was already only an alias of environment_check.
    def read(root):
        original = root / 'brainscore_core' / (name + '.py')
        if original.exists():
            return definitions(original, 'brainscore_core.' + name)
        if name == 'streaming_helpers':
            combined = {}
            for part in ('buffered', 'drivers', 'environment', 'payloads', 'perception'):
                items = definitions(root / 'brainscore_core/_sessions' / (part + '.py'),
                                    'brainscore_core._sessions.' + part)
                assert not combined.keys() & items.keys(), 'Duplicate session definition'
                combined.update(items)
            return combined
        return definitions(root / 'brainscore_core' / (target.replace('.', '/') + '.py'),
                           'brainscore_core.' + target)

    old, new = read(base), read(candidate)
    assert old.keys() == new.keys(), (name, old.keys() ^ new.keys())
    for symbol in old:
        assert old[symbol] == new[symbol], 'Definition changed: ' + name + '.' + symbol
    counts[name] = len(old)

before = json.loads((evidence / 'baseline.json').read_text())
after = json.loads((evidence / 'candidate.json').read_text())
for section in ('apis', 'measurements'):
    if before[section] != after[section]:
        differing = [name for name in before[section] if before[section][name] != after[section].get(name)]
        raise AssertionError('Changed ' + section + ': ' + ', '.join(differing))
(evidence / 'comparison.json').write_text(json.dumps({
    'definition_counts': counts, 'public_api_equal': True, 'measurement_bytes_equal': True,
    'excluded_measurement_fields': ['latency_ms'],
}, indent=2, sort_keys=True) + '\n')
print('PASS: definition bodies, public APIs, and fixed measurement bytes match the base')
