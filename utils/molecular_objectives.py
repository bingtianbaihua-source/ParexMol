"""Configurable molecular objectives. All transformed coordinates are maximized.

An evaluator returns ONE raw property, never a weighted total. Custom evaluators
use ``python:package.module:function`` and receive ``(mol, context, **kwargs)``.
The context contains the target, sampled record and experiment (docking/geometry).
"""
import importlib
import json
import math
from dataclasses import asdict, dataclass, field

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors

from utils.evaluation import scoring_func
from utils.evaluation.docking_vina import VinaDockingTask


@dataclass(frozen=True)
class Objective:
    name: str
    evaluator: str
    direction: str = 'max'
    origin: float = 0.
    scale: float = 1.
    target: float | None = None
    reference: float = 0.
    invalid_value: float = -1.
    parameters: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.name or self.direction not in ('max', 'min', 'target'):
            raise ValueError('Objectives need a name and max/min/target direction')
        if not all(math.isfinite(x) for x in (self.origin, self.scale, self.reference, self.invalid_value)) or self.scale <= 0:
            raise ValueError(f'{self.name}: finite constants and positive scale required')
        if self.direction == 'target' and (self.target is None or not math.isfinite(self.target)):
            raise ValueError(f'{self.name}: target direction requires a finite target')
        if self.invalid_value > self.reference:
            raise ValueError(f'{self.name}: invalid_value must not exceed the HV reference')

    def transform(self, value):
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f'{self.name}: nonfinite property')
        if self.direction == 'target':
            return 1. - abs(value - self.target) / self.scale
        return (value - self.origin) / self.scale * (1 if self.direction == 'max' else -1)


class ObjectiveSet:
    def __init__(self, specifications):
        self.items = tuple(Objective(**spec) for spec in specifications)
        self.names = [item.name for item in self.items]
        if not self.items or len(set(self.names)) != len(self.names):
            raise ValueError('At least one uniquely named objective is required')
        self.evaluators = [resolve_evaluator(item.evaluator) for item in self.items]

    def __len__(self):
        return len(self.items)

    @property
    def schema(self):
        return [asdict(item) for item in self.items]

    @property
    def reference(self):
        return np.array([item.reference for item in self.items])

    @property
    def invalid(self):
        return [item.invalid_value for item in self.items]

    def evaluate(self, mol, context):
        raw, vector = {}, []
        for item, evaluator in zip(self.items, self.evaluators):
            value = evaluate_cached(item.evaluator, evaluator, mol, context, item.parameters)
            raw[item.name] = value
            vector.append(item.transform(value))
        if not np.isfinite(vector).all():
            raise ValueError('Nonfinite transformed objective')
        return raw, vector

    def require_compatible(self, schema):
        if self.schema != schema:
            raise ValueError('Objective schema/order differs from checkpoint or baseline; use a new run and --warm-start for expansion')


def evaluate_cached(name, evaluator, mol, context, parameters):
    """Reuse a measurement across objectives/constraints with identical parameters."""
    key = (name, json.dumps(parameters, sort_keys=True, allow_nan=False))
    cache = context.setdefault('property_cache', {})
    if key not in cache:
        value = float(evaluator(mol, context, **parameters))
        if not math.isfinite(value):
            raise ValueError(f'{name}: nonfinite property')
        cache[key] = value
    return cache[key]


def _chem(mol, context, name):
    if 'chem' not in context:
        context['chem'] = scoring_func.get_chem(mol)
    return context['chem'][name]


def geometry_penalty(mol, context):
    if 'geometry_penalty' not in context:
        context['geometry_penalty'] = float(context['experiment'].amber.amber_energy(mol))
    return context['geometry_penalty']


def vina_min(mol, context):
    if 'vina_min' not in context:
        from utils.runtime import ROOT
        task = VinaDockingTask.from_generated_mol(
            mol, context['target']['source_ligand'],
            protein_root=str(ROOT / 'data/crossdocked_v1.1_rmsd1.0'),
            tmp_dir=str(context['folder'] / 'docking_tmp'))
        context['vina_min'] = float(task.run(mode='minimize', exhaustiveness=16, seed=2021, cpu=2)[0]['affinity'])
    return context['vina_min']


def resolve_evaluator(name):
    if name == 'validity':
        return lambda mol, context: 1.  # Called only after successful reconstruction/sanitization.
    if name in ('qed', 'sa', 'logp', 'lipinski'):
        return lambda mol, context: _chem(mol, context, name)
    if name == 'vina_min':
        return vina_min
    if name == 'geometry_penalty':
        return geometry_penalty
    if name.startswith('rdkit:'):
        function = getattr(Descriptors, name.split(':', 1)[1], None)
        if not callable(function):
            raise ValueError(f'Unknown RDKit descriptor: {name}')
        return lambda mol, context, **kwargs: function(mol, **kwargs)
    if name.startswith('python:'):
        module, function = name[len('python:'):].rsplit(':', 1)
        evaluator = getattr(importlib.import_module(module), function)
        if not callable(evaluator):
            raise ValueError(f'Not a callable evaluator: {name}')
        return evaluator
    raise ValueError(f'Unknown objective evaluator: {name}')


def score_molecule(record, target, folder, experiment, objectives, invalid_cost=10., constraints=None,
                   frontier_objectives=None):
    """Retain failed attempts; no fabricated measurements or scalar reward."""
    result = dict(valid=False, objectives=objectives.invalid, raw_properties={}, cost=float(invalid_cost))
    if constraints:
        result.update(constraints.failed())
    if frontier_objectives is not None:
        result.update(frontier_objectives=frontier_objectives.invalid, raw_frontier_properties={})
    try:
        mol = Chem.Mol(record['mol']) if record.get('mol') is not None else None
        if mol is None:
            raise ValueError('Molecule reconstruction failed')
        Chem.SanitizeMol(mol)
        smiles = Chem.MolToSmiles(mol)
        if '.' in smiles:
            raise ValueError('Disconnected molecule')
        context = dict(experiment=experiment, target=target, record=record, folder=folder)
        raw, vector = objectives.evaluate(mol, context)
        cost = geometry_penalty(mol, context)
        if not math.isfinite(cost) or cost < 0:
            raise ValueError('Invalid geometry cost')
        constraint_result = constraints.evaluate(mol, context) if constraints else {}
        if frontier_objectives is not None:
            frontier_raw, frontier_vector = frontier_objectives.evaluate(mol, context)
            result.update(frontier_objectives=frontier_vector, raw_frontier_properties=frontier_raw)
        result.update(valid=True, smiles=smiles, objectives=vector, raw_properties=raw, cost=cost)
        result.update(constraint_result)
    except (ValueError, RuntimeError, IndexError, KeyError, OSError) as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
    return result
