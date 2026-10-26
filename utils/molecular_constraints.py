"""Raw-property constraints, separate from the maximized objective vector.

Thresholds define individual feasibility; limits budget expected costs over
ALL attempted molecules, including failed reconstruction/scoring attempts.
"""
import math
from dataclasses import asdict, dataclass, field

import numpy as np

from utils.molecular_objectives import evaluate_cached, resolve_evaluator


@dataclass(frozen=True)
class Constraint:
    name: str
    evaluator: str
    relation: str
    threshold: float
    limit: float
    cost_type: str = 'indicator'
    scale: float = 1.
    invalid_cost: float = 1.
    parameters: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.name or self.relation not in ('ge', 'le') or self.cost_type not in ('indicator', 'hinge'):
            raise ValueError('Constraints require a name, ge/le relation and indicator/hinge cost')
        if not all(math.isfinite(x) for x in (self.threshold, self.limit, self.scale, self.invalid_cost)):
            raise ValueError(f'{self.name}: finite constraint constants required')
        if self.scale <= 0 or self.limit < 0 or self.invalid_cost <= self.limit:
            raise ValueError(f'{self.name}: positive scale and invalid_cost > limit >= 0 required')
        if self.cost_type == 'indicator' and (self.invalid_cost != 1 or self.limit >= 1):
            raise ValueError(f'{self.name}: indicator costs require invalid_cost=1 and limit < 1')

    def cost(self, value):
        if not math.isfinite(float(value)):
            raise ValueError(f'{self.name}: nonfinite constraint measurement')
        excess = (self.threshold - value if self.relation == 'ge' else value - self.threshold) / self.scale
        violated = excess > 0
        return (float(violated) if self.cost_type == 'indicator' else max(0., excess)), violated


class ConstraintSet:
    def __init__(self, specifications=()):
        self.items = tuple(Constraint(**spec) for spec in specifications)
        self.names = [item.name for item in self.items]
        if len(set(self.names)) != len(self.names):
            raise ValueError('Constraint names must be unique')
        self.evaluators = [resolve_evaluator(item.evaluator) for item in self.items]

    def __len__(self):
        return len(self.items)

    @property
    def schema(self):
        return [asdict(item) for item in self.items]

    @property
    def limits(self):
        return [item.limit for item in self.items]

    def failed(self):
        return dict(feasible=False, constraint_values={},
                    constraint_costs=[item.invalid_cost for item in self.items],
                    constraint_violations=[True] * len(self))

    def evaluate(self, mol, context):
        raw, costs, violations = {}, [], []
        for item, evaluator in zip(self.items, self.evaluators):
            value = evaluate_cached(item.evaluator, evaluator, mol, context, item.parameters)
            cost, violated = item.cost(value)
            raw[item.name] = value
            costs.append(cost)
            violations.append(violated)
        return dict(feasible=not any(violations), constraint_values=raw,
                    constraint_costs=costs, constraint_violations=violations)

    def require_compatible(self, schema):
        if self.schema != schema:
            raise ValueError('Constraint schema/order differs; prepare a new run and use --warm-start')

    def summarize(self, records):
        if not records:
            raise ValueError('Constraint statistics require attempted molecules')
        costs = np.asarray([r['constraint_costs'] for r in records], dtype=float)
        if costs.shape != (len(records), len(self)) or not np.isfinite(costs).all() or np.any(costs < 0):
            raise ValueError('Constraint costs must be a finite nonnegative attempts x constraints matrix')
        return dict(feasible=sum(r['valid'] and r['feasible'] for r in records),
                    feasible_rate=sum(r['valid'] and r['feasible'] for r in records) / len(records),
                    constraints={item.name: dict(mean_cost=float(costs[:, j].mean()), limit=item.limit,
                        violation_rate=float(np.mean([r['constraint_violations'][j] for r in records])))
                        for j, item in enumerate(self.items)})
