"""Dimension-independent frontier operations and PreCo-style PPO updates.

PCRL/PreCo: https://proceedings.mlr.press/v267/yang25ax.html, equations 6-7,
Appendices E.2 and F.2. We solve in sampled log-density space, retaining the
exact signed PPO clipping derivative for each objective. This is a molecular
PPO adaptation, not a claim of the original paper's convergence guarantees.
"""
import numpy as np
import torch
from scipy.stats import qmc
from torch import nn


def matrix(points, dimension):
    values = np.asarray(points, dtype=float)
    if values.size == 0:
        return np.empty((0, dimension))
    if values.ndim != 2 or values.shape[1] != dimension or not np.isfinite(values).all():
        raise ValueError(f'Expected a finite N x {dimension} objective matrix')
    return values


def nondominated_mask(points):
    return np.array([not np.any(np.all(points >= row, axis=1) & np.any(points > row, axis=1))
                     for row in points], dtype=bool)


def frontier(points, dimension):
    points = np.unique(matrix(points, dimension), axis=0)
    return points[nondominated_mask(points)]


def _box_union(points):
    """Exact dimension-recursive sweep above zero, used through dimension 4."""
    if not len(points):
        return 0.
    if points.shape[1] == 1:
        return float(points.max())
    total, previous = 0., 0.
    for edge in np.unique(points[:, 0]):
        total += (edge - previous) * _box_union(points[points[:, 0] >= edge, 1:])
        previous = edge
    return float(total)


def compare_frontiers(training, generated, reference, samples_power=15, seed=0, integration_upper=None):
    """Exact HV up to 4D; paired, deterministic Sobol estimate above 4D.

    Higher-dimensional comparisons share the SAME integration box and points
    for baseline and union. The reported gain is never a difference of two
    independently sampled estimates. Values below reference still affect
    dominance, but contribute no hypervolume.
    """
    reference = np.asarray(reference, dtype=float)
    if reference.ndim != 1 or not np.isfinite(reference).all():
        raise ValueError('Finite reference vector required')
    d = len(reference)
    old, new = frontier(training, d), frontier(generated, d)
    union = frontier(np.concatenate([old, new]), d)
    shifted = [p - reference for p in (old, new, union)]
    boxes = [p[np.all(p > 0, axis=1)] for p in shifted]
    metadata = {'method': 'exact', 'reference': reference.tolist()}
    if d <= 4:
        old_hv, new_hv, union_hv = [_box_union(p) for p in boxes]
    else:
        if not 5 <= samples_power <= 22:
            raise ValueError('HV Sobol samples_power must be between 5 and 22')
        upper = boxes[2].max(0) if len(boxes[2]) else np.zeros(d)
        if integration_upper is not None:
            upper = np.asarray(integration_upper, dtype=float)-reference
            if upper.shape != reference.shape or not np.isfinite(upper).all() or np.any(upper <= 0):
                raise ValueError('Fixed HV upper bound must be finite and above reference')
            if len(boxes[2]) and np.any(boxes[2] > upper+1e-12):
                raise ValueError('HV coordinates exceed the fixed integration box')
        volume = float(np.prod(upper))
        probes = qmc.Sobol(d, scramble=True, seed=seed).random_base2(samples_power) * upper
        masks = []
        for points in boxes:
            covered = np.zeros(len(probes), dtype=bool)
            for point in points:
                covered |= np.all(probes <= point, axis=1)
            masks.append(covered)
        old_hv, new_hv, union_hv = [float(m.mean() * volume) for m in masks]
        metadata.update(method='sobol_estimate', samples=len(probes), seed=seed,
                        integration_upper=(reference + upper).tolist(), resolution=volume / len(probes))
    additions = [p.tolist() for p in union if not np.any(np.all(old == p, axis=1))]
    return dict(training_hv=old_hv, generated_hv=new_hv, union_hv=union_hv,
                union_hv_gain=max(0., union_hv - old_hv), training_front_size=len(old),
                new_front_points=additions, hv=metadata)


def reference_directions(dimension, count=64, seed=0, floor=0.02):
    if dimension < 1 or count < dimension + 1 or not 0 < floor < 1:
        raise ValueError('Need >= d+1 directions and a positive interior mixture')
    rng = np.random.default_rng(seed)
    directions = np.concatenate([np.eye(dimension), np.ones((1, dimension)) / dimension,
                                 rng.dirichlet(np.ones(dimension), count - dimension - 1)])
    # PreCo divides by preference components; keep every direction interior.
    return (1 - floor) * directions + floor / dimension


class ParetoArchive:
    """Per-pocket nondominated molecules and coverage-driven direction sampling."""
    def __init__(self, schema, directions, constraint_schema=None):
        self.schema = schema
        self.constraint_schema = constraint_schema or []
        self.dimension = len(schema)
        self.reference = np.array([s['reference'] for s in schema])
        self.directions = matrix(directions, self.dimension)
        if not len(self.directions) or np.any(self.directions <= 0):
            raise ValueError('Strictly positive directions required')
        self.entries, self.visits = {}, {}

    def update(self, pocket, records):
        pocket = str(pocket)
        all_rows = self.entries.get(pocket, []) + [dict(r) for r in records
            if r['valid'] and (not self.constraint_schema or r['feasible'])]
        unique = {}
        for row in all_rows:
            vector = matrix([row['objectives']], self.dimension)[0]
            unique.setdefault(tuple(vector), row)
        rows = list(unique.values())
        points = matrix([r['objectives'] for r in rows], self.dimension)
        self.entries[pocket] = [row for row, keep in zip(rows, nondominated_mask(points)) if keep]

    def choose(self, pocket, rng):
        pocket = str(pocket)
        directions = self.directions / np.linalg.norm(self.directions, axis=1, keepdims=True)
        points = matrix([r['objectives'] for r in self.entries.get(pocket, [])], self.dimension)
        points = np.maximum(points - self.reference, 0)
        points = points[np.linalg.norm(points, axis=1) > 0]
        occupancy = np.zeros(len(directions))
        if len(points):
            normalized = points / np.linalg.norm(points, axis=1, keepdims=True)
            occupancy = np.bincount((normalized @ directions.T).argmax(1), minlength=len(directions))
        visits = self.visits.setdefault(pocket, np.zeros(len(directions), dtype=int))
        weights = 1. / ((1 + occupancy) * np.sqrt(1 + visits))
        probabilities = .9 * weights / weights.sum() + .1 / len(weights)
        idx = int(rng.choice(len(weights), p=probabilities))
        visits[idx] += 1
        return self.directions[idx].copy(), dict(direction_index=idx, occupancy=occupancy.tolist(),
                                                 probability=float(probabilities[idx]))

    def state_dict(self):
        return dict(schema=self.schema, directions=self.directions.tolist(), entries=self.entries,
                    constraint_schema=self.constraint_schema,
                    visits={k: v.tolist() for k, v in self.visits.items()})

    @classmethod
    def from_state_dict(cls, state):
        result = cls(state['schema'], state['directions'], state.get('constraint_schema', []))
        result.entries = state['entries']
        result.visits = {k: np.asarray(v, dtype=int) for k, v in state['visits'].items()}
        return result


class VectorCritic(nn.Module):
    def __init__(self, feature_dim, objective_dim, hidden=64, cost_dim=None):
        super().__init__()
        if cost_dim is not None and cost_dim < 1:
            raise ValueError('At least one cost head required for constrained PPO')
        self.scalar_cost = cost_dim is None  # Preserve legacy checkpoint shapes and callers.
        self.body = nn.Sequential(nn.Linear(feature_dim + objective_dim, hidden), nn.Tanh())
        self.values = nn.Linear(hidden, objective_dim)
        self.cost = nn.Linear(hidden, 1 if cost_dim is None else cost_dim)

    def forward(self, features, preference):
        hidden = self.body(torch.cat([features, preference], dim=-1))
        cost = self.cost(hidden)
        return self.values(hidden), cost.squeeze(-1) if self.scalar_cost else cost


class LagrangeMultipliers:
    """Projected dual ascent on mean observed terminal costs, once per rollout.

    This is the integral (plain Lagrangian) update, not a PID controller.
    One global multiplier per constraint budgets expected cost across pockets.
    """
    def __init__(self, names, limits, learning_rate=.05, initial_multiplier=.1):
        self.names = list(names)
        self.limits = np.asarray(limits, dtype=float)
        self.learning_rate = float(learning_rate)
        self.values = np.full(len(self.names), initial_multiplier, dtype=float)
        self.updates = 0
        if (not self.names or len(set(self.names)) != len(self.names)
                or self.limits.shape != (len(self.names),) or not np.isfinite(self.limits).all()
                or np.any(self.limits < 0) or not np.isfinite(self.learning_rate) or self.learning_rate <= 0
                or not np.isfinite(self.values).all() or np.any(self.values < 0)):
            raise ValueError('Invalid Lagrangian names, limits, learning rate or initial multiplier')

    def update(self, costs):
        costs = matrix(costs, len(self.names))
        if not len(costs) or np.any(costs < 0):
            raise ValueError('Dual update needs nonnegative costs for every attempted molecule')
        mean = costs.mean(0)
        before = self.values.copy()
        self.values = np.maximum(0., before + self.learning_rate * (mean - self.limits))
        self.updates += 1
        return dict(names=self.names, mean_costs=mean.tolist(), limits=self.limits.tolist(),
                    multipliers_used=before.tolist(), multipliers_next=self.values.tolist(), updates=self.updates)

    def state_dict(self):
        return dict(names=self.names, limits=self.limits.tolist(), learning_rate=self.learning_rate,
                    values=self.values.tolist(), updates=self.updates)

    def load_state_dict(self, state):
        if (state['names'] != self.names or state['limits'] != self.limits.tolist()
                or state['learning_rate'] != self.learning_rate):
            raise ValueError('Lagrangian state does not match the current constraints/controller')
        values = np.asarray(state['values'], dtype=float)
        if values.shape != self.values.shape or not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError('Invalid saved Lagrange multipliers')
        if not isinstance(state['updates'], int) or state['updates'] < 0:
            raise ValueError('Invalid saved dual update count')
        self.values, self.updates = values.copy(), state['updates']


def min_norm_weights(gradients, max_iter=256, tolerance=1e-9):
    """Frank-Wolfe min ||sum_i alpha_i g_i||², alpha in the simplex."""
    g = gradients.detach().double()
    if g.ndim != 2 or not torch.isfinite(g).all() or not len(g):
        raise ValueError('Finite objective x sample gradients required')
    gram = g @ g.T
    magnitude = gram.abs().max()
    if magnitude == 0:
        return torch.full((len(g),), 1 / len(g), device=g.device, dtype=gradients.dtype)
    gram /= magnitude
    weights = torch.zeros(len(g), device=g.device, dtype=g.dtype)
    weights[torch.diag(gram).argmin()] = 1
    for _ in range(max_iter):
        gradient = gram @ weights
        vertex = torch.zeros_like(weights)
        vertex[gradient.argmin()] = 1
        direction = vertex - weights
        gap = -(direction @ gradient)
        if gap <= tolerance:
            break
        denominator = direction @ gram @ direction
        step = (gap / denominator.clamp_min(1e-30)).clamp(0, 1)
        weights += step * direction
    return weights.to(gradients)


def group_relative_advantages(returns):
    """Per-objective group baseline; constant channels get exactly zero signal."""
    centered=returns-returns.mean(0,keepdim=True)
    scale=returns.std(0,unbiased=False,keepdim=True)
    return torch.where(scale>1e-6,centered/scale.clamp_min(1e-6),torch.zeros_like(centered))


def preco_direction(log_ratio, advantages, returns, preference, reference,
                    clip_epsilon=.2, strength=.1, preference_mix=0.):
    """Return detached ascent direction w.r.t. joint log probabilities.

    A minibatch has one pocket and ONE fixed preference. Each objective has its
    own PPO derivative. The paper's ray-gap alignment vector (Appendix F.2)
    is used explicitly, not autograd through its max operation.
    """
    if advantages.ndim != 2 or advantages.shape != returns.shape:
        raise ValueError('Advantages and returns must be B x objectives')
    if preference.ndim != 1 or len(preference) != advantages.shape[1] or torch.any(preference <= 0):
        raise ValueError('One positive preference vector per update required')
    if not all(torch.isfinite(v).all() for v in (log_ratio, advantages, returns, preference, reference)):
        raise ValueError('Nonfinite PreCo inputs')
    ratio = log_ratio.detach().clamp(-20, 20).exp()
    a = advantages.detach()
    active = ((a >= 0) & (ratio[:, None] <= 1 + clip_epsilon)) | ((a < 0) & (ratio[:, None] >= 1 - clip_epsilon))
    active &= (log_ratio.detach().abs() < 20)[:, None]
    gradients = (ratio[:, None] * a * active / len(a)).T
    value = returns.detach().mean(0) - reference
    gap = ((value / preference).max() * preference - value).clamp_min(0)
    alignment = gap @ gradients
    weights = min_norm_weights(gradients + strength * alignment[None, :])
    if not 0 <= preference_mix <= 1:
        raise ValueError('Preference gradient mixture must be in [0,1]')
    if preference_mix:
        weights = (1-preference_mix)*weights + preference_mix*preference/preference.sum()
    effective = weights + strength * gap
    direction = (effective @ gradients).detach()
    return direction, dict(weights=weights.detach(), alignment_gap=gap.detach(),
                           effective_weights=effective.detach(), direction_norm=direction.norm())
