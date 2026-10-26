"""History-conditioned discovery and shared credit for newly covered HV.

The archive stores stationary property coordinates, never historical rewards.
Each batch is scored against one frozen pre-generation archive. A newly
covered volume element is shared equally among distinct molecular graphs
covering it, and then among their covering poses. Consequently duplicate
samples cannot multiply the batch's discovery reward.
"""
import numpy as np
from rdkit import Chem
from scipy.stats import qmc

from utils.morl import frontier, matrix


def molecular_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError('Discovery needs a valid scored molecular identity')
    Chem.RemoveStereochemistry(mol)
    return Chem.MolToSmiles(mol)


def exclusion_graphs(smiles):
    """Known inputs may include chemical failures, which cannot match a valid generated graph."""
    graphs = set()
    for value in smiles:
        try:
            graphs.add(molecular_graph(value))
        except ValueError:
            # Keep these failures in the original records and attempt counts.
            # Only their unparseable identities are absent from the graph set.
            continue
    return graphs


def shared_hv_credit(history, candidates, reference, groups=None, samples_power=15, seed=0, integration_upper=None):
    """Allocate the *union* HV increment, without double-counting overlap.

    An exact coordinate sweep is used up to 4D; higher dimensions use one
    common Sobol integration grid. Credits sum to the (estimated) batch gain.
    History-dominated points receive zero. Groups identify molecular graphs.
    """
    ref = np.asarray(reference, dtype=float)
    if ref.ndim != 1 or not len(ref) or not np.isfinite(ref).all():
        raise ValueError('A finite nonempty HV reference is required')
    d = len(ref)
    old, new = matrix(history, d)-ref, matrix(candidates, d)-ref
    old = old[np.all(old > 0, axis=1)]
    old = frontier(old, d)
    groups = list(range(len(new))) if groups is None else list(groups)
    if len(groups) != len(new):
        raise ValueError('One graph group per candidate is required')
    _, graph = np.unique(np.asarray(groups, dtype=str), return_inverse=True)
    active = np.flatnonzero(np.all(new > 0, axis=1))
    credit = np.zeros(len(new), dtype=float)

    def share(indices, volume):
        if not len(indices) or volume == 0:
            return
        unique, inverse, counts = np.unique(graph[indices], return_inverse=True, return_counts=True)
        credit[indices] += volume / (len(unique) * counts[inverse])

    def sweep(previous, current, indices, dimension, volume):
        if not len(indices):
            return
        edges = np.unique(np.concatenate(([0.], previous[:, 0], current[:, 0])))
        for lo, hi in zip(edges[:-1], edges[1:]):
            selected = current[:, 0] >= hi
            if not selected.any():
                continue
            occupied = previous[:, 0] >= hi
            width = volume * (hi-lo)
            if dimension == 1:
                if not occupied.any():
                    share(indices[selected], width)
            else:
                sweep(previous[occupied, 1:], current[selected, 1:], indices[selected], dimension-1, width)

    meta = dict(method='exact_shared_coverage', reference=ref.tolist())
    if d <= 4:
        sweep(old, new[active], active, d, 1.)
    else:
        if not 5 <= samples_power <= 22:
            raise ValueError('Sobol samples_power must be between 5 and 22')
        upper = np.maximum(old.max(0) if len(old) else 0.,
                           new[active].max(0) if len(active) else np.zeros(d))
        if integration_upper is not None:
            upper = np.asarray(integration_upper, dtype=float)-ref
            if upper.shape != ref.shape or not np.isfinite(upper).all() or np.any(upper <= 0):
                raise ValueError('Fixed HV upper bound must be finite and above reference')
            if (len(old) and np.any(old > upper+1e-12)) or (len(active) and np.any(new[active] > upper+1e-12)):
                raise ValueError('HV coordinates exceed the fixed integration box')
        volume = float(np.prod(upper))
        probes = qmc.Sobol(d, scramble=True, seed=seed).random_base2(samples_power) * upper
        # Bound memory independently of evaluation sample count.
        for start in range(0, len(probes), 4096):
            part = probes[start:start+4096]
            covered = np.zeros(len(part), dtype=bool)
            for point in old:
                covered |= np.all(part <= point, axis=1)
            remaining = part[~covered]
            if not len(remaining) or not len(active) or volume == 0:
                continue
            masks = np.all(remaining[None, :, :] <= new[active, None, :], axis=-1)
            identifiers = np.unique(graph[active])
            group_counts = np.stack([masks[graph[active] == key].sum(0) for key in identifiers])
            multiplicity = (group_counts > 0).sum(0)
            for key, counts in zip(identifiers, group_counts):
                selected = graph[active] == key
                denominator = np.maximum(multiplicity * counts, 1)
                credit[active[selected]] += (masks[selected] / denominator).sum(1) * volume / len(probes)
        meta.update(method='sobol_shared_coverage', samples=len(probes), seed=seed, integration_upper=(ref+upper).tolist(),
                    resolution=volume/len(probes))
    return credit, dict(batch_hv_gain=float(credit.sum()), **meta)


class DiscoveryArchive:
    """Per-pocket property fronts; only feasible new graphs earn rewards."""
    def __init__(self, schema, constraint_schema=None, samples_power=15, seed=0, history_scope='valid', integration_upper=None,
                 rank_transforms=None):
        self.schema = schema
        self.constraint_schema = constraint_schema or []
        self.reference = np.asarray([s['reference'] for s in schema], dtype=float)
        self.dimension = len(schema)
        if not self.dimension:
            raise ValueError('Discovery requires at least one frontier property')
        self.samples_power, self.seed = samples_power, seed
        self.integration_upper = integration_upper
        if history_scope not in ('valid', 'feasible', 'public_plus_feasible_novel'):
            raise ValueError('Discovery history_scope must be valid or feasible')
        self.history_scope = history_scope
        self.fronts, self.seen, self.statistics = {}, {}, {}
        self.excluded = {}
        self.rank_transforms = rank_transforms
        self.rank_fronts = {}
        if rank_transforms is not None and [s['name'] for s in rank_transforms] != [s['name'] for s in schema]:
            raise ValueError('Unclipped ranking coordinates must follow the frontier schema')

    def rank_coordinates(self, row):
        if self.rank_transforms is None:
            return row['frontier_objectives']
        return matrix([[(float(row[s['property']])-s['origin'])/s['scale'] *
            (1 if s['direction']=='max' else -1) for s in self.rank_transforms]],self.dimension)[0].tolist()

    @property
    def summary_schema(self):
        return (['log_attempts', 'valid_rate', 'feasible_rate', 'log_front_size']
                + [f'{stat}:{s["name"]}' for stat in ('min', 'max', 'mean', 'std') for s in self.schema]
                + [f'mean_cost:{s["name"]}' for s in self.constraint_schema])

    def _statistics(self, pocket):
        return self.statistics.get(str(pocket), dict(attempts=0, valid=0, feasible=0,
                                                    cost_sum=[0.]*len(self.constraint_schema)))

    def summary(self, pocket):
        stats = self._statistics(pocket)
        points = matrix(self.fronts.get(str(pocket), []), self.dimension)-self.reference
        count = max(stats['attempts'], 1)
        values = [np.log1p(stats['attempts']), stats['valid']/count,
                  stats['feasible']/count, np.log1p(len(points))]
        for operation in (np.min, np.max, np.mean, np.std):
            values.extend(operation(points, axis=0) if len(points) else np.zeros(self.dimension))
        values.extend(np.asarray(stats['cost_sum'])/count)
        result = np.asarray(values, dtype=np.float32)
        if len(result) != len(self.summary_schema) or not np.isfinite(result).all():
            raise ValueError('Nonfinite or inconsistent discovery summary')
        return result

    def eligible(self, record):
        return bool(record['valid'] and (not self.constraint_schema or record['feasible']))

    def include_in_history(self, record):
        return bool(record['valid'] and (self.history_scope == 'valid' or self.eligible(record)))

    def initialize(self, pocket, records, excluded_smiles=()):
        if self.history_scope != 'public_plus_feasible_novel':
            self.update(pocket, records)
            return
        pocket = str(pocket)
        if pocket in self.fronts:
            raise ValueError('Public baseline can only be initialized once per pocket')
        self.update(pocket, records)
        # All valid public samples belong to B0 regardless of generated quality gates.
        self.fronts[pocket] = frontier([r['frontier_objectives'] for r in records if r['valid']], self.dimension).tolist()
        self.seen[pocket] = {molecular_graph(r['smiles']) for r in records if r['valid']}
        self.seen[pocket].update(exclusion_graphs(excluded_smiles))
        self.excluded[pocket] = set(self.seen[pocket])
        if self.rank_transforms is not None:
            self.rank_fronts[pocket]=frontier([self.rank_coordinates(r) for r in records if r['valid']],self.dimension).tolist()

    def credit(self, pocket, records, seed=None):
        pocket = str(pocket)
        indices, points, identities = [], [], []
        known = (self.excluded if self.history_scope == 'public_plus_feasible_novel' else self.seen).get(pocket, set())
        for i, row in enumerate(records):
            if self.eligible(row):
                identity = molecular_graph(row['smiles'])
                if identity not in known:
                    indices.append(i)
                    points.append(row['frontier_objectives'])
                    identities.append(identity)
        value, meta = shared_hv_credit(self.fronts.get(pocket, []), points, self.reference,
                                      identities, self.samples_power, self.seed if seed is None or self.integration_upper is not None else seed,
                                      integration_upper=self.integration_upper)
        credits = np.zeros(len(records))
        credits[indices] = value
        return credits, dict(meta, credits=credits.tolist(), eligible_new_graphs=len(set(identities)),
                             history_scope=self.history_scope,
                             history_summary=self.summary(pocket).tolist(),
                             history_front=self.fronts.get(pocket, []))

    def update(self, pocket, records):
        pocket = str(pocket)
        stats = self._statistics(pocket)
        known = self.seen.setdefault(pocket, set())
        points = list(self.fronts.get(pocket, []))
        rank_points = list(self.rank_fronts.get(pocket, []))
        excluded = self.excluded.get(pocket, set())
        for row in records:
            stats['attempts'] += 1
            if self.constraint_schema:
                costs = matrix([row['constraint_costs']], len(self.constraint_schema))[0]
                stats['cost_sum'] = (np.asarray(stats['cost_sum'])+costs).tolist()
            if row['valid']:
                stats['valid'] += 1
                if self.history_scope != 'public_plus_feasible_novel':
                    known.add(molecular_graph(row['smiles']))
            if self.eligible(row):
                stats['feasible'] += 1
            if self.include_in_history(row):
                identity = molecular_graph(row['smiles'])
                if self.history_scope != 'public_plus_feasible_novel' or identity not in excluded:
                    points.append(row['frontier_objectives'])
                    if self.rank_transforms is not None:
                        rank_points.append(self.rank_coordinates(row))
                    if self.history_scope == 'public_plus_feasible_novel':
                        known.add(identity)
        self.statistics[pocket] = stats
        self.fronts[pocket] = frontier(points, self.dimension).tolist()
        if self.rank_transforms is not None:
            self.rank_fronts[pocket] = frontier(rank_points,self.dimension).tolist()

    def state_dict(self):
        state = dict(schema=self.schema, constraint_schema=self.constraint_schema,
                    samples_power=self.samples_power, seed=self.seed, history_scope=self.history_scope, integration_upper=self.integration_upper, fronts=self.fronts,
                    seen={k: sorted(v) for k, v in self.seen.items()}, statistics=self.statistics,
                    excluded={k: sorted(v) for k, v in self.excluded.items()})
        if self.rank_transforms is not None:
            state.update(rank_transforms=self.rank_transforms,rank_fronts=self.rank_fronts)
        return state

    @classmethod
    def from_state_dict(cls, state):
        result = cls(state['schema'], state['constraint_schema'], state['samples_power'], state['seed'],
                     state['history_scope'], integration_upper=state.get('integration_upper'),rank_transforms=state.get('rank_transforms'))
        result.fronts = state['fronts']
        result.seen = {k: set(v) for k, v in state['seen'].items()}
        result.statistics = state['statistics']
        result.excluded = {k: set(v) for k, v in state.get('excluded', {}).items()}
        result.rank_fronts = state.get('rank_fronts',{})
        return result
