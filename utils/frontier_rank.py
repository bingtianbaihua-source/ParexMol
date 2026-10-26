"""Dense, per-objective rank feedback against the frozen pre-rollout frontier."""
import numpy as np

from utils.frontier_discovery import molecular_graph
from utils.morl import matrix


def rank_feedback(archive, pocket, records, temperature=None):
    """Midranks retain a vector; only feasible previously unseen graphs earn credit.

    Inspired by PMMG's per-objective archive win fractions. Here ties count
    one half, the reference is frozen before the batch, and duplicate graph
    credit is shared. Neither rank rewards nor current candidates enter B0/HV.
    """
    pocket = str(pocket)
    fronts=archive.rank_fronts if archive.rank_transforms is not None else archive.fronts
    front = matrix(fronts.get(pocket, []), archive.dimension)
    if temperature is not None and (not np.isfinite(temperature) or temperature<=0):
        raise ValueError('Soft rank needs a positive finite temperature')
    known = archive.seen.get(pocket, set())
    groups = {}
    ranks = np.zeros((len(records), archive.dimension), dtype=float)
    for i, row in enumerate(records):
        if not archive.eligible(row):
            continue
        identity = molecular_graph(row['smiles'])
        if identity in known:
            continue
        groups.setdefault(identity, []).append(i)
        point = matrix([archive.rank_coordinates(row)], archive.dimension)[0]
        if temperature is None:
            ranks[i] = ((point > front).mean(0) + .5*(point == front).mean(0)
                        if len(front) else np.full(archive.dimension, .5))
        else:
            ranks[i] = ((.5+.5*np.tanh((point-front)/(2*temperature))).mean(0)
                        if len(front) else np.full(archive.dimension,.5))
    raw = ranks.copy()
    for indices in groups.values():
        ranks[indices] /= len(indices)
    return ranks, dict(method='per_objective_midrank' if temperature is None else 'unclipped_soft_rank', ties=.5,
        **({} if temperature is None else dict(temperature=temperature,unclipped=True)),
        reference='frozen_pre_rollout_frontier', history_front=front.tolist(),
        novelty='previously_unseen_feasible_graphs', duplicate_credit='divide_by_graph_multiplicity',
        raw_ranks=raw.tolist(), credits=ranks.tolist(),
        nonzero_attempts=int(np.any(ranks > 0, axis=1).sum()))
