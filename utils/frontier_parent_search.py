"""Select frontier parents by causal, predicted coverage of their legal edits.

This is a parent-selection heuristic outside the conditional PPO action. It
queries no property oracle and does not import measurements from another run.
Actual rewards and final HV always use the common measured scores.
"""
import copy
import json
from functools import lru_cache

import numpy as np
from rdkit import Chem

from utils.frontier_discovery import molecular_graph
from utils.molecular_size_policy import edit_catalog

SELECTION='surrogate_edit_hv_graph_then_best_parent'


def _edit_identities(parent,structure_cfg,donors=None):
    _,catalog=edit_catalog(parent,structure_cfg,donors=donors)
    identities=[]
    for entries in catalog:
        for mol,_ in entries:
            smiles=Chem.MolToSmiles(mol)
            identities.append((smiles,molecular_graph(smiles)))
    return tuple(identities)


@lru_cache(maxsize=512)
def _cached_edit_identities(mol_block,smiles,encoded_cfg):
    # Version 7 proposals depend only on this scored identity and configuration.
    # Keep immutable graph identifiers, not mutable molecules, predictions,
    # exclusions or measured properties. Dynamic donor catalogs bypass this.
    return _edit_identities(dict(mol_block=mol_block,smiles=smiles),json.loads(encoded_cfg))


def validate_parent_search(settings):
    if (set(settings)!={'version','exploration'} or settings['version']!=1
            or not 0<settings['exploration']<1):
        raise ValueError('Unsupported surrogate parent-search configuration')


def greedy_coverage(coverage,graphs,batch_size,total_probes):
    """Approximate coverage of distinct graph edits in one batch."""
    if not len(coverage):return 0.
    remaining=np.ones(coverage.shape[1]);alive=np.ones(len(graphs),dtype=bool);value=0.
    for _ in range(min(batch_size,len(graphs))):
        scores=coverage@remaining/total_probes;scores[~alive]=-1
        choice=int(scores.argmax())
        if scores[choice]<0:break
        value+=float(scores[choice]);remaining*=1-coverage[choice]
        alive &= np.asarray([g!=graphs[choice] for g in graphs])
    return value


def choose_parent(pool,pocket,seed,settings,structure_cfg,surrogate,blocked_graphs,batch_size):
    validate_parent_search(settings)
    if type(batch_size) is not int or batch_size<1:raise ValueError('Parent search needs a positive batch size')
    pocket=str(pocket);state=surrogate.pockets[pocket]
    trace=dict(settings=copy.deepcopy(settings),batch_size=batch_size,pocket=pocket,
        training_molecules=len(state['training']),observed_attempts=state['observed_attempts'],
        observed_batches=state['observed_batches'],active=False)
    if len(state['training'])<surrogate.cfg['warmup_molecules']:
        trace['fallback']='surrogate_warmup'
        return pool.choose(pocket,seed),trace
    parents=pool.pockets[pocket];smiles=[];indices={};available=[]
    blocked=set(blocked_graphs)
    donors=parents if structure_cfg['version'] in (8,9) else None
    encoded_cfg=json.dumps(structure_cfg,sort_keys=True) if structure_cfg['version']==7 else None
    for parent in parents:
        identities=(_cached_edit_identities(parent['mol_block'],parent['smiles'],encoded_cfg)
                    if encoded_cfg is not None else _edit_identities(parent,structure_cfg,donors))
        local={};excluded=blocked|{parent['graph']}
        for s,g in identities:
            if g in excluded:continue
            if s not in indices:indices[s]=len(smiles);smiles.append(s)
            local[indices[s]]=g
        available.append(local)
    trace.update(unique_candidates=len(smiles),parent_smiles=[r['smiles'] for r in parents],
                 parent_objectives=[r['objectives'] for r in parents],candidate_counts=list(map(len,available)))
    if not smiles:
        trace['fallback']='no_novel_edits'
        return pool.choose(pocket,seed),trace
    context=surrogate.context(pocket,smiles)
    coverage=context['coverage']
    if coverage is None:raise ValueError('Parent-search surrogate unexpectedly inactive')
    scores=np.asarray([greedy_coverage(coverage[list(local)],list(local.values()),batch_size,
                                      len(surrogate.probes)) for local in available])
    graphs=sorted({r['graph'] for r in parents})
    groups=[[i for i,r in enumerate(parents) if r['graph']==graph] for graph in graphs]
    # A graph with several scored stereoisomers/conformers receives one share
    # of exploration, just as in the original graph-first parent selection.
    graph_scores=np.asarray([scores[group].max() for group in groups])
    best_graphs=graph_scores==graph_scores.max()
    exploitation=best_graphs/best_graphs.sum()
    probabilities=(1-settings['exploration'])*exploitation+settings['exploration']/len(graphs)
    rng=np.random.default_rng(seed);graph_index=int(rng.choice(len(graphs),p=probabilities))
    group=groups[graph_index];best=[i for i in group if scores[i]==scores[group].max()]
    index=best[int(rng.integers(len(best)))];selected=copy.deepcopy(parents[index])
    selected.update(selection_seed=int(seed),frontier_size=len(parents),frontier_graphs=len(graphs),
                    selection=SELECTION,selection_config=copy.deepcopy(settings))
    trace.update(active=True,predicted_batch_gains=scores.tolist(),graphs=graphs,
                 graph_probabilities=probabilities.tolist(),selected_parent_index=index)
    return selected,trace


def verify_search_trace(actual,expected):
    """Keep physical identities exact; tolerate only numerical GP differences."""
    numeric={'predicted_batch_gains','graph_probabilities'}
    if {k:v for k,v in actual.items() if k not in numeric}!={k:v for k,v in expected.items() if k not in numeric}:
        raise ValueError('Parent-search context differs from causal history')
    for key in numeric:
        if (key in actual)!=(key in expected):raise ValueError('Incomplete parent-search probability trace')
        if key in expected:np.testing.assert_allclose(actual[key],expected[key],atol=1e-7,rtol=1e-6)
