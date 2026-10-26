"""Causal, sequential parent planning across one unscored proposal batch.

This planner only uses the frozen pre-batch surrogate. Sampled edits reduce
predicted overlap; their measured properties enter history after the batch.
"""
import copy
import json

import numpy as np

from utils.frontier_parent_search import _cached_edit_identities, _edit_identities


def validate_batch_settings(settings):
    if (set(settings)!={'version','exploration'} or settings['version']!=2
            or not 0<settings['exploration']<1):
        raise ValueError('Unsupported multi-parent batch settings')


def batch_context(pool,pocket,settings,structure_cfg,surrogate,blocked_graphs):
    validate_batch_settings(settings)
    pocket=str(pocket);state=surrogate.pockets[pocket]
    context=dict(pocket=pocket,settings=copy.deepcopy(settings),active=False,
                 blocked=set(blocked_graphs),selected=[],surrogate=surrogate,
                 prior_attempts=state['observed_attempts'],prior_batches=state['observed_batches'])
    if len(state['training'])<surrogate.cfg['warmup_molecules']:
        context['fallback']='surrogate_warmup'
        return context
    parents=pool.pockets[pocket];smiles=[];indices={};available=[]
    donors=parents if structure_cfg['version'] in (8,9) else None
    encoded=json.dumps(structure_cfg,sort_keys=True) if structure_cfg['version']==7 else None
    for parent in parents:
        identities=(_cached_edit_identities(parent['mol_block'],parent['smiles'],encoded)
                    if encoded is not None else _edit_identities(parent,structure_cfg,donors))
        local={}
        for s,g in identities:
            if g in context['blocked'] or g==parent['graph']:continue
            if s not in indices:indices[s]=len(smiles);smiles.append(s)
            local[indices[s]]=g
        available.append(local)
    if not smiles:
        context['fallback']='no_novel_edits'
        return context
    prediction=surrogate.context(pocket,smiles)
    if prediction['coverage'] is None:raise ValueError('Unexpected inactive parent surrogate')
    context.update(active=True,parents=parents,smiles=smiles,indices=indices,available=available,
                   coverage=prediction['coverage'],remaining=prediction['remaining'].copy(),
                   total_probes=len(surrogate.probes))
    return context


def choose_next_parent(pool,context,seed):
    trace=dict(settings=copy.deepcopy(context['settings']),pocket=context['pocket'],seed=int(seed),
               slot=len(context['selected']),prior_attempts=context['prior_attempts'],
               prior_batches=context['prior_batches'],active=context['active'])
    if not context['active']:
        trace['fallback']=context['fallback']
        return pool.choose(context['pocket'],seed),trace
    parents=context['parents'];scores=np.full(len(parents),-1.)
    acquisition=context['coverage']@context['remaining']/context['total_probes']
    for i,local in enumerate(context['available']):
        eligible=[index for index,graph in local.items() if graph not in context['blocked']]
        if eligible:scores[i]=float(acquisition[eligible].max())
    groups={}
    for i,parent in enumerate(parents):
        if scores[i]>=0:groups.setdefault(parent['graph'],[]).append(i)
    if not groups:
        trace.update(active=False,fallback='batch_support_exhausted')
        return pool.choose(context['pocket'],seed),trace
    graphs=sorted(groups)
    graph_scores=np.asarray([scores[groups[g]].max() for g in graphs])
    best=graph_scores==graph_scores.max();epsilon=context['settings']['exploration']
    probabilities=(1-epsilon)*best/best.sum()+epsilon/len(graphs)
    rng=np.random.default_rng(seed);group=groups[graphs[int(rng.choice(len(graphs),p=probabilities))]]
    best_indices=[i for i in group if scores[i]==scores[group].max()]
    index=best_indices[int(rng.integers(len(best_indices)))];selected=copy.deepcopy(parents[index])
    selected.update(selection_seed=int(seed),frontier_size=len(parents),frontier_graphs=len({p['graph'] for p in parents}),
        selection='surrogate_edit_hv_graph_then_best_parent',selection_config=copy.deepcopy(context['settings']))
    trace.update(parent_smiles=[p['smiles'] for p in parents],predicted_single_gains=scores.tolist(),
                 graphs=graphs,graph_probabilities=probabilities.tolist(),selected_parent_index=index)
    return selected,trace


def advance_batch(context,smiles,graph,exhausted=False):
    if context['active'] and not exhausted:
        if smiles not in context['indices'] or graph in context['blocked']:
            raise ValueError('Proposed edit is outside the causal batch support')
        context['remaining']*=1-context['coverage'][context['indices'][smiles]]
    context['blocked'].add(graph)
    context['selected'].append(dict(smiles=smiles,graph=graph,exhausted=bool(exhausted)))


def verify_batch_trace(actual,expected):
    numeric={'predicted_single_gains','graph_probabilities'}
    if {k:v for k,v in actual.items() if k not in numeric}!={k:v for k,v in expected.items() if k not in numeric}:
        raise ValueError('Multi-parent trace differs from causal history')
    for key in numeric:
        if (key in actual)!=(key in expected):raise ValueError('Incomplete multi-parent probability trace')
        if key in expected:np.testing.assert_allclose(actual[key],expected[key],atol=1e-7,rtol=1e-6)


def slot_seed(seed,slot):
    return int(np.random.SeedSequence([int(seed),int(slot)]).generate_state(1,dtype=np.uint64)[0])


def propose_parent_batch(pool,policy,pocket,history,preferences,seed,settings,blocked_graphs):
    from utils.frontier_seeds import parent_metadata
    from utils.molecular_batch_trace import pack_candidate_traces
    if policy.cfg['version'] in (8,9):policy.set_donors(pocket,pool.pockets[str(pocket)])
    context=batch_context(pool,pocket,settings,policy.cfg,policy.surrogate,
                          set(blocked_graphs)|policy.proposed_graphs)
    parents=[];resized=[];parts=[];searches=[]
    for slot in range(len(preferences)):
        subseed=slot_seed(seed,slot)
        parent,search=choose_next_parent(pool,context,subseed)
        remaining=context['remaining'].copy() if context['active'] else None
        seeds,part=policy.propose(parent,history,preferences[slot:slot+1],subseed,
                                  excluded_graphs=blocked_graphs,initial_remaining=remaining)
        choice=int(part['candidate_choices'][0]);item=part['metadata'][0]
        advance_batch(context,part['surrogate']['candidate_smiles'][choice],item['proposal_graph'],
                      exhausted=item['novelty_exhausted'])
        parents.append(parent);resized.extend(seeds);parts.append(part);searches.append(search)
    trace=pack_candidate_traces(parts)
    plan=dict(version=2,seed=int(seed),parents=[parent_metadata(p) for p in parents],searches=searches)
    return parents,resized,trace,plan


def verify_parent_plan(actual,expected):
    if ({k:v for k,v in actual.items() if k!='searches'}!=
            {k:v for k,v in expected.items() if k!='searches'}):
        raise ValueError('Multi-parent plan identity differs from causal history')
    if len(actual['searches'])!=len(expected['searches']):raise ValueError('Parent plan length differs')
    for a,b in zip(actual['searches'],expected['searches']):verify_batch_trace(a,b)


def replay_parent_batch(pool,pocket,settings,structure_cfg,surrogate,blocked_graphs,trace,plan,seed):
    """Verify parents, masks and GP priors using no current-batch scores."""
    import torch
    from rdkit import Chem
    from utils.frontier_discovery import molecular_graph
    from utils.frontier_seeds import parent_metadata
    from utils.molecular_size_policy import edit_catalog
    count=len(trace['metadata'])
    if (plan['version']!=2 or plan['seed']!=seed or len(plan['parents'])!=count
            or len(plan['searches'])!=count or len(trace['surrogate_parts'])!=count):
        raise ValueError('Incomplete mixed-parent batch plan')
    context=batch_context(pool,pocket,settings,structure_cfg,surrogate,blocked_graphs)
    parents=[];offset=0
    donors=pool.pockets[str(pocket)] if structure_cfg['version'] in (8,9) else None
    for slot,part in enumerate(trace['surrogate_parts']):
        parent,search=choose_next_parent(pool,context,slot_seed(seed,slot))
        if plan['parents'][slot]!=parent_metadata(parent):raise ValueError('Parent differs from causal slot')
        verify_batch_trace(plan['searches'][slot],search)
        _,catalog=edit_catalog(parent,structure_cfg,donors=donors)
        entries=[(a,m,info) for a,items in enumerate(catalog) for m,info in items]
        smiles=[Chem.MolToSmiles(m) for _,m,_ in entries]
        if (part['row_start']!=slot or part['row_count']!=1 or part['candidate_start']!=offset
                or part['candidate_count']!=len(entries)):
            raise ValueError('Packed candidate catalog boundaries differ')
        local=surrogate.context(pocket,smiles)
        if part['surrogate']!=local['meta']:raise ValueError('Mixed-parent surrogate context differs')
        if context['active']:
            np.testing.assert_allclose(np.asarray(part['initial_remaining']),context['remaining'],atol=1e-8,rtol=1e-6)
            local['remaining']=context['remaining'].copy()
        elif part['initial_remaining'] is not None:
            raise ValueError('Warmup proposal contains an active coverage state')
        graphs=[molecular_graph(s) for s in smiles]
        available=np.asarray([g not in context['blocked'] and g!=parent['graph'] for g in graphs])
        exhausted=not available.any()
        if exhausted:available[0]=True
        columns=slice(offset,offset+len(entries))
        np.testing.assert_array_equal(trace['candidate_masks'][slot,columns].numpy(),available)
        np.testing.assert_array_equal(trace['candidate_families'][columns].numpy(),[a for a,_,_ in entries])
        if trace['candidate_masks'][slot,:offset].any() or trace['candidate_masks'][slot,offset+len(entries):].any():
            raise ValueError('A mixed-parent row can select another parent catalog')
        if (torch.count_nonzero(trace['proposal_bias'][slot,:offset]) or
                torch.count_nonzero(trace['proposal_bias'][slot,offset+len(entries):])):
            raise ValueError('A mixed-parent row has bias outside its candidate catalog')
        choice=int(trace['candidate_choices'][slot])-offset;item=trace['metadata'][slot]
        if (not 0<=choice<len(entries) or not available[choice]
                or item['candidate_index']!=choice+offset or item['local_candidate_index']!=choice
                or bool(item['novelty_exhausted'])!=exhausted or item['proposal_graph']!=graphs[choice]):
            raise ValueError('Mixed-parent chosen edit differs from its support')
        np.testing.assert_allclose(surrogate.bias(local,available),
                                   trace['proposal_bias'][slot,columns].numpy(),atol=1e-6,rtol=0)
        expected=entries[choice][2];actual=item['edit']
        if expected['kind'] in ('fragment_graft','fragment_replace'):
            for key in ('kind','donor_smiles','donor_source','donor_generation','donor_cut','rooted_fragment'):
                if actual[key]!=expected[key]:raise ValueError('Mixed-parent donor provenance differs')
        if expected['kind']=='fragment_replace':
            for key in ('parent_cut','removed_atoms','retained_atoms','anchor','new_atoms','delta'):
                if actual[key]!=expected[key]:raise ValueError('Mixed-parent replacement differs')
        advance_batch(context,smiles[choice],graphs[choice],exhausted=exhausted)
        parents.append(parent);offset+=len(entries)
    if offset!=len(trace['candidate_features']):raise ValueError('Unexplained packed candidate features')
    return parents
