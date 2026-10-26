"""Pack conditional graph-action traces with different candidate catalogs."""
import copy

import torch


def pack_candidate_traces(traces):
    if not traces:raise ValueError('No candidate traces to pack')
    counts=[len(t['candidate_features']) for t in traces]
    batches=[len(t['actions']) for t in traces]
    result={key:torch.cat([t[key] for t in traces],dim=0) for key in
            ['features','preferences','masks','actions','old_logp','candidate_features','candidate_families']}
    result['candidate_masks']=torch.zeros((sum(batches),sum(counts)),dtype=torch.bool)
    result['proposal_bias']=torch.zeros((sum(batches),sum(counts)),dtype=traces[0]['proposal_bias'].dtype)
    result['candidate_choices']=torch.empty(sum(batches),dtype=torch.long)
    result['metadata']=[];result['surrogate_parts']=[]
    row=column=0
    for trace,size,count in zip(traces,batches,counts):
        result['candidate_masks'][row:row+size,column:column+count]=trace['candidate_masks']
        result['proposal_bias'][row:row+size,column:column+count]=trace['proposal_bias']
        result['candidate_choices'][row:row+size]=trace['candidate_choices']+column
        for value in trace['metadata']:
            item=copy.deepcopy(value)
            item['local_candidate_index']=item['candidate_index']
            item['candidate_index']+=column
            result['metadata'].append(item)
        result['surrogate_parts'].append(dict(row_start=row,row_count=size,candidate_start=column,
            candidate_count=count,surrogate=copy.deepcopy(trace['surrogate']),
            initial_remaining=copy.deepcopy(trace.get('proposal_initial_remaining'))))
        row+=size;column+=count
    return result
