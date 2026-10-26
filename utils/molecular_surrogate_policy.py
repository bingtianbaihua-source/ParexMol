"""PPO with a frozen, causal vector-surrogate proposal prior."""
import copy

import torch
from rdkit import Chem

from utils.frontier_surrogate import SurrogateHistory
from utils.molecular_candidate_policy import CandidatePPO


class SurrogatePPO(CandidatePPO):
    def __init__(self,cfg,frontier_dim,history_dim,reward_dim,cost_dim,device):
        if cfg['version'] not in (7,8,9):raise ValueError('Unsupported surrogate PPO version')
        super().__init__(cfg,frontier_dim,history_dim,reward_dim,cost_dim,device)
        self.surrogate=None

    def initialize_surrogate(self,public,reference,upper,excluded_smiles=()):
        self.surrogate=SurrogateHistory(self.cfg['surrogate'],reference,upper)
        for pocket,rows in public.items():self.surrogate.initialize(pocket,rows,excluded_smiles)

    def observe_surrogate(self,pocket,rows):
        self.surrogate.observe(pocket,rows)

    def trace_data(self,trace):
        data=super().trace_data(trace)
        data['proposal_bias']=trace['proposal_bias'].to(self.device)
        return data

    def joint_distribution(self,data):
        _,joint=super().joint_distribution(data)
        joint=torch.distributions.Categorical(logits=joint.logits+data['proposal_bias'])
        # Guidance changes family marginals too. Metadata and PPO diagnostics
        # must use those marginals, not the unguided family actor distribution.
        masses=torch.stack([joint.probs[:,data['candidate_families']==action].sum(-1)
                            for action in range(self.action_count)],dim=-1)
        family=torch.distributions.Categorical(probs=masses)
        return family,joint

    def proposal_context(self,parent,entries):
        if self.surrogate is None:raise ValueError('Initialize the surrogate from public measurements first')
        return self.surrogate.context(parent['pocket'],[Chem.MolToSmiles(mol) for _,mol,_ in entries])

    def proposal_data(self,data,context):
        bias=self.surrogate.bias(context,data['candidate_masks'][0].cpu().numpy())
        data['proposal_bias']=torch.as_tensor(bias,device=self.device)[None]
        return data

    def advance_proposal(self,context,choice):
        self.surrogate.advance(context,choice)

    def proposal_trace(self,context,batches):
        return dict(surrogate=copy.deepcopy(context['meta']),
                    proposal_bias=torch.cat([x['proposal_bias'] for x in batches]).cpu())

    def state_dict(self):
        state=super().state_dict()
        state['surrogate']=self.surrogate.state_dict()
        return state

    def load_state_dict(self,state):
        super().load_state_dict(state)
        self.surrogate=SurrogateHistory.from_state_dict(state['surrogate'])
        if self.surrogate.dimension!=self.frontier_dim or self.surrogate.cfg!=self.cfg['surrogate']:
            raise ValueError('Surrogate checkpoint schema changed')
