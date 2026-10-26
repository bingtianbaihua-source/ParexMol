"""Joint PPO over edit families and concrete legal graph edits.

Candidate descriptors are input features, never scores or a pre-screening oracle.
Every sampled edit proceeds to diffusion and the common, counted evaluator.
"""
import copy
import itertools

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import Descriptors,rdMolDescriptors,rdFingerprintGenerator
from torch import nn

from utils.frontier_discovery import molecular_graph
from utils.molecular_size_policy import SizePPO,edit_catalog
from utils.molecular_structure_search import prepare_structure_edit


DESCRIPTOR_NAMES=('heavy_atoms','weight','logp','tpsa','donors','acceptors',
    'rotatable_bonds','rings','aromatic_rings','saturated_rings','fraction_sp3',
    'charge','nitrogen_fraction','oxygen_fraction','sulfur_fraction',
    'halogen_fraction','ring_atom_fraction','largest_ring','stereocenters')
FEATURE_NAMES=DESCRIPTOR_NAMES+tuple(f'morgan_radius2_bit_{i}' for i in range(128))
FINGERPRINT=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=128)


def descriptors(mol):
    n=mol.GetNumHeavyAtoms();numbers=[a.GetAtomicNum() for a in mol.GetAtoms()]
    rings=mol.GetRingInfo().AtomRings()
    values=[n/30.,Descriptors.MolWt(mol)/400.,Descriptors.MolLogP(mol)/5.,
        rdMolDescriptors.CalcTPSA(mol)/100.,rdMolDescriptors.CalcNumHBD(mol)/5.,
        rdMolDescriptors.CalcNumHBA(mol)/10.,rdMolDescriptors.CalcNumRotatableBonds(mol)/10.,
        len(rings)/5.,rdMolDescriptors.CalcNumAromaticRings(mol)/5.,
        rdMolDescriptors.CalcNumSaturatedRings(mol)/5.,rdMolDescriptors.CalcFractionCSP3(mol),
        Chem.GetFormalCharge(mol)/3.,numbers.count(7)/n,numbers.count(8)/n,numbers.count(16)/n,
        sum(z in (9,17,35,53) for z in numbers)/n,
        sum(a.IsInRing() for a in mol.GetAtoms())/n,max(map(len,rings),default=0)/8.,
        len(Chem.FindMolChiralCenters(mol,includeUnassigned=True))/5.]
    result=np.asarray(values,dtype=np.float32)
    if not np.isfinite(result).all():raise ValueError('Nonfinite candidate descriptors')
    return result


def graph_features(mol):
    # Connectivity fingerprints distinguish positional isomers that share
    # global descriptors. These encode the molecular graph, not an oracle.
    return np.concatenate([descriptors(mol),FINGERPRINT.GetFingerprintAsNumPy(mol).astype(np.float32)])


class CandidatePPO(SizePPO):
    def __init__(self,cfg,frontier_dim,history_dim,reward_dim,cost_dim,device):
        if cfg['version'] not in (6,7,8,9):raise ValueError('Unsupported candidate PPO version')
        super().__init__(cfg,frontier_dim,history_dim,reward_dim,cost_dim,device)
        context=6+frontier_dim+history_dim+reward_dim
        self.candidate_actor=nn.Sequential(
            nn.Linear(context+2*len(FEATURE_NAMES),cfg['candidate_hidden']),nn.Tanh(),
            nn.Linear(cfg['candidate_hidden'],1,bias=False)).to(device)
        nn.init.zeros_(self.candidate_actor[-1].weight)
        self.optimizer=torch.optim.Adam([
            dict(params=self.actor.parameters(),lr=cfg['actor_lr']),
            dict(params=self.candidate_actor.parameters(),lr=cfg['candidate_lr'])])

    def actor_parameters(self):
        return itertools.chain(self.actor.parameters(),self.candidate_actor.parameters())

    def trace_data(self,trace):
        data=super().trace_data(trace)
        data.update({k:trace[k].to(self.device) for k in
            ('candidate_features','candidate_families','candidate_masks','candidate_choices')})
        return data

    def joint_distribution(self,data):
        family=self.distribution(data['features'],data['preferences'],data['masks'])
        context=torch.cat([data['features'],data['preferences']],dim=-1)
        candidates=data['candidate_features'];batch=len(context);count=len(candidates)
        inputs=torch.cat([context[:,None,:].expand(-1,count,-1),
                          candidates[None,:,:].expand(batch,-1,-1)],dim=-1)
        logits=self.candidate_actor(inputs).squeeze(-1)
        joint=torch.full_like(logits,-torch.inf)
        for action in range(self.action_count):
            indices=data['candidate_families']==action
            if not indices.any():continue
            allowed=data['candidate_masks'][:,indices]
            local=logits[:,indices].masked_fill(~allowed,-torch.inf)
            # Empty families get a finite temporary softmax, then are masked
            # out. This avoids NaN derivatives from softmax([-inf,...,-inf]).
            local=torch.where(allowed.any(-1,keepdim=True),local,torch.zeros_like(local))
            conditional=local.log_softmax(-1).masked_fill(~allowed,-torch.inf)
            joint[:,indices]=family.logits[:,action,None]+conditional
        return family,torch.distributions.Categorical(logits=joint)

    def policy_statistics(self,data):
        family,joint=self.joint_distribution(data)
        return family,joint.log_prob(data['candidate_choices']),joint.entropy()

    def proposal_context(self,parent,entries):
        return None

    def proposal_data(self,data,context):
        return data

    def advance_proposal(self,context,choice):
        pass

    def proposal_trace(self,context,batches):
        return {}

    def catalog(self,parent):
        return edit_catalog(parent,self.cfg)

    @torch.no_grad()
    def propose(self,parent,history,preferences,seed,excluded_graphs=(),initial_remaining=None):
        mol,catalog=self.catalog(parent);n=mol.GetNumHeavyAtoms()
        chemical=[n/self.cfg['max_atoms'],Descriptors.MolWt(mol)/800.,
            sum(a.GetAtomicNum()!=6 for a in mol.GetAtoms())/n,
            rdMolDescriptors.CalcNumRings(mol)/10.,rdMolDescriptors.CalcNumRotatableBonds(mol)/20.,
            Chem.GetFormalCharge(mol)/4.]
        if len(parent['objectives'])!=self.frontier_dim or len(history)!=self.history_dim:
            raise ValueError('Candidate policy parent/history schema mismatch')
        prefs=torch.as_tensor(preferences,dtype=torch.float32,device=self.device).detach()
        features=torch.tensor(chemical+list(parent['objectives'])+list(history),
            dtype=torch.float32,device=self.device).repeat(len(prefs),1)
        entries=[(action,candidate,info) for action,items in enumerate(catalog) for candidate,info in items]
        context=self.proposal_context(parent,entries)
        if initial_remaining is not None:
            remaining=np.asarray(initial_remaining,dtype=float)
            if (context is None or context['coverage'] is None
                    or remaining.shape!=context['remaining'].shape
                    or not np.isfinite(remaining).all() or np.any((remaining<0)|(remaining>1))):
                raise ValueError('Invalid shared batch coverage state')
            context['remaining']=remaining.copy()
        families=torch.tensor([a for a,_,_ in entries],device=self.device)
        original=graph_features(mol);candidate_features=[]
        delta_scale=np.asarray([4.]*len(DESCRIPTOR_NAMES)+[1.]*128,dtype=np.float32)
        for _,candidate,_ in entries:
            values=graph_features(candidate)
            candidate_features.append(np.concatenate([values,delta_scale*(values-original)]))
        candidate_features=torch.tensor(np.stack(candidate_features),device=self.device)
        graphs=[molecular_graph(Chem.MolToSmiles(candidate)) for _,candidate,_ in entries]
        blocked=set(excluded_graphs)|self.proposed_graphs|{molecular_graph(parent['smiles'])}
        generator=torch.Generator(device='cpu').manual_seed(seed)
        rng=np.random.default_rng(seed+170003)
        parents=[];metadata=[];masks=[];candidate_masks=[];choices=[];actions=[];old_logp=[];proposal_batches=[]
        for i in range(len(prefs)):
            available=torch.tensor([g not in blocked for g in graphs],device=self.device)
            exhausted=not bool(available.any())
            if exhausted:available[0]=True
            mask=torch.stack([available[families==a].any() for a in range(self.action_count)])
            data=dict(features=features[i:i+1],preferences=prefs[i:i+1],masks=mask[None],
                candidate_features=candidate_features,candidate_families=families,candidate_masks=available[None])
            data=self.proposal_data(data,context);proposal_batches.append(data)
            family,joint=self.joint_distribution(data)
            choice=int(torch.multinomial(joint.probs[0].cpu(),1,generator=generator))
            self.advance_proposal(context,choice)
            logp=joint.log_prob(torch.tensor([choice],device=self.device))[0]
            action,candidate,info=entries[choice];edit=copy.deepcopy(info)
            blocked.add(graphs[choice]);self.proposed_graphs.add(graphs[choice])
            prepared=prepare_structure_edit(candidate,edit,rng)
            block=Chem.MolToMolBlock(prepared)
            serialized=Chem.MolFromMolBlock(block,removeHs=True,sanitize=True)
            if serialized is None or serialized.GetNumAtoms()!=n+edit['delta']:
                raise ValueError('Candidate action was not executed exactly')
            resized=dict(parent,smiles=Chem.MolToSmiles(serialized),mol_block=block)
            parents.append(resized);masks.append(mask);candidate_masks.append(available)
            actions.append(action);choices.append(choice);old_logp.append(logp)
            metadata.append(dict(action=action,delta=edit['delta'],parent_atoms=n,seed_atoms=n+edit['delta'],
                seed_smiles=resized['smiles'],edit=edit,mask=mask.tolist(),probabilities=family.probs[0].tolist(),
                old_logp=float(logp),seed=int(seed),proposal_graph=graphs[choice],novelty_exhausted=exhausted,
                candidate_index=choice,candidate_probability=float(joint.probs[0,choice]/family.probs[0,action]),
                candidate_count=int(available.sum())))
        trace=dict(features=features.cpu(),preferences=prefs.cpu(),masks=torch.stack(masks).cpu(),
            actions=torch.tensor(actions),old_logp=torch.stack(old_logp).cpu(),metadata=metadata,
            candidate_features=candidate_features.cpu(),candidate_families=families.cpu(),
            candidate_masks=torch.stack(candidate_masks).cpu(),candidate_choices=torch.tensor(choices))
        trace.update(self.proposal_trace(context,proposal_batches))
        if initial_remaining is not None:
            trace['proposal_initial_remaining']=torch.from_numpy(np.asarray(initial_remaining,dtype=float).copy())
        return parents,trace

    def state_dict(self):
        state=super().state_dict()
        state.update(candidate_actor=self.candidate_actor.state_dict(),candidate_feature_schema=list(FEATURE_NAMES))
        return state

    def load_state_dict(self,state):
        if state['candidate_feature_schema']!=list(FEATURE_NAMES):
            raise ValueError('Candidate feature schema changed')
        super().load_state_dict(state)
        self.candidate_actor.load_state_dict(state['candidate_actor'])
