import copy
import unittest
from types import SimpleNamespace

import numpy as np
import torch
from rdkit import Chem,RDLogger

from utils.molecular_candidate_policy import CandidatePPO,graph_features,DESCRIPTOR_NAMES
from utils.molecular_structure_search import CANDIDATE_STRUCTURE_POLICY
from test_molecular_size_policy import parent


class CandidatePolicyTests(unittest.TestCase):
    def setUp(self):
        RDLogger.DisableLog('rdApp.error');RDLogger.DisableLog('rdApp.warning')
        torch.manual_seed(23)
        self.parent=parent('CC1CCC(O)CC1CCO')
        self.parent['smiles']=Chem.MolToSmiles(Chem.MolFromMolBlock(self.parent['mol_block']))
        self.policy=CandidatePPO(CANDIDATE_STRUCTURE_POLICY,5,4,11,2,'cpu')
        self.preferences=torch.full((8,11),1/11)

    def tearDown(self):
        RDLogger.EnableLog('rdApp.error');RDLogger.EnableLog('rdApp.warning')

    def test_joint_probabilities_factorize_and_replay_changing_masks(self):
        _,trace=self.policy.propose(self.parent,[.1]*4,self.preferences,42)
        data=self.policy.trace_data(trace);family,joint=self.policy.joint_distribution(data)
        torch.testing.assert_close(joint.probs.sum(-1),torch.ones(8))
        torch.testing.assert_close(joint.log_prob(trace['candidate_choices']),trace['old_logp'])
        for action in range(self.policy.action_count):
            chosen=trace['candidate_families']==action
            torch.testing.assert_close(joint.probs[:,chosen].sum(-1),family.probs[:,action],atol=1e-6,rtol=1e-5)
        for i,row in enumerate(trace['metadata']):
            expected=np.log(row['probabilities'][row['action']])+np.log(row['candidate_probability'])
            self.assertAlmostEqual(row['old_logp'],expected,places=5)
            if i:
                self.assertFalse(trace['candidate_masks'][i,trace['candidate_choices'][:i]].any())
        self.assertEqual(len({r['proposal_graph'] for r in trace['metadata']}),8)

    def test_connectivity_features_distinguish_edit_locations(self):
        ortho=graph_features(Chem.MolFromSmiles('Oc1ccccc1N'))
        para=graph_features(Chem.MolFromSmiles('Oc1ccc(N)cc1'))
        np.testing.assert_allclose(ortho[:len(DESCRIPTOR_NAMES)],para[:len(DESCRIPTOR_NAMES)],atol=1e-6)
        self.assertFalse(np.array_equal(ortho,para))

    def test_feedback_learns_concrete_edits_and_exact_resume(self):
        with torch.no_grad():
            self.policy.actor[-1].bias.fill_(-30.)
            self.policy.actor[-1].bias[4]=30.
        _,trace=self.policy.propose(self.parent,[.1]*4,self.preferences,44)
        self.assertEqual(trace['actions'].tolist(),[4]*8)
        # Reward the oxygen-rich choices within one fixed family, so an
        # action-category-only actor cannot solve this task.
        values=trace['candidate_features'][trace['candidate_choices'],3]
        self.assertGreater(float(values.std()),.01)
        scores=[dict(objectives=[float(value)]*11,constraint_costs=[0.,0.]) for value in values]
        before=[p.detach().clone() for p in self.policy.candidate_actor.parameters()]
        result=self.policy.update(trace,scores,dict(update_epochs=2,clip_epsilon=.2,preco_strength=.1,
            preference_mix=.7,max_grad_norm=8.,group_relative_advantages=True),np.zeros(11),[0.,0.])
        self.assertLess(result['replay_error'],1e-5)
        self.assertTrue(all(torch.isfinite(p).all() for p in self.policy.actor_parameters()))
        self.assertTrue(any(not torch.equal(a,b) for a,b in zip(before,self.policy.candidate_actor.parameters())))
        _,logp,_=self.policy.policy_statistics(self.policy.trace_data(trace))
        self.assertGreater(float(((values-values.mean())*(logp-trace['old_logp'])).sum().detach()),0.)
        from utils.molecular_size_policy import new_size_policy
        exp=SimpleNamespace(protocol={'size_policy':CANDIDATE_STRUCTURE_POLICY},frontier_objectives=[None]*5,
            history_dim=4,reward_schema=[None]*11,constraints=[None]*2,device='cpu')
        restored=new_size_policy(exp,{'size_policy':copy.deepcopy(self.policy.state_dict())})
        a,ta=self.policy.propose(self.parent,[.1]*4,self.preferences,71)
        b,tb=restored.propose(self.parent,[.1]*4,self.preferences,71)
        self.assertEqual(a,b);self.assertEqual(ta['metadata'],tb['metadata'])

    def test_exhaustion_and_more_objectives_keep_finite_probabilities(self):
        from utils.molecular_size_policy import edit_catalog
        from utils.frontier_discovery import molecular_graph
        p=copy.deepcopy(self.parent);p['objectives']=[.4]*9
        policy=CandidatePPO(CANDIDATE_STRUCTURE_POLICY,9,4,19,2,'cpu')
        _,catalog=edit_catalog(p,CANDIDATE_STRUCTURE_POLICY)
        blocked={molecular_graph(Chem.MolToSmiles(m)) for entries in catalog for m,_ in entries}
        seeds,trace=policy.propose(p,[.1]*4,torch.full((2,19),1/19),3,blocked)
        self.assertEqual(len(seeds),2)
        self.assertTrue(all(r['novelty_exhausted'] for r in trace['metadata']))
        _,logp,entropy=policy.policy_statistics(policy.trace_data(trace))
        torch.testing.assert_close(logp,torch.zeros(2));torch.testing.assert_close(entropy,torch.zeros(2))


if __name__=='__main__':unittest.main()
