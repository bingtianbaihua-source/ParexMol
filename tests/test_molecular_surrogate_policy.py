import copy
import unittest
from types import SimpleNamespace

import numpy as np
import torch
from rdkit import Chem,RDLogger

from utils.frontier_surrogate import SurrogateHistory,features,kernel,posterior
from utils.molecular_surrogate_policy import SurrogatePPO
from utils.molecular_structure_search import SURROGATE_STRUCTURE_POLICY
from test_molecular_size_policy import parent


def record(smiles,values,feasible=True):
    return dict(valid=True,smiles=smiles,frontier_objectives=values,feasible=feasible)


class SurrogateTests(unittest.TestCase):
    def setUp(self):
        RDLogger.DisableLog('rdApp.error');RDLogger.DisableLog('rdApp.warning')
        torch.manual_seed(23)
        self.cfg=copy.deepcopy(SURROGATE_STRUCTURE_POLICY)
        self.cfg['surrogate'].update(warmup_molecules=2,samples_power=7)

    def tearDown(self):
        RDLogger.EnableLog('rdApp.error');RDLogger.EnableLog('rdApp.warning')

    def test_fitted_model_replays_after_later_observations_and_restores_exactly(self):
        self.cfg['surrogate']['version']=2
        self.test_guided_joint_replay_update_and_checkpoint_restore()

    def test_gp_predicts_observations_and_distinguishes_connectivity(self):
        x=np.stack([features(s) for s in ['Oc1ccccc1N','Oc1ccc(N)cc1','CCCCCO']])
        covariance=kernel(x,x)
        np.testing.assert_allclose(np.diag(covariance),1.)
        self.assertGreater(np.linalg.eigvalsh(covariance).min(),0.)
        y=np.array([[.3,.8],[.8,.2],[.1,.6]])
        mean,std=posterior(x,y,x)
        np.testing.assert_allclose(mean,y,atol=.01)
        self.assertTrue(np.all(std>0));self.assertLess(std.max(),.05)

    def test_history_is_pocket_local_causal_and_uses_physical_coordinates(self):
        h=SurrogateHistory(self.cfg['surrogate'],[0.,0.],[1.,1.])
        public=[record('CCCCCC',[.3,.4])]
        h.initialize('a',public,['CCCCCO']);h.initialize('b',public)
        before=h.context('a',['CCCCCC','CCCCCO'])
        self.assertFalse(before['meta']['active'])
        np.testing.assert_array_equal(h.bias(before,[True,True]),[0,0])
        rows=[dict(record('CCCCCN',[.7,.8]),objectives=[99.]*9),record('CCCCCO',[.9,.9]),
              record('CCCCCF',[1.,1.],False),dict(valid=False)]
        h.observe('a',rows)
        self.assertEqual(h.pockets['a']['front'],[[.7,.8]])
        self.assertEqual(h.pockets['a']['training']['CCCCCN'],[.7,.8])
        self.assertEqual(h.pockets['a']['observed_attempts'],4)
        self.assertEqual(len(h.pockets['b']['training']),1)
        self.assertFalse(before['meta']['active'])
        after=h.context('a',['CCCCCC','CCCCCN'])
        self.assertTrue(after['meta']['active'])
        bias=h.bias(after,[True,True]);self.assertGreater(bias[1],bias[0])
        snapshot=copy.deepcopy(after['remaining'])
        h.advance(after,1);self.assertTrue(np.all(after['remaining']<=snapshot))
        restored=SurrogateHistory.from_state_dict(h.state_dict())
        self.assertEqual(restored.state_dict(),h.state_dict())

    def test_guided_joint_replay_update_and_checkpoint_restore(self):
        p=parent('CC1CCC(O)CC1CCO');p['pocket']='0'
        p['smiles']=Chem.MolToSmiles(Chem.MolFromMolBlock(p['mol_block']))
        public=[record(p['smiles'],[.35]*5),record('CCCCCC',[.8,.6,.7,.8,.8])]
        policy=SurrogatePPO(self.cfg,5,4,11,2,'cpu')
        policy.initialize_surrogate({'0':public},[0.]*5,[1.]*5)
        prefs=torch.full((8,11),1/11)
        seeds,trace=policy.propose(p,[.1]*4,prefs,73)
        self.assertTrue(trace['surrogate']['active'])
        self.assertGreater(float(trace['proposal_bias'].std()),.01)
        data=policy.trace_data(trace);family,joint=policy.joint_distribution(data)
        torch.testing.assert_close(joint.log_prob(trace['candidate_choices']),trace['old_logp'])
        for action in range(policy.action_count):
            torch.testing.assert_close(joint.probs[:,trace['candidate_families']==action].sum(-1),family.probs[:,action])
        for row in trace['metadata']:
            self.assertAlmostEqual(row['old_logp'],np.log(row['probabilities'][row['action']])+
                                   np.log(row['candidate_probability']),places=5)
        frozen=trace['proposal_bias'].clone()
        # Later observations must not change the policy likelihood replay.
        policy.observe_surrogate(0,[record(seeds[0]['smiles'],[.9]*5)])
        _,logp,_=policy.policy_statistics(policy.trace_data(trace))
        torch.testing.assert_close(logp,trace['old_logp'])
        values=trace['candidate_features'][trace['candidate_choices'],3]
        scores=[dict(objectives=[float(v)]*11,constraint_costs=[0.,0.]) for v in values]
        result=policy.update(trace,scores,dict(update_epochs=2,clip_epsilon=.2,preco_strength=.1,
            preference_mix=.7,max_grad_norm=8.,group_relative_advantages=True),np.zeros(11),[0.,0.])
        self.assertLess(result['replay_error'],1e-5)
        torch.testing.assert_close(trace['proposal_bias'],frozen)
        from utils.molecular_size_policy import new_size_policy
        exp=SimpleNamespace(protocol={'size_policy':self.cfg},frontier_objectives=[None]*5,
            history_dim=4,reward_schema=[None]*11,constraints=[None]*2,device='cpu')
        restored=new_size_policy(exp,{'size_policy':copy.deepcopy(policy.state_dict())})
        a,ta=policy.propose(p,[.1]*4,prefs,74)
        b,tb=restored.propose(p,[.1]*4,prefs,74)
        self.assertEqual(a,b);self.assertEqual(ta['metadata'],tb['metadata'])
        torch.testing.assert_close(ta['proposal_bias'],tb['proposal_bias'])
        self.assertEqual(ta['surrogate'],tb['surrogate'])

    def test_warmup_exhaustion_and_arbitrary_dimension(self):
        from utils.molecular_size_policy import edit_catalog
        from utils.frontier_discovery import molecular_graph
        p=parent('CCCCCCO');p['pocket']='0';p['objectives']=[.4]*9
        p['smiles']=Chem.MolToSmiles(Chem.MolFromMolBlock(p['mol_block']))
        policy=SurrogatePPO(self.cfg,9,4,19,2,'cpu')
        policy.initialize_surrogate({'0':[record(p['smiles'],[.4]*9)]},[0.]*9,[1.]*9)
        _,catalog=edit_catalog(p,self.cfg)
        blocked={molecular_graph(Chem.MolToSmiles(m)) for entries in catalog for m,_ in entries}
        _,trace=policy.propose(p,[.1]*4,torch.full((2,19),1/19),73,blocked)
        self.assertFalse(trace['surrogate']['active']);self.assertTrue(all(x['novelty_exhausted'] for x in trace['metadata']))
        _,logp,entropy=policy.policy_statistics(policy.trace_data(trace))
        torch.testing.assert_close(logp,torch.zeros(2));torch.testing.assert_close(entropy,torch.zeros(2))
        policy.observe_surrogate(0,[record('CCCCCCN',[.6]*9)])
        context=policy.surrogate.context(0,['CCCCCCO','CCCCCCN'])
        self.assertTrue(context['meta']['active']);self.assertEqual(context['coverage'].shape[0],2)


if __name__=='__main__':unittest.main()
