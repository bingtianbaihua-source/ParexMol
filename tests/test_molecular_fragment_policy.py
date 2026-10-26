import copy
import unittest
from types import SimpleNamespace

import numpy as np
import torch
from rdkit import Chem,RDLogger

from utils.molecular_fragment_search import fragment_library,fragment_catalog
from utils.molecular_fragment_policy import FragmentPPO
from utils.molecular_structure_search import FRAGMENT_STRUCTURE_POLICY,prepare_structure_edit
from utils.molecular_size_policy import new_size_policy
from test_molecular_size_policy import parent


def donor(smiles,generation=0,pocket='0'):
    row=parent(smiles)
    row['smiles']=Chem.MolToSmiles(Chem.MolFromMolBlock(row['mol_block']))
    return dict(row,pocket=pocket,source='public' if generation==0 else 'generated',
                generation=generation,source_ref=smiles)


class FragmentPolicyTests(unittest.TestCase):
    def setUp(self):
        RDLogger.DisableLog('rdApp.error');RDLogger.DisableLog('rdApp.warning')
        self.cfg=copy.deepcopy(FRAGMENT_STRUCTURE_POLICY)
        self.parent=donor('CC1CCC(O)CC1CCO')
        self.donors=[donor('COc1ccccc1CC'),donor('COc1ccc(N)cc1'),donor('CCOC1CCCCC1',2)]

    def tearDown(self):
        RDLogger.EnableLog('rdApp.error');RDLogger.EnableLog('rdApp.warning')

    def test_fitted_model_preserves_fragment_ppo_replay_and_resume(self):
        self.cfg['surrogate']['version']=2
        self.test_joint_ppo_and_restore_use_same_causal_fragment_choices()

    def test_library_is_deterministic_and_grafts_preserve_graph_and_pose(self):
        cfg=dict(self.cfg,max_donor_fragments=2)
        a=fragment_library(self.donors,cfg);b=fragment_library(list(reversed(self.donors)),cfg)
        self.assertEqual(len(a),2);self.assertEqual([r for _,r in a],[r for _,r in b])
        base,catalog=fragment_catalog(self.parent,self.cfg,self.donors)
        grafts=catalog[-1];self.assertGreater(len(grafts),8)
        self.assertTrue(all(4<=r['delta']<=8 for _,r in grafts))
        self.assertGreater(max(r['delta'] for _,r in grafts),3)
        candidate,edit=grafts[0];n=base.GetNumHeavyAtoms()
        self.assertEqual(candidate.GetNumHeavyAtoms(),n+edit['delta'])
        self.assertEqual(len(Chem.GetMolFrags(candidate)),1)
        self.assertTrue(all(not a.GetNumRadicalElectrons() for a in candidate.GetAtoms()))
        np.testing.assert_allclose(candidate.GetConformer().GetPositions()[:n],base.GetConformer().GetPositions())
        links=[b for b in candidate.GetBonds() if (b.GetBeginAtomIdx()<n)!=(b.GetEndAtomIdx()<n)]
        self.assertEqual(len(links),1)
        self.assertGreater(candidate.GetRingInfo().NumRings(),base.GetRingInfo().NumRings())
        prepared=prepare_structure_edit(candidate,copy.deepcopy(edit),np.random.default_rng(2))
        self.assertEqual(prepared.GetNumHeavyAtoms(),candidate.GetNumHeavyAtoms())
        x,y=Chem.Mol(candidate),Chem.Mol(prepared)
        Chem.RemoveStereochemistry(x);Chem.RemoveStereochemistry(y)
        self.assertEqual(Chem.MolToSmiles(x),Chem.MolToSmiles(y))
        self.assertTrue(np.isfinite(prepared.GetConformer().GetPositions()).all())

    def test_donor_support_and_atom_domain_are_explicit(self):
        _,catalog=fragment_catalog(self.parent,self.cfg,[])
        self.assertEqual(catalog[-1],[])
        wrong=[dict(self.donors[0],pocket='other')]
        with self.assertRaisesRegex(ValueError,'pocket'):fragment_catalog(self.parent,self.cfg,wrong)
        small=dict(self.cfg,max_atoms=Chem.MolFromMolBlock(self.parent['mol_block']).GetNumHeavyAtoms()+3)
        _,catalog=fragment_catalog(self.parent,small,self.donors)
        self.assertEqual(catalog[-1],[])

    def test_joint_ppo_and_restore_use_same_causal_fragment_choices(self):
        torch.manual_seed(17)
        cfg=copy.deepcopy(self.cfg);cfg['surrogate'].update(warmup_molecules=2,samples_power=7)
        policy=FragmentPPO(cfg,5,4,11,2,'cpu')
        public=[dict(valid=True,feasible=True,smiles=d['smiles'],frontier_objectives=[.4+i*.1]*5)
                for i,d in enumerate(self.donors[:2])]
        policy.initialize_surrogate({'0':public},[0.]*5,[1.]*5)
        policy.set_donors('0',self.donors)
        with torch.no_grad():
            policy.actor[-1].bias.fill_(-30.);policy.actor[-1].bias[-1]=30.
        prefs=torch.full((8,11),1/11)
        seeds,trace=policy.propose(self.parent,[.1]*4,prefs,31)
        self.assertEqual(trace['actions'].tolist(),[10]*8)
        self.assertEqual(len({r['proposal_graph'] for r in trace['metadata']}),8)
        self.assertTrue(all(r['edit']['donor_smiles'] in {d['smiles'] for d in self.donors} for r in trace['metadata']))
        _,logp,_=policy.policy_statistics(policy.trace_data(trace))
        torch.testing.assert_close(logp,trace['old_logp'])
        values=trace['candidate_features'][trace['candidate_choices'],3]
        scores=[dict(objectives=[float(v)]*11,constraint_costs=[0.,0.]) for v in values]
        stats=policy.update(trace,scores,dict(update_epochs=2,clip_epsilon=.2,preco_strength=.1,
            preference_mix=.7,max_grad_norm=8.,group_relative_advantages=True),np.zeros(11),[0.,0.])
        self.assertLess(stats['replay_error'],1e-5)
        self.assertTrue(all(torch.isfinite(p).all() for p in policy.actor_parameters()))
        exp=SimpleNamespace(protocol={'size_policy':cfg},frontier_objectives=[None]*5,history_dim=4,
            reward_schema=[None]*11,constraints=[None]*2,device='cpu')
        restored=new_size_policy(exp,{'size_policy':copy.deepcopy(policy.state_dict())})
        # The production harness restores this context from the saved frontier
        # pool before every proposal, independently of learned policy weights.
        restored.set_donors('0',self.donors)
        a,ta=policy.propose(self.parent,[.1]*4,prefs,32)
        b,tb=restored.propose(self.parent,[.1]*4,prefs,32)
        self.assertEqual(a,b);self.assertEqual(ta['metadata'],tb['metadata'])
        torch.testing.assert_close(ta['proposal_bias'],tb['proposal_bias'])


if __name__=='__main__':unittest.main()
