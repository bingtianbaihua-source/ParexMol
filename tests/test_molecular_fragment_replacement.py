import copy
import unittest
from types import SimpleNamespace

import numpy as np
import torch
from rdkit import Chem,RDLogger

from utils.molecular_fragment_policy import FragmentPPO
from utils.molecular_fragment_search import fragment_catalog
from utils.molecular_size_policy import new_size_policy
from utils.molecular_structure_search import REPLACEMENT_STRUCTURE_POLICY,prepare_structure_edit
from test_molecular_fragment_policy import donor


class FragmentReplacementTests(unittest.TestCase):
    def setUp(self):
        RDLogger.DisableLog('rdApp.error');RDLogger.DisableLog('rdApp.warning')
        self.cfg=copy.deepcopy(REPLACEMENT_STRUCTURE_POLICY)
        self.parent=donor('CCOc1ccc(CCNC(=O)C2CCCCC2)cc1')
        self.donors=[donor('COc1ccccc1CC'),donor('COc1ccc(N)cc1'),donor('CCOC1CCCCC1',2)]

    def tearDown(self):
        RDLogger.EnableLog('rdApp.error');RDLogger.EnableLog('rdApp.warning')

    def test_replacements_shrink_keep_or_grow_size_and_preserve_the_retained_core(self):
        base,catalog=fragment_catalog(self.parent,self.cfg,self.donors)
        replacements=catalog[-1];self.assertGreater(len(replacements),8)
        self.assertEqual({int(np.sign(info['delta'])) for _,info in replacements},{-1,0,1})
        representatives={}
        for candidate,edit in replacements:
            keep=edit['retained_atoms'];removed=edit['removed_atoms'];n=candidate.GetNumHeavyAtoms()
            self.assertEqual(set(keep)|set(removed),set(range(base.GetNumAtoms())))
            self.assertFalse(set(keep)&set(removed))
            self.assertEqual(n,base.GetNumHeavyAtoms()+edit['delta'])
            self.assertTrue(self.cfg['min_atoms']<=n<=self.cfg['max_atoms'])
            self.assertEqual(len(Chem.GetMolFrags(candidate)),1)
            self.assertFalse(any(a.GetNumRadicalElectrons() for a in candidate.GetAtoms()))
            np.testing.assert_allclose(candidate.GetConformer().GetPositions()[:len(keep)],
                                       base.GetConformer().GetPositions()[keep])
            for i,old in enumerate(keep):
                self.assertEqual(candidate.GetAtomWithIdx(i).GetAtomicNum(),base.GetAtomWithIdx(old).GetAtomicNum())
                for j in range(i):
                    a=candidate.GetBondBetweenAtoms(i,j);b=base.GetBondBetweenAtoms(old,keep[j])
                    self.assertEqual(a.GetBondType() if a else None,b.GetBondType() if b else None)
            links=[b for b in candidate.GetBonds() if (b.GetBeginAtomIdx()<len(keep))!=(b.GetEndAtomIdx()<len(keep))]
            self.assertEqual(len(links),1)
            anchor,departing=edit['parent_cut']
            self.assertIn(anchor,keep);self.assertIn(departing,removed)
            self.assertEqual(edit['anchor'],keep.index(anchor))
            original_bond=base.GetBondBetweenAtoms(anchor,departing)
            self.assertFalse(original_bond.IsInRing());self.assertEqual(original_bond.GetBondType(),Chem.BondType.SINGLE)
            representatives.setdefault(int(np.sign(edit['delta'])),(candidate,edit))
        for candidate,edit in representatives.values():
            prepared=prepare_structure_edit(candidate,copy.deepcopy(edit),np.random.default_rng(8))
            self.assertTrue(np.isfinite(prepared.GetConformer().GetPositions()).all())
            x,y=Chem.Mol(candidate),Chem.Mol(prepared)
            Chem.RemoveStereochemistry(x);Chem.RemoveStereochemistry(y)
            self.assertEqual(Chem.MolToSmiles(x),Chem.MolToSmiles(y))

    def test_replacement_support_respects_donors_domain_and_deterministic_provenance(self):
        _,empty=fragment_catalog(self.parent,self.cfg,[])
        self.assertEqual(empty[-1],[])
        with self.assertRaisesRegex(ValueError,'pocket'):
            fragment_catalog(self.parent,self.cfg,[dict(self.donors[0],pocket='other')])
        cfg=dict(self.cfg,max_atoms=Chem.MolFromMolBlock(self.parent['mol_block']).GetNumHeavyAtoms())
        _,a=fragment_catalog(self.parent,cfg,self.donors)
        _,b=fragment_catalog(self.parent,cfg,list(reversed(self.donors)))
        self.assertTrue(a[-1]);self.assertTrue(all(info['delta']<=0 for _,info in a[-1]))
        self.assertEqual([(Chem.MolToSmiles(m),info) for m,info in a[-1]],
                         [(Chem.MolToSmiles(m),info) for m,info in b[-1]])

    def test_joint_ppo_replay_update_and_resume_with_fitted_vector_surrogate(self):
        for dimension in [5,9]:
            torch.manual_seed(29)
            cfg=copy.deepcopy(self.cfg)
            cfg['surrogate'].update(version=2,warmup_molecules=2,samples_power=7)
            cfg['ring_closure_max_aromatic_overlap']=2
            policy=FragmentPPO(cfg,dimension,4,2*dimension+1,2,'cpu')
            parent=dict(self.parent,objectives=[.4]*dimension)
            public=[dict(valid=True,feasible=True,smiles=d['smiles'],frontier_objectives=[.3+.1*i]*dimension)
                    for i,d in enumerate(self.donors[:2])]
            policy.initialize_surrogate({'0':public},[0.]*dimension,[1.]*dimension)
            policy.set_donors('0',self.donors)
            with torch.no_grad():
                policy.actor[-1].bias.fill_(-30.);policy.actor[-1].bias[-1]=30.
            prefs=torch.full((4,2*dimension+1),1/(2*dimension+1))
            seeds,trace=policy.propose(parent,[.1]*4,prefs,73)
            self.assertEqual(trace['actions'].tolist(),[11]*4)
            self.assertEqual(len({r['proposal_graph'] for r in trace['metadata']}),4)
            self.assertTrue(all(r['edit']['donor_smiles'] in {d['smiles'] for d in self.donors} for r in trace['metadata']))
            policy.observe_surrogate(0,[dict(valid=True,feasible=True,smiles=seeds[0]['smiles'],frontier_objectives=[.7]*dimension)])
            _,logp,_=policy.policy_statistics(policy.trace_data(trace))
            torch.testing.assert_close(logp,trace['old_logp'])
            values=trace['candidate_features'][trace['candidate_choices'],3]
            scores=[dict(objectives=[float(v)]*(2*dimension+1),constraint_costs=[0.,0.]) for v in values]
            before=policy.candidate_actor[-1].weight.detach().clone()
            stats=policy.update(trace,scores,dict(update_epochs=2,clip_epsilon=.2,preco_strength=.1,
                preference_mix=.7,max_grad_norm=8.,group_relative_advantages=True),np.zeros(2*dimension+1),[0.,0.])
            self.assertLess(stats['replay_error'],1e-5)
            self.assertFalse(torch.equal(before,policy.candidate_actor[-1].weight))
            exp=SimpleNamespace(protocol={'size_policy':cfg},frontier_objectives=[None]*dimension,
                history_dim=4,reward_schema=[None]*(2*dimension+1),constraints=[None]*2,device='cpu')
            restored=new_size_policy(exp,{'size_policy':copy.deepcopy(policy.state_dict())})
            restored.set_donors('0',self.donors)
            a,ta=policy.propose(parent,[.1]*4,prefs,74)
            b,tb=restored.propose(parent,[.1]*4,prefs,74)
            self.assertEqual(a,b);self.assertEqual(ta['metadata'],tb['metadata'])
            torch.testing.assert_close(ta['proposal_bias'],tb['proposal_bias'])


if __name__=='__main__':unittest.main()
