import copy
import unittest

import numpy as np
import torch
from rdkit import Chem,RDLogger

from test_molecular_size_policy import parent
from utils.molecular_size_policy import edit_catalog
from utils.molecular_structure_search import SURROGATE_STRUCTURE_POLICY,validate_structure_config
from utils.molecular_surrogate_policy import SurrogatePPO


class RingClosureTopologyTests(unittest.TestCase):
    def setUp(self):
        RDLogger.DisableLog('rdApp.error');RDLogger.DisableLog('rdApp.warning')
        self.cfg=copy.deepcopy(SURROGATE_STRUCTURE_POLICY)
        self.cfg['ring_closure_max_aromatic_overlap']=2

    def tearDown(self):
        RDLogger.EnableLog('rdApp.error');RDLogger.EnableLog('rdApp.warning')

    def test_short_aromatic_bridges_are_removed_but_fused_annulations_remain(self):
        source=parent('CCCc1ccccc1')
        mol,old=edit_catalog(source,SURROGATE_STRUCTURE_POLICY)
        _,new=edit_catalog(source,self.cfg)
        index=self.cfg['actions'].index('close_ring')
        rings=[set(r) for r in mol.GetRingInfo().AtomRings()
               if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in r)]
        def overlap(edit):
            path=set(Chem.GetShortestPath(mol,*edit['bond']))
            return max([len(path&r) for r in rings],default=0)
        self.assertTrue(any(overlap(info)>2 for _,info in old[index]))
        self.assertTrue(any(overlap(info)==2 for _,info in new[index]))
        expected=[(Chem.MolToSmiles(m),info) for m,info in old[index] if overlap(info)<=2]
        self.assertEqual([(Chem.MolToSmiles(m),info) for m,info in new[index]],expected)
        for i in range(len(old)):
            if i!=index:
                self.assertEqual([(Chem.MolToSmiles(m),info) for m,info in old[i]],
                                 [(Chem.MolToSmiles(m),info) for m,info in new[i]])

    def test_aliphatic_ring_support_and_catalog_order_are_preserved(self):
        for smiles in ['CCCCCC','CC1CCC(O)CC1CCO']:
            with self.subTest(smiles=smiles):
                source=parent(smiles)
                source['smiles']=Chem.MolToSmiles(Chem.MolFromMolBlock(source['mol_block']))
                _,old=edit_catalog(source,SURROGATE_STRUCTURE_POLICY)
                _,new=edit_catalog(source,self.cfg)
                serialize=lambda catalog:[[(Chem.MolToSmiles(m),info) for m,info in entries] for entries in catalog]
                self.assertEqual(serialize(old),serialize(new))

    def test_topology_setting_is_explicit_and_invalid_values_are_rejected(self):
        validate_structure_config(self.cfg)
        validate_structure_config(SURROGATE_STRUCTURE_POLICY)
        for value in [True,1,8,2.0,None]:
            with self.subTest(value=value),self.assertRaises(ValueError):
                validate_structure_config(dict(self.cfg,ring_closure_max_aromatic_overlap=value))
        with self.assertRaises(ValueError):
            validate_structure_config(dict(self.cfg,unknown_rule=True))

    def test_filtered_joint_ppo_likelihood_update_and_restore(self):
        cfg=copy.deepcopy(self.cfg);cfg['surrogate'].update(version=2,warmup_molecules=2,samples_power=7)
        torch.manual_seed(31)
        source=parent('CCCc1ccccc1')
        policy=SurrogatePPO(cfg,5,4,11,2,'cpu')
        public=[dict(valid=True,feasible=True,smiles=source['smiles'],frontier_objectives=[.4]*5),
                dict(valid=True,feasible=True,smiles='CCCCCC',frontier_objectives=[.7,.6,.5,.8,.4])]
        policy.initialize_surrogate({'0':public},[0.]*5,[1.]*5)
        prefs=torch.full((8,11),1/11)
        _,trace=policy.propose(source,[.1]*4,prefs,51)
        self.assertTrue(trace['surrogate']['active'])
        _,logp,_=policy.policy_statistics(policy.trace_data(trace))
        torch.testing.assert_close(logp,trace['old_logp'])
        scores=[dict(objectives=[i/8]*11,constraint_costs=[0.,0.]) for i in range(8)]
        stats=policy.update(trace,scores,dict(update_epochs=2,clip_epsilon=.2,preco_strength=.1,
            preference_mix=.7,max_grad_norm=8.,group_relative_advantages=True),np.zeros(11),[0.,0.])
        self.assertLess(stats['replay_error'],1e-5)
        restored=SurrogatePPO(cfg,5,4,11,2,'cpu');restored.load_state_dict(copy.deepcopy(policy.state_dict()))
        first,a=policy.propose(source,[.1]*4,prefs,52)
        second,b=restored.propose(source,[.1]*4,prefs,52)
        self.assertEqual(first,second);self.assertEqual(a['metadata'],b['metadata'])
        torch.testing.assert_close(a['proposal_bias'],b['proposal_bias'])


if __name__=='__main__':unittest.main()
