import copy
import json
import unittest

import numpy as np
import torch
from rdkit import Chem, RDLogger

from utils.frontier_discovery import DiscoveryArchive
from utils.frontier_rank import rank_feedback
from utils.molecular_objectives import ObjectiveSet
from utils.molecular_size_policy import SizePPO, edit_catalog
from utils.molecular_structure_search import DEFAULT_STRUCTURE_POLICY
from utils.morl import preco_direction
from test_molecular_size_policy import parent


def structure_parent():
    value=parent('CC1CCC(O)CC1CCO')
    value['smiles']=Chem.MolToSmiles(Chem.MolFromMolBlock(value['mol_block']))
    return value


class RankFeedbackTests(unittest.TestCase):
    def test_non_hv_contributors_still_get_objective_specific_feedback(self):
        for dimension in (2,5,9):
            schema=ObjectiveSet([dict(name=f'x{i}',evaluator='qed') for i in range(dimension)]).schema
            archive=DiscoveryArchive(schema,history_scope='public_plus_feasible_novel')
            def row(smi,first,second):
                return dict(valid=True,feasible=True,smiles=smi,frontier_objectives=[first,second]+[.5]*(dimension-2))
            archive.initialize(0,[row('CCCCCC',.8,.2),row('CCCCCN',.2,.8)])
            records=[row('CCCCCO',.5,.1)]*2+[row('CCCCCC',.8,.2),dict(valid=False)]
            before=copy.deepcopy(archive.state_dict())
            credits,meta=rank_feedback(archive,0,records)
            self.assertEqual(archive.state_dict(),before)
            np.testing.assert_allclose(credits[:2,:2],[[.25,0],[.25,0]])
            self.assertFalse(credits[2:].any())
            self.assertEqual(meta['nonzero_attempts'],2)
            hv,_=archive.credit(0,records)
            self.assertEqual(float(hv.sum()),0.)
            archive.update(0,records[:2])
            self.assertFalse(rank_feedback(archive,0,records)[0].any())

    def test_rank_channels_preferences_and_gradient_floor_are_dimension_agnostic(self):
        from scripts.run_v18 import MORLExperiment
        for dimension in (5,9):
            exp=object.__new__(MORLExperiment)
            exp.objectives=exp.frontier_objectives=ObjectiveSet([
                dict(name=f'x{i}',evaluator='qed') for i in range(dimension)])
            exp.protocol=dict(discovery=dict(reward_scale=.1,preference_weight=.5,
                ranking=dict(method='per_objective_midrank',preference_weight=.3)))
            preferences=exp.reward_condition(np.full(dimension,1/dimension))
            self.assertEqual(len(exp.reward_schema),2*dimension+1)
            self.assertAlmostEqual(preferences.sum(),1.)
            self.assertAlmostEqual(preferences[dimension],.5)
            torch.manual_seed(7)
            rewards=torch.rand(8,len(preferences));p=torch.tensor(preferences,dtype=torch.float32)
            _,info=preco_direction(torch.zeros(8),rewards,rewards,p,torch.zeros_like(p),preference_mix=.2)
            self.assertTrue(torch.all(info['weights']>=.2*p-1e-7))


class StructureSearchTests(unittest.TestCase):
    def setUp(self):
        RDLogger.DisableLog('rdApp.warning');RDLogger.DisableLog('rdApp.error')

    def tearDown(self):
        RDLogger.EnableLog('rdApp.warning');RDLogger.EnableLog('rdApp.error')

    def test_cleaned_square_planar_sulfur_tag_can_restart_structure_search(self):
        # A scored sulfur identity with a square-planar tag. RDKit
        # removes this SP tag when parsing the same sulfur connectivity.
        scored=r'O=[S@SP1H](O)(O)c1ccc(/C=C\c2ccccc2)cc1'
        source=parent(scored)
        actual=Chem.MolToSmiles(Chem.MolFromMolBlock(source['mol_block']))
        self.assertNotEqual(actual,scored)
        self.assertEqual(actual,Chem.MolToSmiles(Chem.MolFromSmiles(scored)))
        source['smiles']=scored
        before=copy.deepcopy(source)
        mol,catalog=edit_catalog(source,DEFAULT_STRUCTURE_POLICY)
        self.assertEqual(Chem.MolToSmiles(mol),actual)
        self.assertTrue(any(catalog))
        self.assertEqual(source,before)

    def test_identity_fallback_still_rejects_changed_graph_or_stereochemistry(self):
        for original,changed in [('CCCCCCO','CCCCCCN'),
                                 ('CCC[C@H](O)CC','CCC[C@@H](O)CC'),
                                 (r'CCC/C=C/CC',r'CCC/C=C\CC')]:
            with self.subTest(original=original):
                source=parent(original);source['smiles']=changed
                with self.assertRaisesRegex(ValueError,'scored parent identity'):
                    edit_catalog(source,DEFAULT_STRUCTURE_POLICY)

    def test_actions_include_topology_heteroatoms_and_multiatom_size_changes(self):
        source=structure_parent()
        mol,catalog=edit_catalog(source,DEFAULT_STRUCTURE_POLICY)
        self.assertTrue(all(catalog),[len(c) for c in catalog])
        deltas=set()
        for action,entries in zip(DEFAULT_STRUCTURE_POLICY['actions'],catalog):
            for edited,info in entries:
                Chem.SanitizeMol(edited)
                self.assertEqual(len(Chem.GetMolFrags(edited)),1)
                self.assertFalse(any(a.GetNumRadicalElectrons() for a in edited.GetAtoms()))
                self.assertEqual(edited.GetNumHeavyAtoms(),mol.GetNumHeavyAtoms()+info['delta'])
                deltas.add(info['delta'])
        self.assertTrue({-3,-2,-1,0,1,2,3} <= deltas)
        close=catalog[DEFAULT_STRUCTURE_POLICY['actions'].index('close_ring')][0][0]
        opened=catalog[DEFAULT_STRUCTURE_POLICY['actions'].index('open_ring')][0][0]
        self.assertGreater(close.GetRingInfo().NumRings(),mol.GetRingInfo().NumRings())
        self.assertLess(opened.GetRingInfo().NumRings(),mol.GetRingInfo().NumRings())

    def test_structure_ppo_executes_every_action_and_resumes_exactly(self):
        source=structure_parent()
        cfg=copy.deepcopy(DEFAULT_STRUCTURE_POLICY)
        policy=SizePPO(cfg,5,4,11,2,'cpu')
        preferences=torch.full((2,11),1/11)
        for action in range(9):
            with torch.no_grad():
                policy.actor[-1].bias.fill_(-100)
                policy.actor[-1].bias[action]=0
            seeds,trace=policy.propose(source,[.1]*4,preferences,31+action)
            self.assertEqual(trace['actions'].tolist(),[action,action])
            for seed,meta in zip(seeds,trace['metadata']):
                mol=Chem.MolFromMolBlock(seed['mol_block'])
                self.assertEqual(mol.GetNumAtoms(),meta['seed_atoms'])
                self.assertEqual(Chem.MolToSmiles(mol),seed['smiles'])
                self.assertTrue(np.isfinite(mol.GetConformer().GetPositions()).all())
        with torch.no_grad():policy.actor[-1].bias.zero_()
        _,trace=policy.propose(source,[.1]*4,torch.full((16,11),1/11),41)
        scores=[dict(objectives=[float(a==3)]*11,constraint_costs=[0.,0.]) for a in trace['actions']]
        stats=policy.update(trace,scores,dict(update_epochs=2,clip_epsilon=.2,preco_strength=.1,
            preference_mix=.2,max_grad_norm=8.),np.zeros(11),[1.,1.])
        self.assertEqual(len(stats['action_counts']),9)
        restored=SizePPO(cfg,5,4,11,2,'cpu');restored.load_state_dict(copy.deepcopy(policy.state_dict()))
        first,t1=policy.propose(source,[.1]*4,preferences,77)
        second,t2=restored.propose(source,[.1]*4,preferences,77)
        self.assertEqual(first,second);self.assertEqual(t1['metadata'],t2['metadata'])


if __name__=='__main__':unittest.main()
