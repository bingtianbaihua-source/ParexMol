"""Learned size actions must change graph sizes and receive actual PPO credit."""
import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from rdkit import Chem
from torch_geometric.transforms import Compose

import test_frontier_seeds as fixtures

molecule = fixtures.molecule
from utils import transforms
from utils.molecular_size_policy import DEFAULT_SIZE_POLICY, SizePPO, edit_catalog
from utils.local_diffusion import sample_lead


def parent(smiles='CCCCCCC', dimension=5):
    mol = molecule(smiles)
    return dict(mol_block=Chem.MolToMolBlock(mol), smiles=Chem.MolToSmiles(mol),
        objectives=[.5]*dimension, pocket='0', source='public', source_ref='fixture', generation=0)


class MolecularSizeTests(unittest.TestCase):
    def test_different_frontier_parents_keep_their_own_graph_sizes_and_provenance(self):
        fixture=fixtures.FrontierSeedTests();fixture.setUp()
        parents=[parent('CCCCCCC'),parent('c1ccncc1CCO')]
        sizes=[Chem.MolFromMolBlock(p['mol_block']).GetNumAtoms() for p in parents]
        metadata=[dict(seed_atoms=n,delta=0) for n in sizes]
        transform=Compose([transforms.ComputeLigandAtomNoiseDist('ref_prior'),transforms.FeaturizeProteinAtom(),
            transforms.FeaturizeLigandAtom('basic'),transforms.AddDecompIndicator(add_ord_feat=False),
            transforms.FeaturizeLigandBond(mode='fc',set_bond_type=True)])
        kernel=SimpleNamespace(q_v_sample=lambda onehot,t,b:(onehot.argmax(-1),onehot))
        model=SimpleNamespace(alphas_cumprod=torch.tensor([1.,1.]),num_classes=8,num_bond_classes=5,
            atom_type_trans=kernel,bond_type_trans=kernel)
        exp=SimpleNamespace(original={0:fixture.original()},device='cpu',train_transform=transform,
            sampling=SimpleNamespace(sample=SimpleNamespace(energy_drift=None)),
            cfg=SimpleNamespace(data=SimpleNamespace(transform=SimpleNamespace(ligand_atom_mode='basic'))))
        seen=[]
        def rollout(policy,**context):
            seen.append(context)
            return dict(pos=context['init_ligand_pos'],v=context['init_ligand_v'],bond=context['init_ligand_fc_bond_type'])
        protein=SimpleNamespace(to_dict_atom=lambda:dict(pos=np.zeros((12,3))))
        target=dict(target_id=0,receptor='test.pdb',source_ligand='reference.sdf')
        with patch('utils.local_diffusion.PDBProtein',return_value=protein), \
             patch('utils.local_diffusion.rollout',side_effect=rollout), \
             patch('utils.local_diffusion.reconstruct_molecule',return_value=(None,None,None)):
            rows,_=sample_lead(exp,model,target,2,19,0,None,1.,1,frontier_parent=parents[0],
                              frontier_parents=parents,resized_parents=parents,size_metadata=metadata)
        self.assertEqual(torch.bincount(seen[0]['batch_ligand']).tolist(),sizes)
        self.assertEqual([r['parent_smiles'] for r in rows],[p['smiles'] for p in parents])
        self.assertEqual([r['lead_source'] for r in rows],['frontier:'+p['smiles'] for p in parents])
        self.assertEqual([len(r['decomp_mask']) for r in rows],sizes)
        self.assertEqual(len(rows),2)
        with self.assertRaisesRegex(ValueError,'same-pocket'):
            sample_lead(exp,model,target,2,19,0,None,1.,1,frontier_parents=[parents[0],dict(parents[1],pocket='1')])

    def test_legal_masks_and_exact_growth_shrinkage_include_closed_rings(self):
        cfg = copy.deepcopy(DEFAULT_SIZE_POLICY)
        for smiles, expected in [('CCCCCC',[False,True,True]), ('c1ccccc1',[False,True,True]),
                                  ('C1CCCCCC1',[True,True,True])]:
            _, catalog = edit_catalog(parent(smiles), cfg)
            self.assertEqual([bool(c) for c in catalog],expected)
        cfg['max_atoms'] = 7
        _, catalog = edit_catalog(parent(), cfg)
        self.assertEqual([bool(c) for c in catalog],[True,True,False])
        policy = SizePPO(DEFAULT_SIZE_POLICY,5,4,6,2,'cpu')
        seeds, trace = policy.propose(parent(), [.1]*4, torch.full((24,6),1/6),19)
        self.assertEqual(set(trace['actions'].tolist()),{0,1,2})
        for seed, meta in zip(seeds,trace['metadata']):
            mol = Chem.MolFromMolBlock(seed['mol_block'])
            self.assertEqual(mol.GetNumAtoms(),7+meta['delta'])
            self.assertEqual(Chem.MolToSmiles(mol),seed['smiles'])
            self.assertTrue(np.isfinite(mol.GetConformer().GetPositions()).all())

    def test_reward_and_lagrangian_cost_change_size_probabilities_and_resume_exactly(self):
        cfg = dict(DEFAULT_SIZE_POLICY, entropy_coefficient=0.)
        training = dict(update_epochs=2, clip_epsilon=.2, preco_strength=.1, max_grad_norm=1.)
        for dim in (5,8):
            for penalize in (False,True):
                policy = SizePPO(cfg,dim,4,dim+1,2,'cpu')
                for p in policy.critic.parameters():
                    torch.nn.init.zeros_(p)
                prefs = torch.full((30,dim+1),1/(dim+1))
                _, trace = policy.propose(parent(dimension=dim),[.1]*4,prefs,22)
                scores = [dict(objectives=([0.] if penalize else [float(a==2)])*(dim+1),
                    constraint_costs=[float(a==2) if penalize else 0.,0.]) for a in trace['actions']]
                stats = policy.update(trace,scores,training,np.zeros(dim+1),[2.,1.])
                change = stats['mean_probabilities_after'][2]-stats['mean_probabilities_before'][2]
                self.assertGreater(-change if penalize else change,0.)
                self.assertLess(stats['replay_error'],1e-6)
                restored = SizePPO(cfg,dim,4,dim+1,2,'cpu')
                restored.load_state_dict(copy.deepcopy(policy.state_dict()))
                p1,t1 = policy.propose(parent(dimension=dim),[.2]*4,prefs,23)
                p2,t2 = restored.propose(parent(dimension=dim),[.2]*4,prefs,23)
                self.assertEqual(p1,p2)
                self.assertEqual(t1['metadata'],t2['metadata'])
                policy.update(t1,scores,training,np.zeros(dim+1),[2.,1.])
                restored.update(t2,scores,training,np.zeros(dim+1),[2.,1.])
                for p,q in zip(policy.actor.parameters(),restored.actor.parameters()):
                    torch.testing.assert_close(p,q,atol=0,rtol=0)

    def test_variable_size_seeds_form_one_diffusion_batch_and_keep_failed_attempts(self):
        fixture=fixtures.FrontierSeedTests();fixture.setUp()
        p=parent();controller=SizePPO(DEFAULT_SIZE_POLICY,5,4,6,2,'cpu')
        seeds,trace=controller.propose(p,[.1]*4,torch.full((8,6),1/6),19)
        transform=Compose([transforms.ComputeLigandAtomNoiseDist('ref_prior'),transforms.FeaturizeProteinAtom(),
            transforms.FeaturizeLigandAtom('basic'),transforms.AddDecompIndicator(add_ord_feat=False),
            transforms.FeaturizeLigandBond(mode='fc',set_bond_type=True)])
        kernel=SimpleNamespace(q_v_sample=lambda onehot,t,b:(onehot.argmax(-1),onehot))
        model=SimpleNamespace(alphas_cumprod=torch.tensor([1.,1.]),num_classes=8,num_bond_classes=5,
            atom_type_trans=kernel,bond_type_trans=kernel)
        exp=SimpleNamespace(original={0:fixture.original()},device='cpu',train_transform=transform,
            sampling=SimpleNamespace(sample=SimpleNamespace(energy_drift=None)),
            cfg=SimpleNamespace(data=SimpleNamespace(transform=SimpleNamespace(ligand_atom_mode='basic'))))
        seen=[]
        def rollout(policy,**context):
            seen.append(context)
            return dict(pos=context['init_ligand_pos'],v=context['init_ligand_v'],bond=context['init_ligand_fc_bond_type'])
        protein=SimpleNamespace(to_dict_atom=lambda:dict(pos=np.zeros((12,3))))
        with patch('utils.local_diffusion.PDBProtein',return_value=protein), \
             patch('utils.local_diffusion.rollout',side_effect=rollout), \
             patch('utils.local_diffusion.reconstruct_molecule',return_value=(None,None,None)):
            rows,_=sample_lead(exp,model,dict(target_id=0,receptor='test.pdb',source_ligand='reference.sdf'),
                8,19,0,None,1.,1,frontier_parent=p,resized_parents=seeds,size_metadata=trace['metadata'])
        expected=[m['seed_atoms'] for m in trace['metadata']]
        self.assertEqual(torch.bincount(seen[0]['batch_ligand']).tolist(),expected)
        self.assertEqual([len(r['decomp_mask']) for r in rows],expected)
        self.assertEqual(len(rows),8)
        self.assertTrue(all(r['mol'] is None and r['parent_smiles']==p['smiles'] for r in rows))
        self.assertEqual([r['size_action'] for r in rows],trace['metadata'])


if __name__=='__main__':unittest.main()
