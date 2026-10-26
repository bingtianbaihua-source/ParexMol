import copy
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.transforms import Compose

from utils.local_diffusion import frontier_parent_data, sample_lead
from utils.frontier_seeds import FrontierSeedPool, parent_metadata
from utils.molecular_objectives import ObjectiveSet
from utils import transforms


def molecule(smiles):
    mol=Chem.AddHs(Chem.MolFromSmiles(smiles))
    if AllChem.EmbedMolecule(mol,randomSeed=31)<0:raise ValueError('Fixture embedding failed')
    return Chem.RemoveHs(mol)


def scored(mol, value, **kwargs):
    return dict(valid=True,feasible=True,smiles=Chem.MolToSmiles(mol),frontier_objectives=[value]*5,**kwargs)


class FrontierSeedTests(unittest.TestCase):
    def setUp(self):
        self.schema=ObjectiveSet([dict(name=str(i),evaluator='qed') for i in range(5)]).schema
        self.public=molecule('CCCCCC');self.new=molecule('Oc1ccccc1N')

    def test_generated_frontier_becomes_parent_and_checkpoint_is_causal(self):
        pool=FrontierSeedPool(self.schema)
        pool.update(0,[scored(self.public,.4)],[self.public])
        frozen=pool.choose(0,9)
        pool.update(0,[scored(self.new,.8)],[self.new],source='generated',generation=1)
        self.assertEqual(frozen['smiles'],Chem.MolToSmiles(self.public))
        chosen=pool.choose(0,9)
        self.assertEqual(chosen['smiles'],Chem.MolToSmiles(self.new))
        self.assertEqual(chosen['source'],'generated')
        self.assertEqual(chosen['generation'],1)
        restored=FrontierSeedPool.from_state_dict(pool.state_dict())
        self.assertEqual(restored.choose(0,9),chosen)
        pool.update(0,[dict(scored(self.public,1.),feasible=False)],[self.public],generation=2)
        self.assertEqual(pool.choose(0,9),chosen)
        pool.update(1,[scored(self.public,.9)],[self.public])
        self.assertEqual(pool.choose(1,9)['smiles'],frozen['smiles'])
        with self.assertRaisesRegex(ValueError,'no feasible frontier'):
            pool.choose(2,9)

    def original(self):
        return dict(protein_pos=torch.zeros(12,3),protein_element=torch.tensor([6]*12),
            protein_atom_to_aa_type=torch.zeros(12,dtype=torch.long),protein_is_backbone=torch.zeros(12,dtype=torch.bool),
            ligand_pos=torch.tensor(self.public.GetConformer().GetPositions(),dtype=torch.float32),
            ligand_element=torch.tensor([6]*6),ligand_atom_mask=torch.tensor([-1,-1,-1,0,0,0]),
            num_arms=1,num_scaffold=1,pocket_atom_masks=torch.ones(1,12,dtype=torch.bool),
            src_ligand_filename='reference.sdf')

    def test_better_scored_conformer_of_same_graph_can_replace_parent(self):
        pool=FrontierSeedPool(self.schema)
        pool.update(0,[scored(self.public,.4)],[self.public])
        pool.update(0,[scored(self.public,.8)],[self.public],source='generated',generation=1)
        selected=pool.choose(0,9)
        self.assertEqual(selected['objectives'],[.8]*5)
        self.assertEqual(selected['generation'],1)
        self.assertEqual(len(pool.pockets['0']),1)

    def test_scored_octahedral_sulfur_identity_survives_frontier_restart(self):
        # Real interrupted rollout: score and SDF both contain OH5H, while an
        # extra SMILES parse can produce OH16H with current RDKit versions.
        mol=Chem.SDMolSupplier(str(Path(__file__).parent/'fixtures/frontier_octahedral_sulfur.sdf'),removeHs=True)[0]
        expected='O=[S@OH5H](O)(O)Oc1ccc(C2=CCCCC2)c(F)c1'
        self.assertEqual(Chem.MolToSmiles(mol),expected)
        pool=FrontierSeedPool(self.schema)
        pool.update(0,[scored(mol,.8)],[mol],generation=3,source='generated')
        parent=FrontierSeedPool.from_state_dict(pool.state_dict()).choose(0,19)
        self.assertEqual(parent['smiles'],expected)
        restored=Chem.MolFromMolBlock(parent['mol_block'])
        self.assertEqual(Chem.MolToSmiles(restored),expected)
        data=frontier_parent_data(self.original(),parent)
        self.assertEqual(len(data.ligand_pos),mol.GetNumAtoms())
        from utils.molecular_size_policy import DEFAULT_SIZE_POLICY, SizePPO
        policy=SizePPO(DEFAULT_SIZE_POLICY,5,4,6,2,'cpu')
        seeds,trace=policy.propose(parent,[.1]*4,torch.full((8,6),1/6),19)
        self.assertEqual(len(seeds),8)
        for seed,decision in zip(seeds,trace['metadata']):
            data=frontier_parent_data(self.original(),seed)
            self.assertEqual(len(data.ligand_pos),decision['seed_atoms'])

    def test_a_different_scored_molecule_is_still_rejected(self):
        pool=FrontierSeedPool(self.schema)
        with self.assertRaisesRegex(ValueError,'Scored parent identity differs'):
            pool.update(0,[scored(self.new,.8)],[self.public])

    def test_extra_objectives_preserve_tradeoffs_when_choosing_seeds(self):
        for dimension in (6,8):
            schema=ObjectiveSet([dict(name=str(i),evaluator='qed') for i in range(dimension)]).schema
            pool=FrontierSeedPool(schema)
            old=dict(scored(self.public,.5),frontier_objectives=[.5]*dimension)
            tradeoff=dict(scored(self.new,.8),frontier_objectives=[.8]*(dimension-1)+[.4])
            pool.update(0,[old],[self.public])
            pool.update(0,[tradeoff],[self.new],source='generated',generation=1)
            self.assertEqual(len(pool.pockets['0']),2)
            self.assertEqual(len(pool.choose(0,9)['objectives']),dimension)

    def test_new_parent_rebuilds_topology_and_real_forward_state(self):
        pool=FrontierSeedPool(self.schema);pool.update(0,[scored(self.new,.8)],[self.new])
        parent=pool.choose(0,9);original=self.original();saved=copy.deepcopy(original)
        data=frontier_parent_data(original,parent)
        self.assertEqual(len(data.ligand_pos),self.new.GetNumAtoms())
        self.assertEqual(data.ligand_element.tolist(),[a.GetAtomicNum() for a in self.new.GetAtoms()])
        self.assertEqual(data.ligand_bond_index.shape[1],2*self.new.GetNumBonds())
        torch.testing.assert_close(original['ligand_pos'],saved['ligand_pos'])
        torch.testing.assert_close(data.protein_pos,original['protein_pos'])
        self.assertEqual(len(data.ligand_atom_mask),self.new.GetNumAtoms())
        transform=Compose([transforms.ComputeLigandAtomNoiseDist('ref_prior'),transforms.FeaturizeProteinAtom(),
            transforms.FeaturizeLigandAtom('basic'),transforms.AddDecompIndicator(add_ord_feat=False),
            transforms.FeaturizeLigandBond(mode='fc',set_bond_type=True)])
        kernel=SimpleNamespace(q_v_sample=lambda onehot,t,b:(onehot.argmax(-1),onehot))
        model=SimpleNamespace(alphas_cumprod=torch.tensor([1.,1.]),num_classes=8,num_bond_classes=5,
            atom_type_trans=kernel,bond_type_trans=kernel)
        exp=SimpleNamespace(original={0:original},device='cpu',train_transform=transform,
            sampling=SimpleNamespace(sample=SimpleNamespace(energy_drift=None)),
            cfg=SimpleNamespace(data=SimpleNamespace(transform=SimpleNamespace(ligand_atom_mode='basic'))))
        seen=[]
        def rollout(policy,**context):
            seen.append(context)
            return dict(pos=context['init_ligand_pos'],v=context['init_ligand_v'],bond=context['init_ligand_fc_bond_type'])
        protein=SimpleNamespace(to_dict_atom=lambda:dict(pos=np.zeros((12,3))))
        with patch('utils.local_diffusion.PDBProtein',return_value=protein), \
             patch('utils.local_diffusion.rollout',side_effect=rollout), \
             patch('utils.local_diffusion.reconstruct_molecule',return_value=(parent['smiles'],self.new,None)):
            rows,_=sample_lead(exp,model,dict(target_id=0,receptor='test.pdb',source_ligand='reference.sdf'),
                2,9,0,None,1.,1,frontier_parent=parent)
        n=self.new.GetNumAtoms()
        self.assertEqual(seen[0]['init_ligand_pos'].shape,(2*n,3))
        np.testing.assert_allclose(seen[0]['init_ligand_pos'][:n],data.ligand_pos,atol=1e-6)
        self.assertTrue(all(r['parent_smiles']==parent['smiles'] and r['lead_source'].startswith('frontier:') for r in rows))
        self.assertTrue(all(len(r['decomp_mask'])==n for r in rows))


if __name__=='__main__':unittest.main()
