"""Graft or replace fragments using this run's already-scored frontier.

Library construction and graph legality checks query no property oracles.
Each sampled graft still enters diffusion and the counted common evaluator.
"""
import copy

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from scipy.spatial.transform import Rotation

from utils.molecular_structure_search import structure_catalog

FP=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=512)


def fragment_library(donors,cfg):
    unique={}
    for donor in sorted(donors,key=lambda r:(r['smiles'],str(r.get('source_ref','')),r.get('generation',0))):
        mol=Chem.MolFromMolBlock(donor['mol_block'],removeHs=True,sanitize=True)
        if mol is None:raise ValueError('Fragment donor must have a scored conformer')
        positions=mol.GetConformer().GetPositions()
        for bond in mol.GetBonds():
            if bond.IsInRing() or bond.GetBondType()!=Chem.BondType.SINGLE:continue
            a,b=bond.GetBeginAtomIdx(),bond.GetEndAtomIdx()
            cut=Chem.RWMol(mol);cut.RemoveBond(a,b)
            for keep in Chem.GetMolFrags(cut.GetMol(),sanitizeFrags=False):
                if not cfg['fragment_min_atoms']<=len(keep)<=cfg['fragment_max_atoms']:continue
                root,outside=(a,b) if a in keep else (b,a)
                incoming=positions[root]-positions[outside]
                norm=np.linalg.norm(incoming)
                if norm<1e-8:continue
                fragment=Chem.RWMol(mol)
                for index in sorted(set(range(mol.GetNumAtoms()))-set(keep),reverse=True):fragment.RemoveAtom(index)
                root_index=sorted(keep).index(root)
                atom=fragment.GetAtomWithIdx(root_index);atom.SetNoImplicit(False)
                try:Chem.SanitizeMol(fragment)
                except (ValueError,RuntimeError):continue
                if (fragment.GetNumAtoms()!=fragment.GetNumHeavyAtoms() or
                        any(x.GetAtomicNum() not in cfg['atom_types'] or x.GetNumRadicalElectrons() for x in fragment.GetAtoms())
                        or fragment.GetAtomWithIdx(root_index).GetTotalNumHs()<1):continue
                marked=Chem.Mol(fragment)
                for x in marked.GetAtoms():x.SetAtomMapNum(0)
                marked.GetAtomWithIdx(root_index).SetAtomMapNum(1)
                key=Chem.MolToSmiles(marked)
                if key in unique:continue
                info=dict(fragment=Chem.MolToSmiles(fragment),rooted_fragment=key,fragment_root=root_index,
                    donor_smiles=donor['smiles'],donor_source=donor['source'],donor_generation=donor['generation'],
                    donor_cut=[root,outside],incoming_unit=(incoming/norm).tolist())
                unique[key]=(fragment.GetMol(),info)
    entries=[unique[key] for key in sorted(unique)]
    if len(entries)<=cfg['max_donor_fragments']:return entries
    # Keep diverse graph fragments, using no measured or predicted properties.
    fps=np.stack([FP.GetFingerprintAsNumPy(m).astype(float) for m,_ in entries])
    dot=fps@fps.T
    distance=1-dot/np.maximum(fps.sum(1)[:,None]+fps.sum(1)[None,:]-dot,1e-12)
    selected=[max(range(len(entries)),key=lambda i:entries[i][0].GetNumHeavyAtoms())]
    while len(selected)<cfg['max_donor_fragments']:
        separation=distance[:,selected].min(1);separation[selected]=-1
        selected.append(int(separation.argmax()))
    return [entries[i] for i in sorted(selected)]


def fragment_catalog(parent,cfg,donors):
    mol,catalog=structure_catalog(parent,cfg);n=mol.GetNumAtoms()
    if any(d['pocket']!=parent['pocket'] for d in donors):raise ValueError('Fragment donors crossed pocket boundaries')
    library=fragment_library(donors,cfg)
    original=Chem.MolToSmiles(mol);seen={original};positions=mol.GetConformer().GetPositions()
    sites=[a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() in (6,7,8,16)
           and a.GetFormalCharge()==0 and a.GetTotalNumHs()>0]
    for anchor in sites:
        neighbors=[a.GetIdx() for a in mol.GetAtomWithIdx(anchor).GetNeighbors()]
        direction=positions[anchor]-positions[neighbors].mean(0)
        norm=np.linalg.norm(direction)
        if norm<1e-8:continue
        direction/=norm
        for fragment,source in library:
            count=fragment.GetNumAtoms();root=source['fragment_root']
            if n+count>cfg['max_atoms']:continue
            fragment=Chem.Mol(fragment)
            rotation=Rotation.align_vectors(direction[None,:],np.asarray(source['incoming_unit'])[None,:])[0].as_matrix()
            xyz=fragment.GetConformer().GetPositions()
            xyz=(xyz-xyz[root])@rotation.T+positions[anchor]+1.5*direction
            for i,point in enumerate(xyz):fragment.GetConformer().SetAtomPosition(i,point)
            edit=Chem.RWMol(Chem.CombineMols(mol,fragment))
            for index in (anchor,n+root):
                atom=edit.GetAtomWithIdx(index)
                if atom.GetNumExplicitHs():atom.SetNumExplicitHs(atom.GetNumExplicitHs()-1)
            edit.AddBond(anchor,n+root,Chem.BondType.SINGLE)
            try:
                Chem.SanitizeMol(edit);Chem.AssignStereochemistry(edit,cleanIt=True,force=True)
            except (ValueError,RuntimeError):continue
            if any(a.GetNumRadicalElectrons() for a in edit.GetAtoms()):continue
            smiles=Chem.MolToSmiles(edit)
            if smiles in seen:continue
            seen.add(smiles)
            info=dict(copy.deepcopy(source),kind='fragment_graft',delta=count,anchor=anchor,
                      new_atoms=list(range(n,n+count)))
            catalog[cfg['actions'].index('fragment_graft')].append((edit.GetMol(),info))
    if cfg['version']==9:
        catalog[cfg['actions'].index('fragment_replace')]=replacement_catalog(mol,cfg,library)
    return mol,catalog


def replacement_catalog(mol,cfg,library):
    """Replace one side of an acyclic single bond, retaining the other pose.

    The replaced component may contain rings and up to 12 heavy atoms. Donors
    remain the same scored 4--8-atom fragments as grafting. No capped core is
    scored or sampled: a complete replacement is one concrete PPO action.
    """
    n=mol.GetNumAtoms();positions=mol.GetConformer().GetPositions()
    entries=[];seen={Chem.MolToSmiles(mol)}
    for bond in mol.GetBonds():
        if bond.IsInRing() or bond.GetBondType()!=Chem.BondType.SINGLE:continue
        a,b=bond.GetBeginAtomIdx(),bond.GetEndAtomIdx()
        cut=Chem.RWMol(mol);cut.RemoveBond(a,b)
        for removed in Chem.GetMolFrags(cut.GetMol(),sanitizeFrags=False):
            if not 1<=len(removed)<=cfg['replacement_max_atoms']:continue
            anchor,departing=(b,a) if a in removed else (a,b)
            direction=positions[departing]-positions[anchor]
            norm=np.linalg.norm(direction)
            if norm<1e-8:continue
            direction/=norm
            retained=sorted(set(range(n))-set(removed));count_retained=len(retained)
            core=Chem.RWMol(mol)
            for index in sorted(removed,reverse=True):core.RemoveAtom(index)
            new_anchor=retained.index(anchor)
            for fragment,source in library:
                count=fragment.GetNumAtoms();root=source['fragment_root']
                if not cfg['min_atoms']<=count_retained+count<=cfg['max_atoms']:continue
                fragment=Chem.Mol(fragment)
                rotation=Rotation.align_vectors(direction[None,:],np.asarray(source['incoming_unit'])[None,:])[0].as_matrix()
                xyz=fragment.GetConformer().GetPositions()
                xyz=(xyz-xyz[root])@rotation.T+positions[anchor]+1.5*direction
                for i,point in enumerate(xyz):fragment.GetConformer().SetAtomPosition(i,point)
                edit=Chem.RWMol(Chem.CombineMols(core,fragment))
                # The retained anchor replaces an existing bond, so it does
                # not lose an H. The donor root consumes its exposed H.
                atom=edit.GetAtomWithIdx(count_retained+root)
                if atom.GetNumExplicitHs():atom.SetNumExplicitHs(atom.GetNumExplicitHs()-1)
                edit.AddBond(new_anchor,count_retained+root,Chem.BondType.SINGLE)
                try:
                    Chem.SanitizeMol(edit);Chem.AssignStereochemistry(edit,cleanIt=True,force=True)
                except (ValueError,RuntimeError):continue
                if (edit.GetNumAtoms()!=edit.GetNumHeavyAtoms() or len(Chem.GetMolFrags(edit))!=1
                        or any(x.GetNumRadicalElectrons() for x in edit.GetAtoms())):continue
                smiles=Chem.MolToSmiles(edit)
                if smiles in seen:continue
                seen.add(smiles)
                info=dict(copy.deepcopy(source),kind='fragment_replace',delta=count-len(removed),
                    anchor=new_anchor,parent_cut=[anchor,departing],removed_atoms=sorted(removed),
                    retained_atoms=retained,new_atoms=list(range(count_retained,count_retained+count)))
                entries.append((edit.GetMol(),info))
    return entries
