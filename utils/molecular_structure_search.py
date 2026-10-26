"""Property-blind graph-edit proposals before pocket-conditioned diffusion."""
import copy
import itertools

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

DEFAULT_STRUCTURE_POLICY = dict(version=2, operator='molecular_graph_edits_v2',
    actions=['keep','prune','contract','grow','insert','substitute','close_ring','open_ring','bond_order'],
    fragments=['C','N','O','CC','CO','CN','CCC','CCO','CCN'],
    atom_types=[6,7,8,9,16,17], min_atoms=6, max_atoms=60,
    hidden=64, actor_lr=.001, critic_lr=.001, entropy_coefficient=.03)

SCAFFOLD_STRUCTURE_POLICY = dict(copy.deepcopy(DEFAULT_STRUCTURE_POLICY),version=3,
    operator='scaffold_search_v3',actions=['keep','scaffold_reduce','prune','contract','grow','insert','substitute','close_ring','open_ring','bond_order'])

POLAR_STRUCTURE_POLICY = dict(copy.deepcopy(SCAFFOLD_STRUCTURE_POLICY),version=4,
    operator='polar_scaffold_search_v4',
    fragments=DEFAULT_STRUCTURE_POLICY['fragments']+['OC','C=O','C(=O)N','NC=O','C(=O)O'])

NOVEL_STRUCTURE_POLICY = dict(copy.deepcopy(POLAR_STRUCTURE_POLICY),version=5,
    operator='novel_scaffold_search_v5',actor_lr=.003,
    action_prior=[.1,4.,1.,1.,1.,1.,1.,1.,1.,1.])

CANDIDATE_STRUCTURE_POLICY = dict(copy.deepcopy(NOVEL_STRUCTURE_POLICY),version=6,
    operator='candidate_graph_ppo_v6',candidate_lr=.01,candidate_hidden=64,
    candidate_feature_version=2)

SURROGATE_STRUCTURE_POLICY = dict(copy.deepcopy(CANDIDATE_STRUCTURE_POLICY),version=7,
    operator='surrogate_guided_candidate_ppo_v7',surrogate=dict(version=1,warmup_molecules=32,
        samples_power=13,seed=7101,strength=1.5,relative_floor=.02,max_log_span=5.))

FRAGMENT_STRUCTURE_POLICY = dict(copy.deepcopy(SURROGATE_STRUCTURE_POLICY),version=8,
    operator='frontier_fragment_candidate_ppo_v8',
    actions=SURROGATE_STRUCTURE_POLICY['actions']+['fragment_graft'],
    action_prior=SURROGATE_STRUCTURE_POLICY['action_prior']+[2.],
    fragment_min_atoms=4,fragment_max_atoms=8,max_donor_fragments=32)

REPLACEMENT_STRUCTURE_POLICY = dict(copy.deepcopy(FRAGMENT_STRUCTURE_POLICY),version=9,
    operator='frontier_fragment_replace_ppo_v9',
    actions=FRAGMENT_STRUCTURE_POLICY['actions']+['fragment_replace'],
    action_prior=FRAGMENT_STRUCTURE_POLICY['action_prior']+[2.],
    replacement_max_atoms=12)


def validate_structure_config(cfg):
    template={3:SCAFFOLD_STRUCTURE_POLICY,4:POLAR_STRUCTURE_POLICY,5:NOVEL_STRUCTURE_POLICY,
              6:CANDIDATE_STRUCTURE_POLICY,7:SURROGATE_STRUCTURE_POLICY,
              8:FRAGMENT_STRUCTURE_POLICY,9:REPLACEMENT_STRUCTURE_POLICY}.get(cfg.get('version'),DEFAULT_STRUCTURE_POLICY)
    optional={'ring_closure_max_aromatic_overlap'} if 'ring_closure_max_aromatic_overlap' in cfg else set()
    if (set(cfg) != (set(template)|optional) or cfg['version'] != template['version']
            or cfg['operator'] != template['operator']
            or cfg['actions'] != template['actions']
            or not 2 <= cfg['min_atoms'] < cfg['max_atoms']
            or cfg['hidden'] < 1 or min(cfg['actor_lr'],cfg['critic_lr']) <= 0
            or cfg['entropy_coefficient'] < 0
            or not cfg['atom_types'] or not set(cfg['atom_types']) <= {6,7,8,9,15,16,17,35,53}):
        raise ValueError('Unsupported structure policy configuration')
    if optional:
        limit=cfg['ring_closure_max_aromatic_overlap']
        if type(limit) is not int or not 2<=limit<=7:
            raise ValueError('Aromatic overlap limit must be an integer from two to seven')
    if cfg['version']>=5 and (len(cfg['action_prior'])!=len(cfg['actions'])
            or not np.isfinite(cfg['action_prior']).all() or min(cfg['action_prior'])<=0):
        raise ValueError('Structure action priors must be positive finite weights')
    if cfg['version']>=6 and (cfg['candidate_lr']<=0 or cfg['candidate_hidden']<1
            or cfg['candidate_feature_version']!=2):
        raise ValueError('Unsupported candidate policy configuration')
    if cfg['version']>=7:
        s=cfg['surrogate']
        if (set(s)!=set(SURROGATE_STRUCTURE_POLICY['surrogate']) or s['version'] not in (1,2)
                or not isinstance(s['warmup_molecules'],int) or s['warmup_molecules']<2
                or not 5<=s['samples_power']<=16 or not 0<s['relative_floor']<1
                or not np.isfinite([s['strength'],s['max_log_span']]).all()
                or min(s['strength'],s['max_log_span'])<=0):
            raise ValueError('Unsupported surrogate guidance configuration')
    if cfg['version'] in (8,9) and (not 4<=cfg['fragment_min_atoms']<=cfg['fragment_max_atoms']<=12
            or not isinstance(cfg['max_donor_fragments'],int) or not 1<=cfg['max_donor_fragments']<=128):
        raise ValueError('Unsupported frontier-fragment configuration')
    if cfg['version']==9 and (type(cfg['replacement_max_atoms']) is not int
            or not 1<=cfg['replacement_max_atoms']<=12):
        raise ValueError('Unsupported fragment-replacement configuration')
    for smiles in cfg['fragments']:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None or not 1 <= mol.GetNumHeavyAtoms() <= 3 or len(Chem.GetMolFrags(mol)) != 1:
            raise ValueError('Growth fragments must be connected one-to-three-atom structures')


def structure_catalog(parent, cfg):
    from utils.molecular_size_policy import remove_seed_hydrogens
    validate_structure_config(cfg)
    mol = Chem.MolFromMolBlock(parent['mol_block'],removeHs=True,sanitize=True)
    if mol is None:
        raise ValueError('Structure policy requires a scored parent identity')
    actual_smiles = Chem.MolToSmiles(mol)
    if actual_smiles != parent['smiles']:
        # Match the seed pool's identity rule: RDKit may clean a non-tetrahedral
        # sulfur tag during a MolBlock/SMILES round trip. Keep the exact match
        # first, and accept only the same canonical isomeric identity otherwise.
        scored = Chem.MolFromSmiles(parent['smiles'])
        if scored is None or actual_smiles != Chem.MolToSmiles(scored):
            raise ValueError('Structure policy requires a scored parent identity')
    mol = remove_seed_hydrogens(mol)
    n = mol.GetNumAtoms()
    if n != mol.GetNumHeavyAtoms() or not cfg['min_atoms'] <= n <= cfg['max_atoms']:
        raise ValueError('Structure parent is outside the shared heavy-atom domain')
    catalog = [[] for _ in cfg['actions']]
    seen = [set() for _ in catalog]
    catalog[0] = [(mol,dict(kind='keep',delta=0))]
    original = Chem.MolToSmiles(mol)

    def accept(action, edit, **info):
        candidate = edit.GetMol()
        for atom in candidate.GetAtoms():
            if atom.GetAtomicNum() == 6 and atom.GetFormalCharge() == 0:
                atom.SetNoImplicit(False)
        try:
            Chem.SanitizeMol(candidate)
            Chem.AssignStereochemistry(candidate,cleanIt=True,force=True)
        except (ValueError,RuntimeError):
            return
        size = candidate.GetNumHeavyAtoms()
        if (candidate.GetNumAtoms() != size or not cfg['min_atoms'] <= size <= cfg['max_atoms']
                or len(Chem.GetMolFrags(candidate)) != 1
                or any(a.GetNumRadicalElectrons() for a in candidate.GetAtoms())):
            return
        smiles = Chem.MolToSmiles(candidate)
        index = cfg['actions'].index(action)
        if smiles == original or smiles in seen[index]:
            return
        seen[index].add(smiles)
        catalog[index].append((candidate,dict(kind=action,delta=size-n,**info)))

    def consume_h(edit, index):
        atom = edit.GetAtomWithIdx(index)
        if atom.GetNumExplicitHs():
            atom.SetNumExplicitHs(atom.GetNumExplicitHs()-1)

    if cfg['version']>=3:
        # Two cuts can release an entire fused-ring region. Unlike terminal
        # deletion this changes scaffolds in one decision, preserving the
        # original conformer for the retained connected atoms.
        bonds=[(b.GetBeginAtomIdx(),b.GetEndAtomIdx()) for b in mol.GetBonds()
               if not b.GetIsAromatic() and b.GetBondType()==Chem.BondType.SINGLE]
        for cuts in itertools.combinations(bonds,2):
            edit=Chem.RWMol(mol)
            for a,b in cuts:edit.RemoveBond(a,b)
            components=Chem.GetMolFrags(edit.GetMol(),sanitizeFrags=False)
            if len(components)<2:continue
            for keep in components:
                if not cfg['min_atoms']<=len(keep)<=n-3:continue
                candidate=Chem.RWMol(edit)
                removed=sorted(set(range(n))-set(keep),reverse=True)
                for index in removed:candidate.RemoveAtom(index)
                accept('scaffold_reduce',candidate,removed_atoms=sorted(removed),cuts=[list(c) for c in cuts])

    # Remove whole small branches, retaining the connected parent coordinates.
    for bond in mol.GetBonds():
        a,b = bond.GetBeginAtomIdx(),bond.GetEndAtomIdx()
        if not bond.IsInRing():
            cut = Chem.RWMol(mol);cut.RemoveBond(a,b)
            for branch in Chem.GetMolFrags(cut.GetMol(),sanitizeFrags=False):
                if 1 <= len(branch) <= 3 and n-len(branch) >= cfg['min_atoms']:
                    edit = Chem.RWMol(mol)
                    for index in sorted(branch,reverse=True):edit.RemoveAtom(index)
                    accept('prune',edit,removed_atoms=list(branch))
        if bond.GetBondType() == Chem.BondType.SINGLE and not bond.GetIsAromatic():
            if n < cfg['max_atoms']:
                for number in [6,7,8]:
                    edit = Chem.RWMol(mol);edit.RemoveBond(a,b)
                    new = edit.AddAtom(Chem.Atom(number))
                    edit.AddBond(a,new,Chem.BondType.SINGLE);edit.AddBond(new,b,Chem.BondType.SINGLE)
                    midpoint=(np.asarray(mol.GetConformer().GetAtomPosition(a))+
                              np.asarray(mol.GetConformer().GetAtomPosition(b)))/2
                    edit.GetConformer().SetAtomPosition(new,midpoint)
                    accept('insert',edit,new_atoms=[new],bond=[a,b],element=number)
            if bond.IsInRing():
                edit=Chem.RWMol(mol);edit.RemoveBond(a,b)
                accept('open_ring',edit,bond=[a,b])
        if not bond.GetIsAromatic() and bond.GetBondType() in (Chem.BondType.SINGLE,Chem.BondType.DOUBLE):
            edit=Chem.RWMol(mol)
            order=Chem.BondType.DOUBLE if bond.GetBondType()==Chem.BondType.SINGLE else Chem.BondType.SINGLE
            if order==Chem.BondType.DOUBLE:
                if not all(mol.GetAtomWithIdx(i).GetTotalNumHs()>0 for i in [a,b]):continue
                consume_h(edit,a);consume_h(edit,b)
            edit.GetBondBetweenAtoms(a,b).SetBondType(order)
            accept('bond_order',edit,bond=[a,b],order=float(order))

    sites=[]
    for atom in mol.GetAtoms():
        i=atom.GetIdx();neighbors=[a.GetIdx() for a in atom.GetNeighbors()]
        if (n>cfg['min_atoms'] and atom.GetDegree()==2 and not atom.GetIsAromatic()
                and atom.GetFormalCharge()==0
                and all(b.GetBondType()==Chem.BondType.SINGLE for b in atom.GetBonds())
                and mol.GetBondBetweenAtoms(*neighbors) is None):
            edit=Chem.RWMol(mol);edit.AddBond(*neighbors,Chem.BondType.SINGLE);edit.RemoveAtom(i)
            accept('contract',edit,atom=i)
        if atom.GetFormalCharge()==0 and not atom.GetIsAromatic():
            for number in cfg['atom_types']:
                if number==atom.GetAtomicNum():continue
                edit=Chem.RWMol(mol)
                replacement=Chem.Atom(number)
                edit.ReplaceAtom(i,replacement)
                accept('substitute',edit,atom=i,element=number)
        if cfg['version']>=4 and atom.GetIsAromatic() and atom.GetFormalCharge()==0 and atom.GetDegree()==2:
            # Exchange aromatic CH and pyridine-like N without deleting the
            # ring or changing its atom count. Do not replace pyrrolic [nH]
            # or fused-ring junctions, whose valence/electron roles differ.
            number=None
            if atom.GetAtomicNum()==6 and atom.GetTotalNumHs()==1 and 7 in cfg['atom_types']:
                number=7
            elif atom.GetAtomicNum()==7 and atom.GetTotalNumHs()==0 and 6 in cfg['atom_types']:
                number=6
            if number is not None:
                edit=Chem.RWMol(mol);replacement=Chem.Atom(number)
                replacement.SetIsAromatic(True)
                replacement.SetNoImplicit(number==7)
                edit.ReplaceAtom(i,replacement)
                accept('substitute',edit,atom=i,element=number,aromatic=True)
        if atom.GetAtomicNum() in (6,7,8,16) and atom.GetFormalCharge()==0 and atom.GetTotalNumHs()>0:
            sites.append(i)
            for smiles in cfg['fragments']:
                fragment=Chem.MolFromSmiles(smiles)
                if n+fragment.GetNumAtoms()>cfg['max_atoms']:continue
                edit=Chem.RWMol(mol);consume_h(edit,i)
                for child in fragment.GetAtoms():edit.AddAtom(Chem.Atom(child))
                for bond in fragment.GetBonds():
                    edit.AddBond(n+bond.GetBeginAtomIdx(),n+bond.GetEndAtomIdx(),bond.GetBondType())
                edit.AddBond(i,n,Chem.BondType.SINGLE)
                accept('grow',edit,anchor=i,new_atoms=list(range(n,edit.GetNumAtoms())),fragment=smiles)
    # Optional graph-only restriction on short aromatic bridges. Cycles that
    # bridge several atoms of a rigid aromatic ring often fail common 3D
    # embedding. Preserve ordinary fused-ring annulations (two shared atoms).
    overlap_limit=cfg.get('ring_closure_max_aromatic_overlap')
    aromatic_rings=([set(ring) for ring in mol.GetRingInfo().AtomRings()
                     if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring)]
                    if overlap_limit is not None else [])
    for a,b in itertools.combinations(sites,2):
        if mol.GetBondBetweenAtoms(a,b) is not None:continue
        path=Chem.GetShortestPath(mol,a,b)
        if 4<=len(path)<=7:
            if overlap_limit is not None and any(len(set(path)&ring)>overlap_limit for ring in aromatic_rings):
                continue
            edit=Chem.RWMol(mol);consume_h(edit,a);consume_h(edit,b)
            edit.AddBond(a,b,Chem.BondType.SINGLE)
            accept('close_ring',edit,bond=[a,b])
    return mol,catalog


def prepare_structure_edit(mol, info, rng):
    from utils.molecular_size_policy import remove_seed_hydrogens
    mol=Chem.Mol(mol)
    if info['kind']=='keep':return mol
    conf=mol.GetConformer()
    new_atoms=set(info.get('new_atoms',[]))
    if info['kind']=='grow':
        placed=set(range(min(new_atoms)))
        pending=set(new_atoms)
        while pending:
            progressed=False
            for i in sorted(pending):
                anchors=[a.GetIdx() for a in mol.GetAtomWithIdx(i).GetNeighbors() if a.GetIdx() in placed]
                if not anchors:continue
                anchor=anchors[0];positions=np.asarray(conf.GetPositions())
                neighbors=[a.GetIdx() for a in mol.GetAtomWithIdx(anchor).GetNeighbors()
                           if a.GetIdx() in placed and a.GetIdx()!=i]
                direction=positions[anchor]-positions[neighbors].mean(0) if neighbors else rng.normal(size=3)
                direction+=.3*rng.normal(size=3)
                direction/=max(np.linalg.norm(direction),1e-8)
                conf.SetAtomPosition(i,positions[anchor]+1.45*direction)
                pending.remove(i);placed.add(i);progressed=True
            if not progressed:raise ValueError('Growth fragment has no connected anchor')
    hydrogenated=Chem.AddHs(mol,addCoords=True)
    props=AllChem.MMFFGetMoleculeProperties(hydrogenated)
    ff=(AllChem.MMFFGetMoleculeForceField(hydrogenated,props) if props is not None else
        AllChem.UFFGetMoleculeForceField(hydrogenated) if AllChem.UFFHasAllMoleculeParams(hydrogenated) else None)
    if ff is not None:
        # Allow ring and linker edits to relax further than single-atom additions.
        for i in range(mol.GetNumAtoms()):
            if i in new_atoms:continue
            if props is not None:ff.MMFFAddPositionConstraint(i,.75,20.)
            else:ff.UFFAddPositionConstraint(i,.75,20.)
        coordinates=np.array(hydrogenated.GetConformer().GetPositions(),copy=True)
        try:
            ff.Initialize();ff.Minimize(maxIts=200)
        except RuntimeError as error:
            # An RDKit line-search failure need not discard a legal edit or
            # substitute another proposal. Restore this exact edit's initial
            # coordinates and let diffusion plus the common scorer assess it.
            for i,position in enumerate(coordinates):
                hydrogenated.GetConformer().SetAtomPosition(i,position)
            info['local_relaxation']='retained_unrelaxed_conformer'
            info['relaxation_error']=str(error)
    mol=remove_seed_hydrogens(hydrogenated)
    if not np.isfinite(mol.GetConformer().GetPositions()).all():
        raise FloatingPointError('Nonfinite structure-edit coordinates')
    Chem.AssignStereochemistryFrom3D(mol,replaceExistingTags=True)
    return mol
