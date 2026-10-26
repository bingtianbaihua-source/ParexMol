"""Apply the project's empirical geometry score to its available statistics."""
from itertools import combinations
import math
from rdkit.Chem import rdMolTransforms
from utils.transforms import MAP_ATOM_TYPE_ONLY_TO_INDEX


def geometry_score(amber,mol):
    # AmberFeaturizer already omits bond/angle patterns absent from its tables.
    # Extend that behavior to elements outside the diffusion atom vocabulary;
    # never substitute another element or invent an energy for missing statistics.
    conf=mol.GetConformer()
    indices=[MAP_ATOM_TYPE_ONLY_TO_INDEX.get(a.GetAtomicNum()) for a in mol.GetAtoms()]
    total=0.; assessed=0; possible=0
    for bond in mol.GetBonds():
        i,j=bond.GetBeginAtomIdx(),bond.GetEndAtomIdx()
        possible+=1
        if indices[i] is None or indices[j] is None:
            continue
        key=tuple(sorted((indices[i],indices[j])))
        if key in amber.bond_info:
            total+=amber.relu_mse_score(rdMolTransforms.GetBondLength(conf,i,j),amber.bond_info[key]); assessed+=1
    for atom in mol.GetAtoms():
        j=atom.GetIdx()
        for a,b in combinations(atom.GetNeighbors(),2):
            i,k=a.GetIdx(),b.GetIdx(); possible+=1
            if any(indices[t] is None for t in (i,j,k)):
                continue
            first,last=sorted((indices[i],indices[k]))
            key=(first,indices[j],last)
            if key in amber.angle_info:
                value=rdMolTransforms.GetAngleDeg(conf,i,j,k)*math.pi/180
                total+=amber.relu_mse_score(value,amber.angle_info[key]); assessed+=1
    return float(total),dict(assessed_terms=assessed,total_terms=possible,
        coverage=assessed/possible if possible else 0.,
        unsupported_elements=sorted({a.GetSymbol() for a in mol.GetAtoms() if a.GetAtomicNum() not in MAP_ATOM_TYPE_ONLY_TO_INDEX}))
