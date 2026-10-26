"""Per-pocket feasible frontier molecules used as frozen diffusion restart states."""
import copy
from pathlib import Path

import numpy as np
from rdkit import Chem

from utils.frontier_discovery import molecular_graph
from utils.morl import matrix, nondominated_mask


class FrontierSeedPool:
    def __init__(self, schema, selection_config=None):
        self.schema = copy.deepcopy(schema)
        self.pockets = {}
        self.selection_config=copy.deepcopy(selection_config)
        if selection_config is not None and (set(selection_config) not in
                ({'exploration','samples_power'},{'exploration','samples_power','potential_margin'})
                or not 0<selection_config['exploration']<1 or not 5<=selection_config['samples_power']<=18):
            raise ValueError('Invalid HV-guided parent selection configuration')
        if selection_config is not None and 'potential_margin' in selection_config:
            if not 0<selection_config['potential_margin']<=1:
                raise ValueError('Parent HV potential margin must be within (0,1]')

    def update(self, pocket, rows, molecules=None, generation=0, source='public'):
        """Commit scored parents only after a completed round; never query an oracle here."""
        pocket = str(pocket)
        if molecules is not None and len(molecules) != len(rows):
            raise ValueError('One molecular record per score is required')
        entries = {(r['smiles'], tuple(r['objectives'])): r for r in self.pockets.get(pocket, [])}
        for i, row in enumerate(rows):
            if not row['valid'] or not row.get('feasible', True):
                continue
            smiles = row['smiles']
            key = (smiles, tuple(row['frontier_objectives']))
            if key in entries:
                continue
            matrix([row['frontier_objectives']], len(self.schema))
            if row.get('pose_dir'):
                path = Path(row['pose_dir'])/'standardized.sdf'
                mol = next(iter(Chem.SDMolSupplier(str(path), removeHs=True)))
                pose_source = str(path)
            else:
                record = None if molecules is None else molecules[i]
                mol = record.get('mol') if isinstance(record, dict) else record
                pose_source = 'supplied_scored_conformer'
            if mol is None:
                raise ValueError(f'No scored parent conformer for {smiles}')
            mol = Chem.RemoveHs(Chem.Mol(mol))
            Chem.SanitizeMol(mol)
            if (mol.GetNumConformers() != 1 or not mol.GetConformer().Is3D()
                    or not np.isfinite(mol.GetConformer().GetPositions()).all()):
                raise ValueError('A frontier parent must have one finite 3D conformer')
            actual_smiles = Chem.MolToSmiles(mol)
            # The scorer already emits canonical isomeric SMILES. Check that
            # exact identity first: RDKit can alternate non-tetrahedral sulfur
            # permutations (OH5H/OH16H) when re-parsing its own SMILES. A second
            # canonicalization must not reject an exact scored-pose match.
            if actual_smiles != smiles and actual_smiles != Chem.MolToSmiles(Chem.MolFromSmiles(smiles)):
                raise ValueError('Scored parent identity differs from its stored conformer')
            entries[key] = dict(smiles=smiles, graph=molecular_graph(smiles),
                objectives=list(row['frontier_objectives']), mol_block=Chem.MolToMolBlock(mol),
                source=source, source_ref=row.get('source_key', pose_source), pose_source=pose_source,
                generation=int(generation), pocket=pocket)
        ordered = [entries[k] for k in sorted(entries)]
        points = matrix([r['objectives'] for r in ordered], len(self.schema))
        self.pockets[pocket] = [r for r, keep in zip(ordered, nondominated_mask(points)) if keep]

    def choose(self, pocket, seed):
        pocket = str(pocket)
        entries = self.pockets.get(pocket, [])
        if not entries:
            raise ValueError(f'Pocket {pocket} has no feasible frontier molecule to restart from')
        graphs = sorted({r['graph'] for r in entries})
        rng = np.random.default_rng(seed)
        probabilities=None
        if self.selection_config is None:
            graph = graphs[int(rng.integers(len(graphs)))]
        else:
            from scipy.stats import qmc
            points=np.asarray([r['objectives'] for r in entries])
            if np.any(points<0) or np.any(points>1):raise ValueError('HV parent selection needs bounded unit-box coordinates')
            probes=qmc.Sobol(len(self.schema),scramble=True,seed=7101).random_base2(self.selection_config['samples_power'])
            masks=np.stack([np.any(np.all(probes[None,:,:]<=points[[i for i,r in enumerate(entries) if r['graph']==g],None,:],axis=-1),axis=0) for g in graphs])
            if 'potential_margin' in self.selection_config:
                # A local optimistic box is a parent-selection heuristic, not
                # a property prediction or an HV reward. Only cells outside
                # the actual scored front are eligible. Share overlapping
                # potential cells so redundant parents receive no extra mass.
                uncovered=~masks.any(0)
                optimistic=np.minimum(points+self.selection_config['potential_margin'],1.)
                masks=np.stack([np.any(np.all(probes[None,:,:]<=optimistic[[i for i,r in enumerate(entries)
                    if r['graph']==g],None,:],axis=-1),axis=0)&uncovered for g in graphs])
            # Share each covered cell among graphs covering it. This remains
            # nonzero for redundant Pareto points and favors useful HV regions.
            values=(masks/np.maximum(masks.sum(0),1)).mean(1)
            exploitation=values/values.sum() if values.sum()>0 else np.full(len(graphs),1/len(graphs))
            epsilon=self.selection_config['exploration']
            probabilities=(1-epsilon)*exploitation+epsilon/len(graphs)
            graph=graphs[int(rng.choice(len(graphs),p=probabilities))]
        variants = sorted({r['smiles'] for r in entries if r['graph'] == graph})
        smiles = variants[int(rng.integers(len(variants)))]
        conformers = [r for r in entries if r['smiles'] == smiles]
        selected = copy.deepcopy(conformers[int(rng.integers(len(conformers)))])
        selected.update(selection_seed=int(seed), frontier_size=len(entries), frontier_graphs=len(graphs),
                        selection='uniform_graph_then_stereo_then_conformer')
        if probabilities is not None:
            selected.update(selection='hv_shared_coverage_graph_then_stereo_then_conformer',
                graph_probability=float(probabilities[graphs.index(graph)]),selection_config=self.selection_config)
            if 'potential_margin' in self.selection_config:
                selected['selection']='hv_local_potential_graph_then_stereo_then_conformer'
        return selected

    def state_dict(self):
        state=dict(version=1, schema=copy.deepcopy(self.schema), pockets=copy.deepcopy(self.pockets))
        if self.selection_config is not None:state['selection_config']=copy.deepcopy(self.selection_config)
        return state

    @classmethod
    def from_state_dict(cls, state):
        if state['version'] != 1:
            raise ValueError('Unsupported frontier seed state')
        result = cls(state['schema'],state.get('selection_config'))
        result.pockets = copy.deepcopy(state['pockets'])
        return result


def parent_metadata(parent):
    return {key: copy.deepcopy(value) for key, value in parent.items() if key != 'mol_block'}
