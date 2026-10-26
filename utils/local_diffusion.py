"""Local diffusion proposals: forward-noise a frozen parent, then denoise.

The parent may be selected from a historical frontier before the rollout.
Its initial distribution stays fixed during that PPO update, so the reverse
transition likelihood ratio conditions on the same parent and history.
Every attempted reconstruction is retained.
"""
import copy
import logging
import random

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import BRICS
from torch_geometric.data import Batch

from datasets.pl_data import FOLLOW_BATCH
from models.transitions import index_to_log_onehot
from utils.diffusion_ppo import EXCLUDE, ROOT, rollout
from utils import reconstruct as recon
from utils import misc, prior, transforms
from utils.data import PDBProtein, ProteinLigandData, parse_sdf_file, torchify_dict


def reconstruct_molecule(i, pred_pos, pred_atom_type, pred_v, pred_bond_index=None, pred_bond_type=None, atom_enc_mode=None):
    try:
        mol = recon.reconstruct_from_generated_with_bond(pred_pos, pred_atom_type, pred_bond_index, pred_bond_type)
        smiles = Chem.MolToSmiles(mol)
        return smiles, mol, True 
    except recon.MolReconsError:
        logging.getLogger('parexmol').warning(f'Reconstruct failed {i}')
        return '', None, False


def frontier_parent_data(original, parent):
    """Rebuild ligand tensors and spatial decomposition for a different molecule.

    BRICS fragments remain intact and are assigned to the nearest original
    arm/scaffold region. Empty arm regions are removed and indices compacted.
    Protein tensors remain those of this pocket; no old ligand features survive.
    """
    mol = Chem.MolFromMolBlock(parent['mol_block'], removeHs=True, sanitize=True)
    if mol is None or Chem.MolToSmiles(mol) != parent['smiles']:
        raise ValueError('Invalid or mismatched frontier parent MolBlock')
    reference_pos = torch.as_tensor(original['ligand_pos']).float()
    reference_mask = torch.as_tensor(original['ligand_atom_mask'])
    labels = sorted(set(reference_mask.tolist()))
    centers = torch.stack([reference_pos[reference_mask == label].mean(0) for label in labels])
    split = Chem.RWMol(mol)
    for (a, b), _ in BRICS.FindBRICSBonds(mol):
        split.RemoveBond(a, b)
    fragments = Chem.GetMolFrags(split.GetMol(), asMols=False, sanitizeFrags=False)
    coordinates = np.asarray(mol.GetConformer().GetPositions())
    assignment = torch.empty(mol.GetNumAtoms(), dtype=torch.long)
    for fragment in fragments:
        center = torch.tensor(coordinates[list(fragment)].mean(0), dtype=torch.float32)
        nearest = int(torch.linalg.vector_norm(centers-center, dim=1).argmin())
        assignment[list(fragment)] = labels[nearest]
    present_arms = sorted(set(assignment.tolist())-{-1})
    compact = assignment.clone()
    for new, old in enumerate(present_arms):
        compact[assignment == old] = new
    protein = {k.removeprefix('protein_'): copy.deepcopy(v) for k, v in original.items() if k.startswith('protein_')}
    data = ProteinLigandData.from_protein_ligand_dicts(
        protein_dict=protein, ligand_dict=torchify_dict(parse_sdf_file(Chem.Mol(mol))))
    data.num_arms = len(present_arms)
    data.num_scaffold = int((compact == -1).any())
    data.ligand_atom_mask = compact
    # Recompute protein contacts for the actual parent arm geometry.
    data.pocket_atom_masks = torch.stack([
        torch.linalg.vector_norm(data.protein_pos-data.ligand_pos[compact == i].mean(0), dim=1) < 6.
        for i in range(data.num_arms)]) if data.num_arms else torch.zeros((0, len(data.protein_pos)), dtype=torch.bool)
    data.src_ligand_filename = original['src_ligand_filename']
    return data


def forward_noise(model, batch, timestep):
    """Use the pretrained model's joint position/atom/bond forward kernels."""
    atom_batch = batch.ligand_element_batch
    times = torch.full((int(atom_batch.max())+1,), timestep, dtype=torch.long, device=atom_batch.device)
    centers = batch.ligand_decomp_centers[batch.ligand_decomp_mask]
    stds = batch.ligand_decomp_stds[batch.ligand_decomp_mask]
    alpha = model.alphas_cumprod[times][atom_batch, None]
    position = centers + alpha.sqrt()*(batch.ligand_pos-centers) + (1-alpha).sqrt()*stds*torch.randn_like(batch.ligand_pos)
    atoms, _ = model.atom_type_trans.q_v_sample(
        index_to_log_onehot(batch.ligand_atom_feature_full, model.num_classes), times, atom_batch)
    bonds, _ = model.bond_type_trans.q_v_sample(
        index_to_log_onehot(batch.ligand_fc_bond_type, model.num_bond_classes), times,
        batch.ligand_fc_bond_type_batch)
    return position, atoms, bonds


@torch.no_grad()
def sample_lead(exp, model, target, count, seed, trace_steps, objective_condition,
                guidance_scale, start_timestep, history_condition=None, frontier_parent=None,
                resized_parents=None, size_metadata=None, seed_atom_logit_bias=0.,seed_bond_logit_bias=0.,
                frontier_parents=None):
    if (not isinstance(start_timestep, int) or not 1 <= start_timestep < 1000
            or not 0 <= trace_steps <= start_timestep):
        raise ValueError('Lead optimization needs 1<=start_timestep<1000 and enough trace steps')
    if not all(np.isfinite(x) and x>=0 for x in (seed_atom_logit_bias,seed_bond_logit_bias)):
        raise ValueError('Seed logit biases must be finite and nonnegative')
    misc.seed_all(seed)
    if frontier_parents is not None and (len(frontier_parents)!=count
            or any(p['pocket']!=str(target['target_id']) for p in frontier_parents)):
        raise ValueError('One same-pocket frontier parent per attempt is required')
    if resized_parents is not None and ((frontier_parent is None and frontier_parents is None) or len(resized_parents) != count
            or size_metadata is None or len(size_metadata) != count):
        raise ValueError('One resized seed and size decision per attempted molecule required')
    source_parents=[frontier_parent]*count if frontier_parents is None else frontier_parents
    parents = source_parents if resized_parents is None else resized_parents
    data_list = []
    for parent in parents:
        if parent is not None:
            if parent['pocket'] != str(target['target_id']):
                raise ValueError('Frontier parent belongs to another pocket')
            data = frontier_parent_data(exp.original[target['target_id']], parent)
        else:
            data = ProteinLigandData(**copy.deepcopy(exp.original[target['target_id']]))
        prior.compute_golden_prior_from_data(data)
        data_list.append(exp.train_transform(data))
    batch = Batch.from_data_list(data_list,
                                 follow_batch=FOLLOW_BATCH, exclude_keys=EXCLUDE).to(exp.device)
    position, atoms, bonds = forward_noise(model, batch, start_timestep)
    protein = PDBProtein(str(ROOT/'data/crossdocked_v1.1_rmsd1.0'/target['receptor']))
    full = torch.as_tensor(protein.to_dict_atom()['pos'], dtype=torch.float32, device=exp.device)
    model.trace_steps = set(random.Random(seed+900000).sample(range(1, start_timestep+1), trace_steps))
    model.saved_traces = []
    context = dict(protein_pos=batch.protein_pos, protein_v=batch.protein_atom_feature.float(),
                   batch_protein=batch.protein_element_batch, init_ligand_pos=position, init_ligand_v=atoms,
                   ligand_v_aux=batch.ligand_atom_aux_feature.float(), batch_ligand=batch.ligand_element_batch,
                   prior_centers=batch.ligand_decomp_centers, prior_stds=batch.ligand_decomp_stds,
                   ligand_fc_bond_index=batch.ligand_fc_bond_index, init_ligand_fc_bond_type=bonds,
                   batch_ligand_bond=batch.ligand_fc_bond_type_batch, ligand_decomp_batch=batch.ligand_decomp_mask,
                   ligand_decomp_index=batch.ligand_atom_mask, energy_drift_opt=exp.sampling.sample.energy_drift,
                   full_protein_pos=full.repeat(count, 1),
                   full_batch_protein=torch.arange(count, device=exp.device).repeat_interleave(len(full)),
                   objective_condition=objective_condition, guidance_scale=guidance_scale,
                   start_timestep=start_timestep)
    if history_condition is not None:
        context['history_condition'] = history_condition
    if seed_atom_logit_bias:
        context.update(seed_atom_types=batch.ligand_atom_feature_full.detach().clone(),
                       seed_atom_logit_bias=float(seed_atom_logit_bias))
    if seed_bond_logit_bias:
        context.update(seed_bond_types=batch.ligand_fc_bond_type.detach().clone(),
                       seed_bond_logit_bias=float(seed_bond_logit_bias))
    output = rollout(model, **context)
    mode = exp.cfg.data.transform.ligand_atom_mode
    records = []
    for i in range(count):
        atom_mask = batch.ligand_element_batch == i
        bond_mask = batch.ligand_fc_bond_type_batch == i
        offset = int(atom_mask.nonzero()[0])
        pos = output['pos'][atom_mask].cpu().numpy().astype(float)
        atom = output['v'][atom_mask].cpu().numpy()
        bond = output['bond'][bond_mask].cpu().numpy()
        edges = (batch.ligand_fc_bond_index[:, bond_mask]-offset).cpu().tolist()
        numbers = transforms.get_atomic_number_from_index(atom, mode=mode)
        smiles, mol, _ = reconstruct_molecule(i, pos, numbers, atom, edges, bond, mode)
        source_parent=source_parents[i]
        parent_info = {} if source_parent is None else dict(
            parent_smiles=source_parent['smiles'], parent_source=source_parent['source'],
            parent_generation=source_parent['generation'], parent_source_ref=source_parent['source_ref'])
        records.append(dict(mol=mol, smiles=smiles, pred_pos=pos, pred_v=atom,
                            pred_bond_index=edges, pred_bond_type=bond,
                            decomp_mask=data_list[i].ligand_atom_mask.tolist(),
                            lead_source=target['source_ligand'] if source_parent is None else 'frontier:'+source_parent['smiles'],
                            reference_ligand=target['source_ligand'], start_timestep=start_timestep, **parent_info))
        if size_metadata is not None:
            records[-1]['size_action'] = size_metadata[i]
    return records, model.saved_traces
