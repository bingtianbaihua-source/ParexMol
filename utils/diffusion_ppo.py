"""Diffusion transitions, trajectory likelihoods and molecular input preparation."""
import json
import math
import pickle
import time
from pathlib import Path

import lmdb
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.transforms import Compose
from torch_scatter import scatter_mean, scatter_sum

from models.common import extract
from models.decompdiff import center_pos
from models.transitions import index_to_log_onehot, log_sample_categorical
from utils import misc, transforms as trans
from utils.runtime import ROOT, DB_STEM

EXCLUDE = ['ligand_nbh_list', 'pocket_atom_masks', 'pocket_prior_masks',
           'scaffold_prior', 'arms_prior', 'ligand_atom_feature',
           'ligand_hybridization', 'ligand_file', 'arms_dict', 'arms_list',
           'scaffold_list', 'ligand_rdmol']


class BudgetExpired(Exception):
    pass


def expire(signum, frame):
    raise BudgetExpired('Per-process wall-time limit reached')


class Experiment:
    def __init__(self, device, outdir, manifest, base_checkpoint):
        self.device, self.outdir = device, outdir
        self.ckpt = torch.load(base_checkpoint, map_location='cpu', weights_only=False)
        self.cfg = self.ckpt['config']
        self.sampling = misc.load_config(ROOT / 'configs/sampling.yml')
        self.targets = json.loads(Path(manifest).read_text())['targets']
        self.protein_feat = trans.FeaturizeProteinAtom()
        self.ligand_feat = trans.FeaturizeLigandAtom(
            self.cfg.data.transform.ligand_atom_mode,
            prior_types=self.cfg.model.get('prior_types', False))
        self.indicator = trans.AddDecompIndicator(
            max_num_arms=self.cfg.data.transform.max_num_arms,
            global_prior_index=self.ligand_feat.ligand_feature_dim,
            add_ord_feat=self.cfg.data.transform.get('add_ord_feat', True))
        self.train_transform = Compose([
            trans.ComputeLigandAtomNoiseDist(version='ref_prior'), self.protein_feat,
            self.ligand_feat, self.indicator,
            trans.FeaturizeLigandBond(mode='fc', set_bond_type=True)])
        self.original = {}
        with lmdb.open(str(ROOT / 'data' / (DB_STEM + '.lmdb')),
                       subdir=False, readonly=True, create=False, lock=False) as env:
            with env.begin() as txn:
                for target in self.targets:
                    record = pickle.loads(txn.get(target['original_lmdb_key'].encode()))
                    assert record['src_ligand_filename'] == target['source_ligand']
                    self.original[target['target_id']] = record
        self.amber = trans.AmberFeaturizer.__new__(trans.AmberFeaturizer)
        bonds = torch.load(ROOT / 'data/bond_stat.pt', weights_only=False)
        angles = torch.load(ROOT / 'data/angle_stat.pt', weights_only=False)
        self.amber.bond_info = {k: [np.mean(v), np.std(v)] for k, v in bonds.items()}
        self.amber.angle_info = {k: [np.mean(np.radians(v)), np.std(np.radians(v))]
                                 for k, v in angles.items()}


def seed_biased_log_probs(log_probs, seed_types, strength):
    """A fixed seed prior is part of both the sampled and replayed policy."""
    if strength==0:
        return F.log_softmax(log_probs,-1)
    if not math.isfinite(strength) or strength<0 or seed_types is None or seed_types.shape!=log_probs.shape[:1]:
        raise ValueError('Invalid seed categorical prior')
    bias=F.one_hot(seed_types,num_classes=log_probs.shape[-1]).to(log_probs)*strength
    return F.log_softmax(log_probs+bias,-1)


def transition(model, context, state, timestep):
    pos, atom, bond = state
    batch, batch_bond = context['batch_ligand'], context['batch_ligand_bond']
    t = torch.full((int(batch.max()) + 1,), timestep, device=pos.device, dtype=torch.long)
    kwargs = dict(protein_pos=context['protein_pos'], protein_v=context['protein_v'],
                  batch_protein=context['batch_protein'], init_ligand_pos=pos,
                  init_ligand_v=atom, init_ligand_v_aux=context['ligand_v_aux'],
                  batch_ligand=batch, ligand_fc_bond_index=context['ligand_fc_bond_index'],
                  init_ligand_fc_bond_type=bond, time_step=t)
    if context.get('history_condition') is not None:
        kwargs['history_condition'] = context['history_condition']
    if context.get('objective_condition') is None:
        preds = model(**kwargs)
    else:
        from models.conditioning import guided_predictions
        preds = guided_predictions(model, kwargs, context['objective_condition'],
                                   context.get('guidance_scale', 1.0))
    assert model.model_mean_type == 'C0'
    mean = model.q_pos_posterior(preds['pred_ligand_pos'], pos, t, batch)
    std = (0.5 * extract(model.posterior_logvar, t, batch)).exp() * context['prior_stds'][context['ligand_decomp_batch']]
    atom_logp = model.atom_type_trans.q_v_posterior(
        F.log_softmax(preds['pred_ligand_v'], -1), index_to_log_onehot(atom, model.num_classes), t, batch)
    bond_logp = model.bond_type_trans.q_v_posterior(
        F.log_softmax(preds['pred_bond'], -1), index_to_log_onehot(bond, model.num_bond_classes), t, batch_bond)
    # Normalize explicitly, matching the categorical sampler's distribution.
    atom_logp=seed_biased_log_probs(atom_logp,context.get('seed_atom_types'),context.get('seed_atom_logit_bias',0.))
    bond_logp=seed_biased_log_probs(bond_logp,context.get('seed_bond_types'),context.get('seed_bond_logit_bias',0.))
    return mean, std, atom_logp, bond_logp


def joint_logp(params, action, drift, context):
    mean, std, atom_logp, bond_logp = params
    pos, atom, bond = action
    pos_lp = (-0.5 * ((pos - mean - drift) / std).square()
              - std.log() - 0.5 * math.log(2 * math.pi)).sum(-1)
    atom_lp = atom_logp.gather(1, atom[:, None]).squeeze(1)
    bond_lp = bond_logp.gather(1, bond[:, None]).squeeze(1)
    # The action is the full molecular transition: sums, not dimension averages.
    return (scatter_sum(pos_lp + atom_lp, context['batch_ligand'], dim=0)
            + scatter_sum(bond_lp, context['batch_ligand_bond'], dim=0))


def reference_kl(params, reference, context):
    mean, std, atom, bond = params
    ref_mean, _, ref_atom, ref_bond = reference
    pos_kl = (0.5 * ((mean - ref_mean) / std).square()).sum(-1)
    atom_kl = (atom.exp() * (atom - ref_atom)).sum(-1)
    bond_kl = (bond.exp() * (bond - ref_bond)).sum(-1)
    return (scatter_sum(pos_kl + atom_kl, context['batch_ligand'], dim=0)
            + scatter_sum(bond_kl, context['batch_ligand_bond'], dim=0))


def value_features(state, context, timestep):
    pos, atom, bond = state
    batch, bb = context['batch_ligand'], context['batch_ligand_bond']
    centered = pos - scatter_mean(pos, batch, dim=0)[batch]
    geometry = scatter_mean(centered.square(), batch, dim=0).sqrt() / 10
    composition = scatter_mean(F.one_hot(atom, 8).float(), batch, dim=0)
    bonds = scatter_mean(F.one_hot(bond, 5).float(), bb, dim=0)
    clock = torch.full((len(geometry), 1), timestep / 1000, device=pos.device)
    size = scatter_sum(torch.ones_like(atom).float(), batch, dim=0)[:, None] / 50
    return torch.cat([geometry, composition, bonds, clock, size], -1).detach()


@torch.no_grad()
def rollout(model, **context):
    protein, pos, offset = center_pos(context['protein_pos'], context['init_ligand_pos'],
                                     context['batch_protein'], context['batch_ligand'], mode='protein')
    context = dict(context, protein_pos=protein)
    state = (pos, context['init_ligand_v'], context['init_ligand_fc_bond_type'])
    traces = []
    started = time.monotonic()
    start_timestep = context.get('start_timestep', 999)
    if not isinstance(start_timestep, int) or not 1 <= start_timestep < 1000:
        raise ValueError('Denoising start timestep must be an integer in [1, 999]')
    for timestep in reversed(range(start_timestep + 1)):
        params = transition(model, context, state, timestep)
        mean, std, atom_lp, bond_lp = params
        atom = log_sample_categorical(atom_lp)
        bond = log_sample_categorical(bond_lp)
        t = torch.full((int(context['batch_ligand'].max()) + 1,), timestep,
                       dtype=torch.long, device=pos.device)
        guided = model.adjust_pos_with_energy_drift(
            state[0].detach().clone(), context['energy_drift_opt'], context['prior_centers'], atom,
            context['ligand_decomp_batch'], context['batch_ligand'], context['ligand_decomp_index'],
            offset, context['full_protein_pos'], context['full_batch_protein'], mean.clone(), timestep, t)
        drift = guided - mean
        # Preserve the original sampler's RNG consumption even at t=0.
        action = ((guided + int(timestep != 0) * std * torch.randn_like(mean)).detach(), atom, bond)
        if timestep in model.trace_steps:
            traces.append({'state': tuple(x.detach() for x in state), 'action': action,
                           't': timestep, 'drift': drift.detach(),
                           'old_logp': joint_logp(params, action, drift, context).detach(),
                           'features': value_features(state, context, timestep)})
        state = action
        if timestep % 250 == 0:
            print(f'denoising t={timestep}, elapsed={time.monotonic() - started:.1f}s', flush=True)
    if traces:
        model.saved_traces.append((context, traces))
    return {'pos': state[0] + offset[context['batch_ligand']], 'v': state[1], 'bond': state[2]}
