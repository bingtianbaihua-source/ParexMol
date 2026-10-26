"""ParexMol V18: frontier-guided molecular optimization with vector PPO.
"""
import argparse
import copy
import json
import logging
import signal
import time
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem

from models.decompdiff import DecompScorePosNet3D
from utils.diffusion_ppo import Experiment, BudgetExpired, expire, transition, joint_logp, reference_kl
from utils.runtime import ROOT, save_json
from utils import misc
from utils.molecular_objectives import ObjectiveSet, score_molecule
from utils.molecular_constraints import ConstraintSet
from utils.frontier_discovery import DiscoveryArchive, molecular_graph, exclusion_graphs
from utils.frontier_seeds import FrontierSeedPool, parent_metadata
from utils.frontier_parent_search import choose_parent,validate_parent_search,verify_search_trace,SELECTION as SEARCH_SELECTION
from utils.frontier_parent_batch import validate_batch_settings,propose_parent_batch,verify_parent_plan
from utils.frontier_rank import rank_feedback
from utils.molecular_size_policy import new_size_policy, validate_config as validate_size_config
from utils.morl import VectorCritic, ParetoArchive, LagrangeMultipliers, reference_directions, compare_frontiers, preco_direction

CONFIG = ROOT / 'configs/v18_5obj.json'


class MORLExperiment(Experiment):
    def __init__(self, device, outdir, protocol):
        self.protocol = protocol
        if protocol.get('stereochemistry', 'preserve') not in ('preserve', 'from_3d'):
            raise ValueError('stereochemistry must be preserve or from_3d')
        self.objectives = ObjectiveSet(protocol['objectives'])
        self.constraints = ConstraintSet(protocol.get('constraints', []))
        discovery = protocol.get('discovery')
        if protocol.get('size_policy'):
            validate_size_config(protocol['size_policy'])
            if not protocol.get('frontier_start'):
                raise ValueError('Learned size decisions require frontier parents')
        if protocol.get('frontier_start'):
            if discovery is None or 'lead_optimization' not in protocol:
                raise ValueError('Frontier restarts require discovery objectives and local diffusion')
            selection=protocol['frontier_start']['selection']
            if selection not in ('uniform_graph_then_stereo_then_conformer','hv_shared_coverage_graph_then_stereo_then_conformer',
                                 'hv_local_potential_graph_then_stereo_then_conformer',SEARCH_SELECTION):
                raise ValueError('Unsupported frontier parent selection')
            if protocol['frontier_start'] != dict(selection=selection,
                    conformer='scored_prepared', decomposition='brics_nearest_reference_regions'):
                raise ValueError('Unsupported frontier restart protocol')
        if 'parent_lookahead' in protocol:
            if protocol['parent_lookahead'].get('version')==2:
                validate_batch_settings(protocol['parent_lookahead'])
            else:validate_parent_search(protocol['parent_lookahead'])
            if (not protocol.get('frontier_start') or protocol['frontier_start']['selection']!=SEARCH_SELECTION
                    or protocol.get('size_policy',{}).get('version',0)<7):
                raise ValueError('Parent lookahead requires frontier restarts and a causal vector surrogate')
        elif protocol.get('frontier_start',{}).get('selection')==SEARCH_SELECTION:
            raise ValueError('Surrogate parent selection requires parent_lookahead settings')
        self.frontier_objectives = ObjectiveSet(discovery['objectives']) if discovery is not None else None
        if discovery is not None:
            if (not np.isfinite(discovery['reward_scale']) or discovery['reward_scale'] <= 0
                    or not 0 < discovery['preference_weight'] < 1):
                raise ValueError('Discovery needs a positive fixed reward scale and an interior preference weight')
            if 'frontier_hv_gain' in self.objectives.names:
                raise ValueError('frontier_hv_gain is reserved for the history-dependent reward channel')
            ranking = discovery.get('ranking')
            if ranking is not None:
                fields={'method','preference_weight'}
                if ranking['method']=='unclipped_soft_rank':
                    fields|={'temperature','transforms'}
                    if not 0<ranking['temperature']<=1:raise ValueError('Invalid rank temperature')
                elif ranking['method']!='per_objective_midrank':raise ValueError('Unsupported rank method')
                if set(ranking)!=fields or not 0 < ranking['preference_weight'] < 1-discovery['preference_weight']:
                    raise ValueError('Ranking requires a positive weight alongside HV and physical objectives')
        if bool(self.constraints) != ('lagrangian' in protocol):
            raise ValueError('PPO-Lagrangian requires both nonempty constraints and a lagrangian configuration')
        if self.constraints:
            if 'cost_coefficient' in protocol['training']:
                raise ValueError('Remove fixed cost_coefficient when using PPO-Lagrangian')
            LagrangeMultipliers(self.constraints.names, self.constraints.limits, **protocol['lagrangian'])
        super().__init__(device, outdir, ROOT / protocol['targets_manifest'], ROOT / protocol['initial_checkpoint'])

    @property
    def train_targets(self):
        return [target for target in self.targets if target.get('rl_split', 'train') == 'train']

    def new_discovery_archive(self):
        if self.frontier_objectives is None:
            return None
        return DiscoveryArchive(self.frontier_objectives.schema, self.constraints.schema,
                                history_scope=self.protocol['discovery'].get('history_scope', 'valid'),
                                rank_transforms=self.protocol['discovery'].get('ranking',{}).get('transforms'),
                                **self.protocol['discovery'].get('hypervolume', {}))

    def new_seed_pool(self, public):
        pool = FrontierSeedPool(self.frontier_objectives.schema,self.protocol.get('parent_selection'))
        source = None
        for pocket, rows in public.items():
            molecules = None
            if any(r['valid'] and r.get('feasible', True) and not r.get('pose_dir') for r in rows):
                if source is None:
                    source = torch.load(ROOT/self.protocol['baseline_dataset'], map_location='cpu', weights_only=False)['records']
                molecules = self.stereochemical_records([dict(mol=source[r['source_key']]['ligand_rdmol']) for r in rows])
            pool.update(pocket, rows, molecules, source='public', generation=0)
            pool.choose(pocket, 0)  # Fail explicitly if the public pool has no usable frontier seed.
        return pool

    @property
    def history_schema(self):
        archive = self.new_discovery_archive()
        if archive is None:
            return None
        return dict(version=1, frontier_schema=archive.schema, constraint_schema=archive.constraint_schema,
                    summary_fields=archive.summary_schema, history_scope=archive.history_scope)

    @property
    def history_dim(self):
        return 0 if self.frontier_objectives is None else len(self.history_schema['summary_fields'])

    @property
    def reward_schema(self):
        if self.frontier_objectives is None:
            return self.objectives.schema
        schema = self.objectives.schema + [dict(name='frontier_hv_gain', reference=0.,
            scale=self.protocol['discovery']['reward_scale'], allocation='graph_shared_hv',
            frontier_schema=self.frontier_objectives.schema)]
        if self.protocol['discovery'].get('ranking'):
            ranking=self.protocol['discovery']['ranking']
            schema += [dict(name='frontier_rank:'+spec['name'], reference=0., scale=1.,
                method=ranking['method'], ties=.5, novelty='unseen_feasible_graph',
                **({} if ranking['method']=='per_objective_midrank' else
                   dict(temperature=ranking['temperature'],transforms=ranking['transforms'])),
                allocation='graph_shared_rank', frontier_objective=spec)
                for spec in self.frontier_objectives.schema]
        return schema

    def reward_condition(self, preferences):
        preferences = np.asarray(preferences)
        if self.frontier_objectives is None:
            return preferences
        weight = self.protocol['discovery']['preference_weight']
        ranking = self.protocol['discovery'].get('ranking')
        if ranking:
            rank_weight = ranking['preference_weight']
            if len(self.objectives) != len(self.frontier_objectives):
                raise ValueError('Rank preferences require matching property and frontier dimensions')
            return np.concatenate([preferences*(1-weight-rank_weight),
                np.full(preferences.shape[:-1]+(1,),weight),preferences*rank_weight],axis=-1)
        return np.concatenate([preferences*(1-weight),
                               np.full(preferences.shape[:-1]+(1,), weight)], axis=-1)

    def sample(self, model, target, count, seed, trace_steps=0,
               objective_condition=None, guidance_scale=1.0, history_condition=None, frontier_parent=None,
               resized_parents=None, size_metadata=None, frontier_parents=None):
        extra = {} if history_condition is None else dict(history_condition=history_condition)
        if frontier_parent is not None:
            extra['frontier_parent'] = frontier_parent
        if resized_parents is not None:
            extra.update(resized_parents=resized_parents, size_metadata=size_metadata)
        if frontier_parents is not None:
            extra['frontier_parents']=frontier_parents
        from utils.local_diffusion import sample_lead
        records, traces = sample_lead(self, model, target, count, seed, trace_steps,
                                     objective_condition, guidance_scale, **self.protocol['lead_optimization'], **extra)
        return self.stereochemical_records(records), traces

    def stereochemical_records(self, records):
        """Match predictor input and stored molecular identity to sampled 3D geometry."""
        if self.protocol.get('stereochemistry', 'preserve') == 'preserve':
            return records
        result = []
        for record in records:
            if record.get('mol') is None:
                result.append(record)
                continue
            mol = Chem.Mol(record['mol'])
            try:
                Chem.SanitizeMol(mol)
            except (ValueError, RuntimeError):
                result.append(record)  # Keep the attempted chemical failure for ordinary scoring.
                continue
            if mol.GetNumConformers() != 1 or not np.isfinite(mol.GetConformer().GetPositions()).all():
                raise ValueError('Stereo from coordinates requires one finite 3D conformer')
            if not mol.GetConformer().Is3D():
                raise ValueError('Stereo from coordinates requires a 3D conformer')
            Chem.AssignStereochemistryFrom3D(mol, replaceExistingTags=True)
            result.append(dict(record, mol=mol, smiles=Chem.MolToSmiles(mol), stereochemistry='from_3d'))
        return result

    def model(self, checkpoint=None, conditioned=True, warm_start=False):
        state = self.ckpt if checkpoint is None else torch.load(checkpoint, map_location='cpu', weights_only=False)
        config = copy.deepcopy(self.cfg.model)
        config.objective_condition_dim = len(self.reward_schema) if conditioned else 0
        config.history_condition_dim = self.history_dim if conditioned else 0
        model = DecompScorePosNet3D(
            config, protein_atom_feature_dim=self.protein_feat.protein_feature_dim + self.indicator.protein_feature_dim,
            ligand_atom_feature_dim=self.ligand_feat.ligand_feature_dim + self.indicator.ligand_feature_dim,
            num_classes=self.ligand_feat.ligand_feature_dim, prior_atom_types=self.ligand_feat.atom_types_prob,
            prior_bond_types=self.ligand_feat.bond_types_prob).to(self.device)
        weights = dict(state['model'])
        if 'objective_schema' in state:
            if not warm_start:
                self.objectives.require_compatible(state['objective_schema'])
                if state.get('reward_schema', state['objective_schema']) != self.reward_schema:
                    raise ValueError('Checkpoint reward channels differ; use an explicit warm start')
                if state.get('history_schema') != self.history_schema:
                    raise ValueError('Checkpoint history summary differs; use an explicit warm start')
                # Sampling old actors under a new scoring protocol is allowed;
                # training resume separately checks constraints and dual state.
            else:
                old = {spec['name']: spec for spec in state.get('reward_schema', state['objective_schema'])}
                new = {spec['name']: spec for spec in self.reward_schema}
                if not old.keys() <= new.keys() or any(new[k] != v for k, v in old.items()):
                    raise ValueError('Warm-start expansion must preserve existing objective definitions')
                old_names = list(old)
                key = 'objective_emb.0.weight'
                first = torch.zeros_like(model.state_dict()[key], device='cpu')
                for i, name in enumerate(new):
                    if name in old:
                        first[:, i] = weights[key][:, old_names.index(name)]
                first[:, -1] = weights[key][:, -1]  # condition-presence bit
                weights[key] = first
                if state.get('history_schema') is not None and state['history_schema'] != self.history_schema:
                    raise ValueError('Warm start must preserve an existing history summary schema')
        elif any(k.startswith('objective_emb.') for k in weights):
            raise ValueError('Conditioned checkpoint has no MORL objective schema')
        missing, unexpected = model.load_state_dict(weights, strict=False)
        expected = {k for k in model.state_dict() if k.startswith('objective_emb.')} if 'objective_schema' not in state and conditioned else set()
        if conditioned and self.history_dim and state.get('history_schema') is None and (warm_start or 'objective_schema' not in state):
            expected |= {k for k in model.state_dict() if k.startswith('history_emb.')}
        if set(missing) != expected or unexpected:
            raise ValueError(f'Checkpoint mismatch: missing={missing}, unexpected={unexpected}')
        return model.eval()

    def score(self, records, target, folder):
        records = self.stereochemical_records(records)
        folder.mkdir(parents=True, exist_ok=True)
        if 'scoring_service' in self.protocol:
            from utils.scoring_client import score_request
            service = self.protocol['scoring_service']
            relative = folder.resolve().relative_to(self.outdir.resolve())
            stage = 'initial' if relative.parts[0] == 'training_baseline' else 'search'
            smiles = [Chem.MolToSmiles(r['mol']) if r.get('mol') is not None else None for r in records]
            measured = score_request(service['url'], service['arm'], str(relative), smiles, stage)
            if len(measured) != len(records):
                raise ValueError('Scoring service omitted generation attempts')
            results = []
            for i, row in enumerate(measured):
                if len(row['objectives']) != len(self.objectives) or len(row['constraint_costs']) != len(self.constraints):
                    raise ValueError('Scoring service schema differs from the experiment')
                results.append(dict(row, sample_id=i))
            save_json(folder/'scores.json', dict(objective_schema=self.objectives.schema,
                constraint_schema=self.constraints.schema, records=results,
                **({} if self.frontier_objectives is None else dict(frontier_schema=self.frontier_objectives.schema))))
            return results
        # Optional evaluator batching keeps expensive external models out of
        # the per-molecule loop; failures here are scoring faults, not bad molecules.
        context = dict(experiment=self, target=target, folder=folder)
        properties_to_score = [self.objectives, self.constraints]
        if self.frontier_objectives is not None:
            properties_to_score.append(self.frontier_objectives)
        for properties in properties_to_score:
            for item, evaluator in zip(properties.items, properties.evaluators):
                prepare_batch = getattr(evaluator, 'prepare_batch', None)
                if prepare_batch is not None:
                    prepare_batch(records, context, **item.parameters)
        results = [dict(score_molecule(r, target, folder, self, self.objectives,
                                      self.protocol['training']['invalid_cost'], self.constraints,
                                      self.frontier_objectives), sample_id=i)
                   for i, r in enumerate(records)]
        save_json(folder / 'scores.json', dict(objective_schema=self.objectives.schema,
                                               constraint_schema=self.constraints.schema, records=results,
                                               **({} if self.frontier_objectives is None else
                                                  dict(frontier_schema=self.frontier_objectives.schema))))
        return results


def baseline(exp):
    data = json.loads((exp.outdir / 'baseline.json').read_text())
    exp.objectives.require_compatible(data['objective_schema'])
    exp.constraints.require_compatible(data.get('constraint_schema', []))
    if getattr(exp, 'frontier_objectives', None) is not None:
        exp.frontier_objectives.require_compatible(data.get('frontier_schema', []))
    return data['pockets']


def prepare(exp):
    path = exp.outdir / 'baseline.json'
    if path.exists():
        baseline(exp)
        if json.loads((exp.outdir / 'protocol.json').read_text()) != exp.protocol:
            raise ValueError('Prepared run uses a different protocol; choose a new --outdir')
        return
    source = torch.load(ROOT / exp.protocol['baseline_dataset'], map_location='cpu', weights_only=False)
    pockets, invalid_public_samples = {}, []
    for target in exp.targets:
        if 'preference_keys' in target:
            selected = [(k, source['records'][k]) for k in target['preference_keys']]
            for _, record in selected:
                name = Path(record['src_ligand_filename'])
                receptor = str(name.parent / (name.name[:10] + '.pdb'))
                if receptor != target['receptor']:
                    raise ValueError('Public baseline record belongs to another receptor')
        else:
            selected = [(k, r) for k, r in source['records'].items() if r['src_ligand_filename'] == target['source_ligand']]
        if not selected:
            raise ValueError(f'No public baseline molecules for {target["source_ligand"]}')
        # Published candidates can themselves be chemically invalid. Retain
        # those failures, but never silently exclude a valid molecule because
        # receptor preparation, docking, or another evaluator failed.
        chemical_errors = {}
        for key, record in selected:
            try:
                mol = record['ligand_rdmol']
                if mol is None:
                    raise ValueError('Missing published molecule')
                mol = Chem.Mol(mol)
                Chem.SanitizeMol(mol)
                if '.' in Chem.MolToSmiles(mol):
                    raise ValueError('Disconnected published molecule')
            except (ValueError, RuntimeError) as exc:
                chemical_errors[key] = f'{type(exc).__name__}: {exc}'
        records = [dict(mol=r['ligand_rdmol']) for _, r in selected]
        scored = exp.score(records, target, exp.outdir / 'training_baseline' / str(target['target_id']))
        if exp.protocol.get('scoring_service'):
            # The shared benchmark defines validity after the same bounded
            # conformer/domain pipeline for every arm, including public data.
            # Infrastructure/docking/predictor exceptions are never returned as
            # these chemical-preparation failures by the scoring service.
            for (key, _), row in zip(selected, scored):
                if not row['valid'] and row.get('error') in (
                        'common_conformer_failed', 'outside_shared_6_to_60_heavy_atom_domain'):
                    chemical_errors[key] = 'Shared preparation: '+row['error']
        failed = {key for (key, _), row in zip(selected, scored) if not row['valid']}
        if len(scored) != len(selected) or failed != set(chemical_errors) or not any(r['valid'] for r in scored):
            raise ValueError('Baseline scoring failed; see training_baseline scores.json')
        invalid_public_samples.extend(dict(target_id=target['target_id'], source_key=key, error=error)
                                      for key, error in chemical_errors.items())
        pockets[str(target['target_id'])] = [dict(r, source_key=k) for r, (k, _) in zip(scored, selected)]
    save_json(path, dict(objective_schema=exp.objectives.schema, pockets=pockets,
                         constraint_schema=exp.constraints.schema,
                         invalid_public_samples=invalid_public_samples,
                         source=exp.protocol['baseline_dataset'],
                         **({} if exp.frontier_objectives is None else
                            dict(frontier_schema=exp.frontier_objectives.schema))))
    save_json(exp.outdir / 'protocol.json', exp.protocol)


def replay_microbatches(context, step, reference, size):
    """Slice whole molecular graphs, preserving actions, drift and prior indices."""
    if size < 1:
        raise ValueError('Update microbatch size must be positive')
    count = len(step['old_logp'])
    if size >= count:
        return [(slice(0, count), context, step, reference)]
    chunks = []
    for start in range(0, count, size):
        stop = min(start + size, count)
        part = slice(start, stop)
        masks = {key: (context[key] >= start) & (context[key] < stop)
                 for key in ('batch_protein', 'batch_ligand', 'batch_ligand_bond')}
        protein, atom, bond = (masks[k] for k in ('batch_protein', 'batch_ligand', 'batch_ligand_bond'))
        local_index = torch.full_like(context['batch_ligand'], -1)
        local_index[atom] = torch.arange(int(atom.sum()), device=atom.device)
        subcontext = dict(
            protein_pos=context['protein_pos'][protein], protein_v=context['protein_v'][protein],
            batch_protein=context['batch_protein'][protein] - start,
            batch_ligand=context['batch_ligand'][atom] - start,
            batch_ligand_bond=context['batch_ligand_bond'][bond] - start,
            ligand_v_aux=context['ligand_v_aux'][atom],
            ligand_fc_bond_index=local_index[context['ligand_fc_bond_index'][:, bond]],
            prior_stds=context['prior_stds'], ligand_decomp_batch=context['ligand_decomp_batch'][atom],
            objective_condition=context['objective_condition'][part],
            guidance_scale=context.get('guidance_scale', 1.0))
        if context.get('history_condition') is not None:
            subcontext['history_condition'] = context['history_condition'][part]
        for key,mask in [('atom',atom),('bond',bond)]:
            if f'seed_{key}_types' in context:
                subcontext[f'seed_{key}_types']=context[f'seed_{key}_types'][mask]
                subcontext[f'seed_{key}_logit_bias']=context[f'seed_{key}_logit_bias']
        def sliced_state(state):
            return state[0][atom], state[1][atom], state[2][bond]
        substep = dict(state=sliced_state(step['state']), action=sliced_state(step['action']),
                       drift=step['drift'][atom], t=step['t'], old_logp=step['old_logp'][part],
                       features=step['features'][part])
        subreference = (reference[0][atom], reference[1][atom], reference[2][atom], reference[3][bond])
        chunks.append((part, subcontext, substep, subreference))
    return chunks


def backward_microbatch(model, context, step, reference, direction, cost_advantage,
                       expected_logp, group_size, cfg, cost_multipliers=None):
    """Accumulate one contribution; release its graph before the next forward."""
    params = transition(model, context, step['state'], step['t'])
    logp = joint_logp(params, step['action'], step['drift'], context)
    replay_error = float((logp.detach() - expected_logp).abs().max())
    if replay_error > .03:
        raise ValueError(f'Microbatch direction/backward replay mismatch: {replay_error}')
    actor_loss = -(direction * logp).sum()  # direction already uses the full-group normalization
    ratio = (logp - step['old_logp']).clamp(-20, 20).exp()
    clipped_ratio = ratio.clamp(1-cfg['clip_epsilon'], 1+cfg['clip_epsilon'])
    if cost_multipliers is None:
        cost_loss = cfg['cost_coefficient'] * torch.maximum(
            ratio * cost_advantage, clipped_ratio * cost_advantage).sum() / group_size
    else:
        # B x K even for K=1. Conservative clipped surrogate for minimizing costs.
        cost_surrogate = torch.maximum(ratio[:, None] * cost_advantage,
                                       clipped_ratio[:, None] * cost_advantage).sum(0) / group_size
        cost_loss = (cost_multipliers * cost_surrogate).sum()
    kl = reference_kl(params, reference, context).sum() / group_size
    loss = actor_loss + cost_loss + cfg['reference_kl_coefficient'] * kl
    if not torch.isfinite(loss):
        raise FloatingPointError('Nonfinite vector PPO microbatch loss')
    loss.backward()
    return float(actor_loss.detach()), float(kl.detach()), replay_error


def update(model, reference, actor_optimizer, critic, critic_optimizer, traces, scores, objectives, cfg,
           microbatch_size=2, lagrangian=None, reward_reference=None):
    device = next(model.parameters()).device
    rewards = torch.tensor([r['objectives'] for r in scores], dtype=torch.float32, device=device)
    costs = torch.tensor([r['cost'] if lagrangian is None else r['constraint_costs'] for r in scores],
                         dtype=torch.float32, device=device)
    cost_multipliers = None
    if lagrangian is not None:
        if costs.shape != (len(scores), len(lagrangian.names)) or not torch.isfinite(costs).all() or torch.any(costs < 0):
            raise ValueError('PPO-Lagrangian needs one finite nonnegative cost vector per attempted molecule')
        cost_multipliers = torch.tensor(lagrangian.values.copy(), dtype=costs.dtype, device=device)
    ref_point = torch.tensor(objectives.reference if reward_reference is None else reward_reference,
                             dtype=torch.float32, device=device)
    items, offset = [], 0
    for context, steps in traces:
        n = len(steps[0]['old_logp'])
        preferences = context['objective_condition'].detach()
        if not torch.allclose(preferences, preferences[:1].expand_as(preferences)):
            raise ValueError('Collect one fixed preference per rollout group')
        for step in steps:
            with torch.no_grad():
                replay = joint_logp(transition(model, context, step['state'], step['t']), step['action'], step['drift'], context)
                error = float((replay - step['old_logp']).abs().max())
                if error > .03:
                    raise ValueError(f'Conditioned old-policy replay mismatch: {error}')
                # Reference remains the frozen, unconditioned pretrained model.
                ref_context = dict(context, objective_condition=None, history_condition=None)
                ref = tuple(p.detach() for p in transition(reference, ref_context, step['state'], step['t']))
                features = step['features']
                if context.get('history_condition') is not None:
                    features = torch.cat([features, context['history_condition'].to(features)], dim=-1)
                value, cost_value = critic(features, preferences)
                returns = rewards[offset:offset+n]
                cost_returns = costs[offset:offset+n]
                if cost_value.shape != cost_returns.shape:
                    raise ValueError('Cost critic output shape differs from cost returns')
                advantages = returns - value
                if cfg.get('group_relative_advantages',False):
                    from utils.morl import group_relative_advantages
                    advantages=group_relative_advantages(returns)
                cost_advantage = cost_returns - cost_value
            chunks = replay_microbatches(context, step, ref, microbatch_size)
            items.append((chunks, step, features, preferences, returns, cost_returns, advantages, cost_advantage))
        offset += n
    if offset != len(scores) or not items:
        raise ValueError('Trajectory/reward count mismatch')
    rows = []
    for _ in range(cfg['update_epochs']):
        for idx in torch.randperm(len(items)).tolist():
            chunks, step, features, preferences, returns, cost_returns, advantages, cost_advantage = items[idx]
            actor_optimizer.zero_grad(set_to_none=True)
            # Solve the multi-objective direction ONCE over all molecules at
            # the current policy. Do not retain the large molecular graphs.
            with torch.no_grad():
                logp = torch.cat([joint_logp(transition(model, c, s['state'], s['t']),
                                            s['action'], s['drift'], c) for _, c, s, _ in chunks])
                direction, info = preco_direction(logp - step['old_logp'], advantages, returns,
                    preferences[0], ref_point, cfg['clip_epsilon'], cfg['preco_strength'],cfg.get('preference_mix',0.))
            actor_loss, kl, replay_error = 0., 0., 0.
            for part, context, substep, ref in chunks:
                loss_part, kl_part, error = backward_microbatch(model, context, substep, ref,
                    direction[part], cost_advantage[part], logp[part], len(returns), cfg, cost_multipliers)
                actor_loss += loss_part
                kl += kl_part
                replay_error = max(replay_error, error)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['max_grad_norm'])
            if not torch.isfinite(norm):
                raise FloatingPointError('Nonfinite vector PPO actor update')
            actor_optimizer.step()
            values, cost_values = critic(features, preferences)
            critic_loss = (values - returns).square().mean() + (cost_values - cost_returns).square().mean()
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_norm = torch.nn.utils.clip_grad_norm_(critic.parameters(), cfg['max_grad_norm'])
            if not torch.isfinite(critic_loss) or not torch.isfinite(critic_norm):
                raise FloatingPointError('Nonfinite vector PPO critic update')
            critic_optimizer.step()
            rows.append(dict(actor_loss=actor_loss, critic_loss=float(critic_loss.detach()),
                             kl=kl, grad_norm=float(norm), microbatch_replay_error=replay_error,
                             **{k: v.cpu().tolist() for k, v in info.items()}))
    result = dict(optimizer_steps=len(rows), update_microbatch_size=microbatch_size,
                mean_objectives=rewards.mean(0).tolist(),
                mean_cost=float(costs.mean()), mean_weights=np.mean([r['weights'] for r in rows], axis=0).tolist(),
                last_update=rows[-1])
    if lagrangian is not None:
        # Fresh observations count once, irrespective of denoising steps/epochs/microbatches.
        result['lagrangian'] = lagrangian.update(costs.detach().cpu().numpy())
        result.pop('mean_cost')  # Different constraint units must not be averaged together.
    return result


def train(exp, args):
    cfg = exp.protocol['training']
    public = baseline(exp)
    targets = exp.train_targets
    if not targets:
        raise ValueError('No RL training pockets in the manifest')
    train_ids = {str(target['target_id']) for target in targets}
    checkpoint = exp.outdir / 'final.pt'
    if checkpoint.exists() and not args.resume:
        raise ValueError('Run already has a checkpoint; use --resume or a new --outdir')
    if args.resume and args.warm_start:
        raise ValueError('--resume and --warm-start are mutually exclusive')
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False) if args.resume else None
    if saved:
        exp.objectives.require_compatible(saved['objective_schema'])
        exp.constraints.require_compatible(saved.get('constraint_schema', []))
        if saved['protocol'] != exp.protocol:
            raise ValueError('Resume requires the same protocol')
        if 'targets' in saved and saved['targets'] != exp.targets:
            raise ValueError('Resume requires the same target manifest and RL split')
    misc.seed_all(args.seed)
    model = exp.model(checkpoint if args.resume else args.warm_start, warm_start=bool(args.warm_start))
    reference = exp.model(conditioned=False).requires_grad_(False)
    lagrangian = (LagrangeMultipliers(exp.constraints.names, exp.constraints.limits, **exp.protocol['lagrangian'])
                  if exp.constraints else None)
    critic = VectorCritic(18+exp.history_dim, len(exp.reward_schema), cfg['critic_hidden'],
                          cost_dim=len(exp.constraints) if exp.constraints else None).to(exp.device)
    head = list(model.objective_emb.parameters())
    if exp.history_dim:
        head += list(model.history_emb.parameters())
    head_ids = {id(p) for p in head}
    actor_optimizer = torch.optim.Adam([
        dict(params=[p for p in model.parameters() if id(p) not in head_ids], lr=cfg['actor_lr']),
        dict(params=head, lr=cfg['condition_lr'])], betas=(.95, .999))
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=cfg['critic_lr'])
    archive = ParetoArchive(exp.objectives.schema, reference_directions(len(exp.objectives), **exp.protocol['directions']),
                            exp.constraints.schema)
    discovery = exp.new_discovery_archive()
    seed_pool = exp.new_seed_pool(public) if exp.protocol.get('frontier_start') else None
    size_policy = new_size_policy(exp, saved)
    if size_policy is not None and size_policy.cfg['version']>=7 and saved is None:
        size_policy.initialize_surrogate({p:r for p,r in public.items() if p in train_ids},
            exp.frontier_objectives.reference,exp.protocol['discovery']['hypervolume']['integration_upper'],
            exp.protocol['discovery'].get('excluded_smiles',[]))
    for pocket, rows in public.items():
        if pocket in train_ids:
            archive.update(pocket, rows)
            if discovery is not None:
                discovery.initialize(pocket, rows, exp.protocol['discovery'].get('excluded_smiles', []))
    rounds, steps = 0, 0
    rng = np.random.default_rng(args.seed)
    if saved:
        if lagrangian is not None:
            lagrangian.load_state_dict(saved['lagrangian'])
        critic.load_state_dict(saved['critic'])
        actor_optimizer.load_state_dict(saved['optimizer'])
        critic_optimizer.load_state_dict(saved['critic_optimizer'])
        archive = ParetoArchive.from_state_dict(saved['archive'])
        if discovery is not None:
            discovery = DiscoveryArchive.from_state_dict(saved['discovery_archive'])
        if seed_pool is not None:
            if 'frontier_seed_pool' not in saved:
                raise ValueError('Frontier restart resume requires a saved molecular seed pool; start a new run')
            seed_pool = FrontierSeedPool.from_state_dict(saved['frontier_seed_pool'])
        exp.constraints.require_compatible(archive.constraint_schema)
        rounds, steps = saved['rounds'], saved['optimizer_steps']
        rng.bit_generator.state = saved['direction_rng']
        args.seed = saved['seed']

    def save():
        config = copy.deepcopy(exp.cfg)
        config.model.objective_condition_dim = len(exp.reward_schema)
        config.model.history_condition_dim = exp.history_dim
        state = dict(model=model.state_dict(), config=config, objective_schema=exp.objectives.schema,
                     constraint_schema=exp.constraints.schema,
                     algorithm='molecular_preco_ppo_lagrangian' if lagrangian is not None else 'molecular_preco_ppo',
                     lagrangian=None if lagrangian is None else lagrangian.state_dict(),
                     protocol=exp.protocol, critic=critic.state_dict(),
                     targets=exp.targets,
                     optimizer=actor_optimizer.state_dict(), critic_optimizer=critic_optimizer.state_dict(),
                     archive=archive.state_dict(), direction_rng=rng.bit_generator.state,
                     execution=dict(update_microbatch_size=args.update_microbatch_size),
                     rounds=rounds, optimizer_steps=steps, seed=args.seed)
        if discovery is not None:
            state.update(reward_schema=exp.reward_schema, history_schema=exp.history_schema,
                         discovery_archive=discovery.state_dict(),
                         algorithm='molecular_frontier_preco_ppo_lagrangian' if lagrangian is not None
                                   else 'molecular_frontier_preco_ppo')
        if seed_pool is not None:
            state['frontier_seed_pool'] = seed_pool.state_dict()
        if size_policy is not None:
            state['size_policy'] = size_policy.state_dict()
            state['algorithm'] += '_learned_size'
        temporary = checkpoint.with_suffix('.tmp')
        torch.save(state, temporary)
        temporary.replace(checkpoint)
        save_json(exp.outdir / 'archive.json', archive.state_dict())

    save()
    start = time.monotonic()
    signal.signal(signal.SIGALRM, expire)
    signal.signal(signal.SIGTERM, expire)
    signal.setitimer(signal.ITIMER_REAL, args.max_seconds)
    status = 'completed'
    try:
        for _ in range(args.max_rounds):
            target = targets[rounds % len(targets)]
            preference, choice = archive.choose(target['target_id'], rng)
            condition = torch.tensor(exp.reward_condition(preference), device=exp.device,
                                      dtype=torch.float32).repeat(cfg['group_size'], 1)
            history_kwargs = {}
            if discovery is not None:
                history_kwargs['history_condition'] = torch.tensor(discovery.summary(target['target_id']),
                    device=exp.device, dtype=torch.float32).repeat(cfg['group_size'], 1)
            folder = exp.outdir / 'rollouts' / f'{rounds:06d}'
            folder.mkdir(parents=True, exist_ok=True)
            parent,parents,size_trace = None,None,None
            if seed_pool is not None:
                if exp.protocol.get('parent_lookahead',{}).get('version')==2:
                    parents,resized,size_trace,search=propose_parent_batch(seed_pool,size_policy,target['target_id'],
                        history_kwargs['history_condition'][0].cpu().tolist(),condition,args.seed*10000+rounds,
                        exp.protocol['parent_lookahead'],discovery.seen.get(str(target['target_id']),set()))
                    parent=parents[0]
                    history_kwargs['frontier_parents']=parents
                    save_json(folder/'parents.json',[parent_metadata(p) for p in parents])
                    save_json(folder/'parent_search.json',search)
                elif exp.protocol.get('parent_lookahead'):
                    parent,search=choose_parent(seed_pool,target['target_id'],args.seed*10000+rounds,
                        exp.protocol['parent_lookahead'],size_policy.cfg,size_policy.surrogate,
                        discovery.seen.get(str(target['target_id']),set())|size_policy.proposed_graphs,cfg['group_size'])
                    save_json(folder/'parent_search.json',search)
                else:
                    parent = seed_pool.choose(target['target_id'], args.seed*10000+rounds)
                history_kwargs['frontier_parent'] = parent
                save_json(folder/'parent.json', parent_metadata(parent))
            if size_policy is not None:
                if size_trace is None:
                    if size_policy.cfg['version'] in (8,9):
                        size_policy.set_donors(target['target_id'],seed_pool.pockets[str(target['target_id'])])
                    resized, size_trace = size_policy.propose(parent,
                        history_kwargs['history_condition'][0].cpu().tolist(), condition, args.seed*10000+rounds,
                        excluded_graphs=discovery.seen.get(str(target['target_id']),set()))
                history_kwargs.update(resized_parents=resized, size_metadata=size_trace['metadata'])
                torch.save(dict(size_trace, resized_parents=resized), folder/'size_actions.pt')
                save_json(folder/'size_actions.json', size_trace['metadata'])
            if str(exp.device).startswith('cuda'):
                torch.cuda.reset_peak_memory_stats(exp.device)
            records, traces = exp.sample(model, target, cfg['group_size'], args.seed * 10000 + rounds,
                                         trace_steps=cfg['sampled_transitions'], objective_condition=condition,
                                         **history_kwargs)
            if len(records) != cfg['group_size']:
                raise ValueError('Sampler did not return the configured number of attempts')
            torch.save(records, folder / 'molecules.pt')
            scores = exp.score(records, target, folder)
            training_scores, reward_reference = scores, None
            discovery_result = None
            if discovery is not None:
                credits, discovery_result = discovery.credit(target['target_id'], scores, args.seed*10000+rounds)
                training_scores = [dict(row, objectives=row['objectives']+[float(value)/exp.protocol['discovery']['reward_scale']])
                                   for row, value in zip(scores, credits)]
                reward_reference = np.concatenate([exp.objectives.reference, [0.]])
                if exp.protocol['discovery'].get('ranking'):
                    ranks, rank_meta = rank_feedback(discovery, target['target_id'], scores,
                        exp.protocol['discovery']['ranking'].get('temperature'))
                    training_scores = [dict(row, objectives=row['objectives']+rank.tolist())
                                       for row, rank in zip(training_scores, ranks)]
                    reward_reference = np.concatenate([reward_reference, np.zeros(ranks.shape[1])])
                    discovery_result['ranking'] = rank_meta
                save_json(folder/'discovery.json', dict(discovery_result, reward_schema=exp.reward_schema,
                    history_schema=exp.history_schema, reward_vectors=[row['objectives'] for row in training_scores]))
            size_result = None
            if size_policy is not None:
                # Both hierarchy levels use the same PRE-update multipliers.
                # The existing diffusion update advances the dual only once.
                size_result = size_policy.update(size_trace, training_scores, cfg,
                    exp.objectives.reference if reward_reference is None else reward_reference,
                    None if lagrangian is None else lagrangian.values.copy())
            result = update(model, reference, actor_optimizer, critic, critic_optimizer,
                            traces, training_scores, exp.objectives, cfg, microbatch_size=args.update_microbatch_size,
                            lagrangian=lagrangian, reward_reference=reward_reference)
            if size_result is not None:
                result['size_policy'] = size_result
            if discovery is not None:
                result['mean_reward_vector'] = result.pop('mean_objectives')
                result['discovery'] = discovery_result
                discovery.update(target['target_id'], scores)
            if size_policy is not None and size_policy.cfg['version']>=7:
                size_policy.observe_surrogate(target['target_id'],scores)
            if exp.constraints:
                result.update(exp.constraints.summarize(scores))
            if str(exp.device).startswith('cuda'):
                result.update(cuda_peak_allocated_gib=torch.cuda.max_memory_allocated(exp.device) / 2**30,
                              cuda_peak_reserved_gib=torch.cuda.max_memory_reserved(exp.device) / 2**30)
            archive.update(target['target_id'], scores)
            if seed_pool is not None:
                seed_pool.update(target['target_id'], scores, records, generation=rounds+1, source='generated')
            rounds += 1
            steps += result['optimizer_steps']
            row = dict(round=rounds, target_id=target['target_id'], preference=preference.tolist(),
                       direction=choice, valid=sum(r['valid'] for r in scores), attempts=len(scores),
                       elapsed_seconds=time.monotonic()-start, **result)
            if parent is not None:
                row['parent'] = parent_metadata(parent)
            if parents is not None:
                row['parents']=[parent_metadata(p) for p in parents]
            with (exp.outdir / 'training.jsonl').open('a') as stream:
                stream.write(json.dumps(row, allow_nan=False) + '\n')
            save()  # resume always starts at a complete on-policy round
            model.saved_traces = []
            print(json.dumps(row, allow_nan=False), flush=True)
    except BudgetExpired:
        status = 'time_limit; checkpoint retains last complete round'
    except Exception:
        status = 'failed; checkpoint retains last complete round'
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        save_json(exp.outdir / 'run.json', dict(status=status, seconds=time.monotonic()-start,
                  completed_rounds=rounds, optimizer_steps=steps, objective_names=exp.objectives.names))
    if steps == 0:
        raise RuntimeError('No full PPO update completed; increase the per-run time allowance')


def sample(exp, args):
    base = args.base
    checkpoint = args.checkpoint or exp.outdir / 'final.pt'
    model = exp.model(None if base else checkpoint, conditioned=not base)
    size_policy = (new_size_policy(exp, torch.load(checkpoint, map_location='cpu', weights_only=False))
                   if not base and exp.protocol.get('size_policy') else None)
    dirs = reference_directions(len(exp.objectives), **exp.protocol['directions'])
    checkpoint_path = ROOT / exp.protocol['initial_checkpoint'] if base else checkpoint
    stat = checkpoint_path.stat()
    request = dict(objective_schema=exp.objectives.schema, protocol=exp.protocol,
                   attempts=args.num_samples, seed=args.seed,
                   checkpoint='base' if base else str(checkpoint),
                   checkpoint_size=stat.st_size, checkpoint_mtime_ns=stat.st_mtime_ns)
    public = baseline(exp) if exp.frontier_objectives is not None else None
    if public is not None:
        request.update(reward_schema=exp.reward_schema, history_schema=exp.history_schema,
                       history_source='public_baseline_then_evaluation_batches')
    if exp.protocol.get('frontier_start') and not base:
        request['frontier_start'] = exp.protocol['frontier_start']
    for target in exp.targets:
        discovery = exp.new_discovery_archive()
        if size_policy is not None and size_policy.cfg['version']>=7:
            size_policy.initialize_surrogate({str(target['target_id']):public[str(target['target_id'])]},
                exp.frontier_objectives.reference,exp.protocol['discovery']['hypervolume']['integration_upper'],
                exp.protocol['discovery'].get('excluded_smiles',[]))
        seed_pool = (exp.new_seed_pool({str(target['target_id']): public[str(target['target_id'])]})
                     if exp.protocol.get('frontier_start') and not base else None)
        if discovery is not None:
            discovery.initialize(target['target_id'], public[str(target['target_id'])],
                                 exp.protocol['discovery'].get('excluded_smiles', []))
        folder = exp.outdir / 'evaluation' / args.tag / str(target['target_id'])
        saved_request = folder / 'sampling_request.json'
        target_request = dict(request, target=target)
        if saved_request.exists():
            if not args.resume or json.loads(saved_request.read_text()) != target_request:
                raise ValueError('Evaluation already started or settings changed; use matching --resume or a new --tag')
            if (folder / 'molecules.pt').exists() and (folder / 'sampling.json').exists():
                continue
        elif (folder / 'molecules.pt').exists():
            raise ValueError('Legacy evaluation already exists; choose a new --tag')
        folder.mkdir(parents=True, exist_ok=True)
        save_json(saved_request, target_request)
        (folder / 'batches').mkdir(exist_ok=True)
        records, conditions, sampling_seeds, online_scores, discovery_batches = [], [], [], [], []
        for offset in range(0, args.num_samples, exp.protocol['evaluation']['batch_size']):
            count = min(exp.protocol['evaluation']['batch_size'], args.num_samples-offset)
            prefs = dirs[np.arange(offset, offset+count) % len(dirs)]
            reward_prefs = exp.reward_condition(prefs)
            condition = None if base else torch.tensor(reward_prefs, dtype=torch.float32, device=exp.device)
            history_kwargs, history_summary = {}, None
            if discovery is not None:
                history_summary = discovery.summary(target['target_id']).tolist()
                if not base:
                    history_kwargs['history_condition'] = torch.tensor(history_summary, device=exp.device,
                        dtype=torch.float32).repeat(count, 1)
            # Fresh evaluation noise, separate from the online training rounds.
            sample_seed = 1_000_000_000 + args.seed*10000 + target['target_id']*1000 + offset
            parent,parent_search,parents,prepared_trace=None,None,None,None
            if seed_pool is not None:
                if exp.protocol.get('parent_lookahead',{}).get('version')==2:
                    parents,resized,prepared_trace,parent_search=propose_parent_batch(seed_pool,size_policy,
                        target['target_id'],history_summary,condition,sample_seed,exp.protocol['parent_lookahead'],
                        discovery.seen.get(str(target['target_id']),set()))
                    parent=parents[0];history_kwargs['frontier_parents']=parents
                elif exp.protocol.get('parent_lookahead'):
                    parent,parent_search=choose_parent(seed_pool,target['target_id'],sample_seed,
                        exp.protocol['parent_lookahead'],size_policy.cfg,size_policy.surrogate,
                        discovery.seen.get(str(target['target_id']),set())|size_policy.proposed_graphs,count)
                else:parent=seed_pool.choose(target['target_id'],sample_seed)
            if parent is not None:
                history_kwargs['frontier_parent'] = parent
            batch_file = folder / 'batches' / f'{offset:06d}.pt'
            if batch_file.exists():
                saved = torch.load(batch_file, map_location='cpu', weights_only=False)
                if saved['seed'] != sample_seed or saved['offset'] != offset:
                    raise ValueError('Saved evaluation batch differs from requested sampling')
                if parent is not None and saved.get('parent') != parent_metadata(parent):
                    raise ValueError('Saved evaluation frontier parent differs from causal history')
                if parents is not None:
                    if saved.get('parents')!=[parent_metadata(p) for p in parents]:
                        raise ValueError('Saved evaluation parent batch differs from causal history')
                    verify_parent_plan(saved['parent_search'],parent_search)
                    torch.testing.assert_close(saved['size_actions']['candidate_choices'],prepared_trace['candidate_choices'],atol=0,rtol=0)
                    torch.testing.assert_close(saved['size_actions']['old_logp'],prepared_trace['old_logp'],atol=1e-5,rtol=0)
                elif parent_search is not None:verify_search_trace(saved['parent_search'],parent_search)
                batch = saved['records']
                if size_policy is not None and size_policy.cfg['version']>=5:
                    size_policy.proposed_graphs.update(
                        row['proposal_graph'] for row in saved['size_actions']['metadata'])
                if discovery is not None:
                    if saved['history_summary'] != history_summary:
                        raise ValueError('Evaluation history differs from the saved pre-generation summary')
                    batch_scores, contribution = saved['scores'], saved['discovery']
            else:
                size_trace = prepared_trace
                if size_policy is not None:
                    if size_trace is None:
                        if size_policy.cfg['version'] in (8,9):
                            size_policy.set_donors(target['target_id'],seed_pool.pockets[str(target['target_id'])])
                        resized, size_trace = size_policy.propose(parent, history_summary, condition, sample_seed,
                            excluded_graphs=discovery.seen.get(str(target['target_id']),set()))
                    history_kwargs.update(resized_parents=resized, size_metadata=size_trace['metadata'])
                batch, _ = exp.sample(model, target, count, sample_seed,
                                      objective_condition=condition, **history_kwargs)
                if len(batch) != count:
                    raise ValueError('Sampler did not return the configured evaluation attempts')
                temporary = batch_file.with_suffix('.tmp')
                batch_state = dict(seed=sample_seed, offset=offset, records=batch)
                if size_trace is not None:
                    batch_state['size_actions'] = size_trace
                if parent is not None:
                    batch_state['parent'] = parent_metadata(parent)
                if parents is not None:batch_state['parents']=[parent_metadata(p) for p in parents]
                if parent_search is not None:batch_state['parent_search']=parent_search
                if discovery is not None:
                    batch_scores = exp.score(batch, target, folder/'batches'/f'{offset:06d}_scores')
                    _, contribution = discovery.credit(target['target_id'], batch_scores, sample_seed)
                    batch_state.update(history_summary=history_summary, scores=batch_scores, discovery=contribution)
                torch.save(batch_state, temporary)
                temporary.replace(batch_file)
            if len(batch) != count:
                raise ValueError('Sampler did not return the configured evaluation attempts')
            records.extend(batch)
            conditions.extend(reward_prefs.tolist())
            sampling_seeds.append(sample_seed)
            if discovery is not None:
                discovery.update(target['target_id'], batch_scores)
                online_scores.extend(dict(row, sample_id=offset+i) for i, row in enumerate(batch_scores))
                discovery_batches.append(dict(contribution, offset=offset, seed=sample_seed))
            if size_policy is not None and size_policy.cfg['version']>=7:
                size_policy.observe_surrogate(target['target_id'],batch_scores)
            if seed_pool is not None:
                seed_pool.update(target['target_id'], batch_scores, batch, generation=offset+count, source='generated')
            stop_after = getattr(args, 'stop_after_attempts', None)
            if stop_after is not None and len(records) >= stop_after:
                break
        if len(records) < args.num_samples:
            save_json(folder/'progress.json', dict(attempts=len(records), requested=args.num_samples,
                status='checkpointed_prefix; resume with the same total --num-samples'))
            continue
        temporary = folder / 'molecules.tmp'
        torch.save(records, temporary)
        temporary.replace(folder / 'molecules.pt')
        if discovery is not None:
            save_json(folder/'scores.json', dict(objective_schema=exp.objectives.schema,
                constraint_schema=exp.constraints.schema, frontier_schema=exp.frontier_objectives.schema,
                records=online_scores))
            save_json(folder/'discovery.json', dict(history_schema=exp.history_schema,
                reward_schema=exp.reward_schema, history_source=request['history_source'],
                batches=discovery_batches, final_archive=discovery.state_dict()))
        save_json(folder / 'sampling.json', dict(objective_schema=exp.objectives.schema, attempts=len(records),
                  constraint_schema=exp.constraints.schema,
                  seed=args.seed, sampling_seeds=sampling_seeds,
                  checkpoint='base' if base else str(checkpoint), preferences=conditions,
                  **({} if discovery is None else dict(reward_schema=exp.reward_schema,
                      history_schema=exp.history_schema, history_source=request['history_source']))))


def score(exp, args):
    for target in exp.targets:
        folder = exp.outdir / 'evaluation' / args.tag / str(target['target_id'])
        meta = json.loads((folder / 'sampling.json').read_text())
        exp.objectives.require_compatible(meta['objective_schema'])
        exp.constraints.require_compatible(meta.get('constraint_schema', []))
        records = torch.load(folder / 'molecules.pt', map_location='cpu', weights_only=False)
        if len(records) != meta['attempts']:
            raise ValueError('Evaluation count mismatch')
        cached = folder / 'scores.json'
        if args.resume and cached.exists():
            data = json.loads(cached.read_text())
            exp.objectives.require_compatible(data['objective_schema'])
            exp.constraints.require_compatible(data.get('constraint_schema', []))
            if len(data['records']) != len(records):
                raise ValueError('Cached evaluation score count mismatch')
            continue
        exp.score(records, target, folder)


def report(exp, args):
    public, pockets = baseline(exp), {}
    for target in exp.targets:
        p = str(target['target_id'])
        data = json.loads((exp.outdir / 'evaluation' / args.tag / p / 'scores.json').read_text())
        exp.objectives.require_compatible(data['objective_schema'])
        exp.constraints.require_compatible(data.get('constraint_schema', []))
        rows = data['records']
        valid = [r['objectives'] for r in rows if r['valid']]
        pockets[p] = dict(compare_frontiers([r['objectives'] for r in public[p] if r['valid']], valid, exp.objectives.reference,
                                           **exp.protocol['hypervolume']), valid=len(valid), attempts=len(rows))
        if exp.constraints:
            pockets[p].update(exp.constraints.summarize(rows))
            pockets[p]['feasible_hypervolume'] = compare_frontiers(
                [r['objectives'] for r in public[p] if r['valid'] and r['feasible']],
                [r['objectives'] for r in rows if r['valid'] and r['feasible']],
                exp.objectives.reference, **exp.protocol['hypervolume'])
            pockets[p]['baseline_constraints'] = exp.constraints.summarize(public[p])
        if exp.frontier_objectives is not None:
            archive = exp.new_discovery_archive()
            excluded = set()
            if archive.history_scope == 'public_plus_feasible_novel':
                initial_smiles = exp.protocol['discovery'].get('excluded_smiles', [])
                excluded = exclusion_graphs(initial_smiles)
                excluded.update(molecular_graph(r['smiles']) for r in public[p] if r['valid'])
            pockets[p]['discovery_frontier'] = compare_frontiers(
                [r['frontier_objectives'] for r in public[p] if r['valid'] and
                    (archive.history_scope == 'public_plus_feasible_novel' or archive.include_in_history(r))],
                [r['frontier_objectives'] for r in rows if archive.eligible(r) and
                    molecular_graph(r['smiles']) not in excluded],
                exp.frontier_objectives.reference, **exp.protocol['discovery'].get('hypervolume', {}))
    result = dict(objective_schema=exp.objectives.schema, pockets=pockets,
                  constraint_schema=exp.constraints.schema,
                  mean_hv_gain=float(np.mean([p['union_hv_gain'] for p in pockets.values()])),
                  scope='Fresh generated molecules versus the fixed public training baseline, per pocket. Training archive is excluded.')
    if exp.frontier_objectives is not None:
        result.update(frontier_schema=exp.frontier_objectives.schema, reward_schema=exp.reward_schema,
                      mean_discovery_hv_gain=float(np.mean([p['discovery_frontier']['union_hv_gain']
                                                          for p in pockets.values()])))
    if exp.constraints:
        result['mean_feasible_generated_hv'] = float(np.mean([p['feasible_hypervolume']['generated_hv'] for p in pockets.values()]))
        result['mean_feasible_hv_gain'] = float(np.mean([p['feasible_hypervolume']['union_hv_gain'] for p in pockets.values()]))
    folder = exp.outdir / 'evaluation' / args.tag
    save_json(folder / 'report.json', result)
    lines = [f'多目标 PPO 评估：{args.tag}', '',
             '目标：' + '、'.join(exp.objectives.names), '',
             '| 口袋 | 有效/尝试 | 训练基线 HV | 并集 HV | ΔHV | 计算方法 |',
             '|---|---:|---:|---:|---:|---|']
    for pocket, row in pockets.items():
        lines.append(f'| {pocket} | {row["valid"]}/{row["attempts"]} | {row["training_hv"]:.6f} | '
                     f'{row["union_hv"]:.6f} | {row["union_hv_gain"]:.6f} | {row["hv"]["method"]} |')
    lines += ['', f'对口袋等权平均的 ΔHV：{result["mean_hv_gain"]:.6f}。', '',
              '比较使用独立生成样本与固定公开训练基线，不把训练档案计入评估生成集。',
              '五维及以上使用 Sobol 超体积估计，具体积分设置见 report.json。',
              '此单模型报告不能判断是否优于基础模型或标量 PPO，需使用相同生成与评分预算进行对照。']
    if exp.constraints:
        lines += ['', '上述表格包含所有有效分子。约束优化以以下可行集合指标为准，公开基线也应用相同约束。', '',
                  '| 口袋 | 可行/尝试 | 可行基线 HV | 可行生成 HV | 可行 ΔHV |',
                  '|---|---:|---:|---:|---:|']
        for pocket, row in pockets.items():
            h = row['feasible_hypervolume']
            lines.append(f'| {pocket} | {row["feasible"]}/{row["attempts"]} | {h["training_hv"]:.6f} | '
                         f'{h["generated_hv"]:.6f} | {h["union_hv_gain"]:.6f} |')
        lines += ['', f'平均可行生成 HV：{result["mean_feasible_generated_hv"]:.6f}；'
                     f'平均可行 ΔHV：{result["mean_feasible_hv_gain"]:.6f}。',
                  '约束均值与违反率的分母包含所有生成尝试；详见 report.json。空可行基线的 HV 为 0。']
    (folder / 'report.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'train', 'sample', 'score', 'report'])
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--outdir', type=Path)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--seed', type=int, default=7101)
    parser.add_argument('--max-seconds', type=float, default=900)
    parser.add_argument('--max-rounds', type=int, default=16)
    parser.add_argument('--update-microbatch-size', type=int, default=2,
                        help='Molecules per backward pass; the complete PPO group and direction stay unchanged')
    parser.add_argument('--num-samples', type=int, default=16)
    parser.add_argument('--stop-after-attempts', type=int,
                        help='Checkpoint a sampling prefix without changing the fixed total sampling request')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--warm-start', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--base', action='store_true')
    parser.add_argument('--tag', default='v18')
    args = parser.parse_args()
    if args.update_microbatch_size < 1:
        parser.error('--update-microbatch-size must be positive')
    torch.set_num_threads(2)
    logging.basicConfig(level=logging.WARNING)
    protocol = json.loads(args.config.read_text())
    outdir = args.outdir or ROOT / protocol['outdir']
    outdir.mkdir(parents=True, exist_ok=True)
    exp = MORLExperiment(args.device, outdir, protocol)
    if args.stage == 'prepare':
        prepare(exp)
    else:
        frozen = json.loads((outdir / 'protocol.json').read_text())
        if frozen != protocol:
            raise ValueError('Run protocol differs; prepare a new output directory')
        globals()[args.stage](exp, args)


if __name__ == '__main__':
    main()
