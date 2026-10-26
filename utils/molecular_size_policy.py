"""Hierarchical graph PPO: one categorical decision before each diffusion trajectory.

Version 1 learns -1/0/+1 size decisions. Version 2 learns nine structure-edit
families, including multiatom changes, substitutions and ring edits. Concrete
legal edits are sampled uniformly within each family without property screening.
Version 5 also excludes previously proposed or scored graphs and samples each
batch without replacement. PPO replays each conditional chemical/novelty mask
and the old probability; uniform within-family choices cancel in the ratio.
"""
import copy

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from torch import nn

from utils.morl import VectorCritic, preco_direction

DEFAULT_SIZE_POLICY = dict(version=1, deltas=[-1, 0, 1], min_atoms=6, max_atoms=60,
    operator='terminal_delete_or_saturated_contract__carbon_attach', hidden=64,
    actor_lr=.001, critic_lr=.001, entropy_coefficient=.01)


def validate_config(cfg):
    if cfg.get('version') in (2,3,4,5,6,7,8,9):
        from utils.molecular_structure_search import validate_structure_config
        return validate_structure_config(cfg)
    if (set(cfg) != set(DEFAULT_SIZE_POLICY) or cfg['version'] != 1
            or cfg['deltas'] != [-1, 0, 1] or cfg['operator'] != DEFAULT_SIZE_POLICY['operator']
            or not 2 <= cfg['min_atoms'] < cfg['max_atoms'] or cfg['hidden'] < 1
            or cfg['actor_lr'] <= 0 or cfg['critic_lr'] <= 0 or cfg['entropy_coefficient'] < 0):
        raise ValueError('Unsupported molecular size policy configuration')


def remove_seed_hydrogens(mol):
    # RDKit otherwise retains H nodes next to non-tetrahedral stereocentres.
    # The size policy controls heavy-atom nodes, including after force-field
    # preparation adds explicit H atoms for its local relaxation.
    parameters = Chem.RemoveHsParameters()
    parameters.removeNontetrahedralNeighbors = True
    return Chem.RemoveHs(mol, parameters)


def edit_catalog(parent, cfg, donors=None):
    """Enumerate chemically legal locations, with no docking/property calls."""
    if cfg.get('version') in (8,9):
        from utils.molecular_fragment_search import fragment_catalog
        if donors is None:raise ValueError('Fragment edits require the causal frontier donor pool')
        return fragment_catalog(parent,cfg,donors)
    if cfg.get('version') in (2,3,4,5,6,7):
        from utils.molecular_structure_search import structure_catalog
        return structure_catalog(parent,cfg)
    validate_config(cfg)
    mol = Chem.MolFromMolBlock(parent['mol_block'], removeHs=True, sanitize=True)
    if mol is None or Chem.MolToSmiles(mol) != parent['smiles']:
        raise ValueError('Size policy requires a valid scored parent')
    mol = remove_seed_hydrogens(mol)
    n = mol.GetNumAtoms()
    if n != mol.GetNumHeavyAtoms() or not cfg['min_atoms'] <= n <= cfg['max_atoms']:
        raise ValueError('Size parent is outside the heavy-atom domain')
    catalog = [[], [(mol, dict(kind='keep'))], []]
    seen = [set(), set(), set()]

    def accept(index, edit, info):
        candidate = edit.GetMol()
        # Deleting a substituent can expose a bracketed stereocentre. Allow it
        # to acquire an implicit H instead of silently creating a radical.
        for atom in candidate.GetAtoms():
            if atom.GetAtomicNum() == 6 and atom.GetFormalCharge() == 0:
                atom.SetNoImplicit(False)
        try:
            Chem.SanitizeMol(candidate)
        except (ValueError, RuntimeError):
            return
        if any(a.GetNumRadicalElectrons() for a in candidate.GetAtoms()):
            return
        Chem.AssignStereochemistry(candidate, cleanIt=True, force=True)
        if len(Chem.GetMolFrags(candidate)) != 1 or candidate.GetNumHeavyAtoms() != n + cfg['deltas'][index]:
            return
        smiles = Chem.MolToSmiles(candidate)
        if smiles in seen[index]:
            return
        seen[index].add(smiles)
        catalog[index].append((candidate, info))

    for atom in mol.GetAtoms():
        i = atom.GetIdx()
        if n > cfg['min_atoms']:
            neighbors = [a.GetIdx() for a in atom.GetNeighbors()]
            if atom.GetDegree() == 1 and not atom.GetIsAromatic():
                edit = Chem.RWMol(mol)
                edit.RemoveAtom(i)
                accept(0, edit, dict(kind='terminal_delete', atom=i))
            elif (atom.GetDegree() == 2 and not atom.GetIsAromatic()
                  and atom.GetFormalCharge() == 0
                  and all(b.GetBondType() == Chem.BondType.SINGLE for b in atom.GetBonds())
                  and mol.GetBondBetweenAtoms(*neighbors) is None):
                edit = Chem.RWMol(mol)
                edit.AddBond(*neighbors, Chem.BondType.SINGLE)
                edit.RemoveAtom(i)
                accept(0, edit, dict(kind='saturated_contract', atom=i))
        if (n < cfg['max_atoms'] and atom.GetAtomicNum() in (6, 7, 8, 16)
                and atom.GetFormalCharge() == 0 and atom.GetTotalNumHs() > 0):
            edit = Chem.RWMol(mol)
            if atom.GetNumExplicitHs():
                edit.GetAtomWithIdx(i).SetNumExplicitHs(atom.GetNumExplicitHs()-1)
            new = edit.AddAtom(Chem.Atom(6))
            edit.AddBond(i, new, Chem.BondType.SINGLE)
            accept(2, edit, dict(kind='carbon_attach', atom=i))
    return mol, catalog


def prepared_edit(mol, info, rng):
    """Keep the parent's pocket pose; locally relax the edited seed, without scoring."""
    mol = Chem.Mol(mol)
    if info['kind'] == 'keep':
        return mol
    conf = mol.GetConformer()
    if info['kind'] == 'carbon_attach':
        i, new = info['atom'], mol.GetNumAtoms()-1
        pos = np.asarray(conf.GetPositions())
        neighbors = [a.GetIdx() for a in mol.GetAtomWithIdx(i).GetNeighbors() if a.GetIdx() != new]
        direction = pos[i]-pos[neighbors].mean(0) if neighbors else rng.normal(size=3)
        if np.linalg.norm(direction) < .1:
            direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        conf.SetAtomPosition(new, pos[i]+1.5*direction)
    n = mol.GetNumAtoms()
    hydrogenated = Chem.AddHs(mol, addCoords=True)
    props = AllChem.MMFFGetMoleculeProperties(hydrogenated)
    ff = (AllChem.MMFFGetMoleculeForceField(hydrogenated, props) if props is not None else
          AllChem.UFFGetMoleculeForceField(hydrogenated) if AllChem.UFFHasAllMoleculeParams(hydrogenated) else None)
    if ff is not None:
        for i in range(n - (info['kind'] == 'carbon_attach')):
            if props is not None:
                ff.MMFFAddPositionConstraint(i, .25, 100.)
            else:
                ff.UFFAddPositionConstraint(i, .25, 100.)
        ff.Initialize()
        ff.Minimize(maxIts=150)
    mol = remove_seed_hydrogens(hydrogenated)
    if not np.isfinite(mol.GetConformer().GetPositions()).all():
        raise FloatingPointError('Nonfinite resized seed coordinates')
    Chem.AssignStereochemistryFrom3D(mol, replaceExistingTags=True)
    return mol


class SizePPO:
    def __init__(self, cfg, frontier_dim, history_dim, reward_dim, cost_dim, device):
        validate_config(cfg)
        self.cfg = copy.deepcopy(cfg)
        self.frontier_dim, self.history_dim = frontier_dim, history_dim
        self.reward_dim, self.cost_dim = reward_dim, cost_dim
        self.device = torch.device(device)
        self.action_count = len(cfg['actions']) if cfg['version']>=2 else len(cfg['deltas'])
        # N, molecular weight, heteroatom fraction, rings, rotatable bonds, charge;
        # arbitrary-dimensional scored parent properties and frontier history.
        feature_dim = 6 + frontier_dim + history_dim
        self.actor = nn.Sequential(nn.Linear(feature_dim+reward_dim, cfg['hidden']), nn.Tanh(),
            nn.Linear(cfg['hidden'], self.action_count)).to(device)
        nn.init.zeros_(self.actor[-1].weight)
        nn.init.zeros_(self.actor[-1].bias)
        if cfg['version']>=5:
            with torch.no_grad():
                self.actor[-1].bias.copy_(torch.tensor(cfg['action_prior'],device=device).log())
        self.critic = VectorCritic(feature_dim, reward_dim, cfg['hidden'], cost_dim=cost_dim).to(device)
        self.optimizer = torch.optim.Adam(self.actor.parameters(), lr=cfg['actor_lr'])
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=cfg['critic_lr'])
        self.updates = 0
        self.proposed_graphs = set()

    def distribution(self, features, preferences, masks):
        logits = self.actor(torch.cat([features, preferences], dim=-1))
        return torch.distributions.Categorical(logits=logits.masked_fill(~masks, -torch.inf))

    @torch.no_grad()
    def propose(self, parent, history, preferences, seed, excluded_graphs=()):
        mol, catalog = edit_catalog(parent, self.cfg)
        n = mol.GetNumHeavyAtoms()
        chemical = [n/self.cfg['max_atoms'], Descriptors.MolWt(mol)/800.,
            sum(a.GetAtomicNum() != 6 for a in mol.GetAtoms())/n,
            rdMolDescriptors.CalcNumRings(mol)/10., rdMolDescriptors.CalcNumRotatableBonds(mol)/20.,
            Chem.GetFormalCharge(mol)/4.]
        if len(parent['objectives']) != self.frontier_dim or len(history) != self.history_dim:
            raise ValueError('Size policy parent/history schema mismatch')
        preferences = torch.as_tensor(preferences, dtype=torch.float32, device=self.device).detach()
        features = torch.tensor(chemical+list(parent['objectives'])+list(history),
            dtype=torch.float32, device=self.device).repeat(len(preferences), 1)
        masks = torch.tensor([bool(items) for items in catalog], device=self.device).repeat(len(preferences), 1)
        dist = self.distribution(features, preferences, masks)
        generator = torch.Generator(device='cpu').manual_seed(seed)
        actions = torch.multinomial(dist.probs.cpu(), 1, generator=generator).squeeze(1).to(self.device)
        old_logp = dist.log_prob(actions)
        rng = np.random.default_rng(seed+170003)
        novelty = self.cfg['version']>=5
        if novelty:
            from utils.frontier_discovery import molecular_graph
            blocked=set(excluded_graphs)|self.proposed_graphs|{molecular_graph(parent['smiles'])}
            graphs=[[molecular_graph(Chem.MolToSmiles(m)) for m,_ in items] for items in catalog]
            masks=[];actions=[];old_logp=[];probabilities=[]
            selections=[];fallbacks=[]
            # Each step conditions on earlier proposals in this batch. The
            # frozen mask is replayed in PPO; uniform site choices cancel in
            # the likelihood ratio because they contain no policy parameters.
            for i in range(len(preferences)):
                available=[[j for j,g in enumerate(group) if g not in blocked] for group in graphs]
                fallback=not any(available)
                if fallback:available=[[0]]+[[] for _ in catalog[1:]]
                mask=torch.tensor([bool(items) for items in available],device=self.device)
                current=self.distribution(features[i:i+1],preferences[i:i+1],mask[None])
                action=int(torch.multinomial(current.probs[0].cpu(),1,generator=generator))
                index=available[action][int(rng.integers(len(available[action])))]
                blocked.add(graphs[action][index]);self.proposed_graphs.add(graphs[action][index])
                selections.append(index);fallbacks.append(fallback);masks.append(mask)
                actions.append(action);old_logp.append(current.log_prob(torch.tensor([action],device=self.device))[0])
                probabilities.append(current.probs[0])
            masks=torch.stack(masks);actions=torch.tensor(actions,device=self.device)
            old_logp=torch.stack(old_logp);probabilities=torch.stack(probabilities)
        parents, metadata = [], []
        for i, action in enumerate(actions.tolist()):
            candidate, edit = catalog[action][selections[i] if novelty else int(rng.integers(len(catalog[action])))]
            edit=copy.deepcopy(edit)
            if self.cfg['version']>=2:
                from utils.molecular_structure_search import prepare_structure_edit
                prepared = prepare_structure_edit(candidate,edit,rng)
            else:
                prepared = prepared_edit(candidate, edit, rng)
            block = Chem.MolToMolBlock(prepared)
            serialized = Chem.MolFromMolBlock(block, removeHs=True, sanitize=True)
            if serialized is None:
                raise ValueError('Edited seed did not survive MolBlock serialization')
            resized = dict(parent, smiles=Chem.MolToSmiles(serialized), mol_block=block)
            delta = edit['delta'] if self.cfg['version']>=2 else self.cfg['deltas'][action]
            if (prepared.GetNumHeavyAtoms() != n+delta or prepared.GetNumAtoms() != n+delta
                    or serialized.GetNumAtoms() != n+delta):
                raise ValueError('Size action was not executed exactly')
            parents.append(resized)
            metadata.append(dict(action=action, delta=delta, parent_atoms=n, seed_atoms=n+delta,
                seed_smiles=resized['smiles'], edit=edit, mask=masks[i].tolist(),
                probabilities=(probabilities if novelty else dist.probs)[i].tolist(), old_logp=float(old_logp[i]), seed=int(seed)))
            if novelty:
                metadata[-1].update(proposal_graph=graphs[action][selections[i]],
                    novelty_exhausted=fallbacks[i])
        trace = dict(features=features.cpu(), preferences=preferences.cpu(), masks=masks.cpu(),
            actions=actions.cpu(), old_logp=old_logp.cpu(), metadata=metadata)
        return parents, trace

    def trace_data(self, trace):
        return {k: trace[k].to(self.device) for k in ('features','preferences','masks','actions','old_logp')}

    def policy_statistics(self, data):
        dist=self.distribution(data['features'],data['preferences'],data['masks'])
        return dist,dist.log_prob(data['actions']),dist.entropy()

    def actor_parameters(self):
        return self.actor.parameters()

    def update(self, trace, scores, training, reference, multipliers=None):
        data = self.trace_data(trace)
        features, prefs, masks, actions, old = [data[k] for k in ('features','preferences','masks','actions','old_logp')]
        rewards = torch.tensor([r['objectives'] for r in scores], device=self.device, dtype=torch.float32)
        costs = torch.tensor([r['cost'] if self.cost_dim is None else r['constraint_costs'] for r in scores],
            device=self.device, dtype=torch.float32)
        if rewards.shape != (len(actions), self.reward_dim):
            raise ValueError('Size PPO reward/attempt mismatch')
        if not torch.allclose(prefs, prefs[:1].expand_as(prefs)):
            raise ValueError('Size PPO needs one frozen preference per training group')
        ref = torch.as_tensor(reference, device=self.device, dtype=torch.float32)
        with torch.no_grad():
            initial,initial_logp,_ = self.policy_statistics(data)
            error = float((initial_logp-old).abs().max())
            if error > 1e-5:
                raise ValueError(f'Size old-policy replay mismatch: {error}')
            values, cost_values = self.critic(features, prefs)
            advantages, cost_advantages = rewards-values, costs-cost_values
            if training.get('group_relative_advantages',False):
                from utils.morl import group_relative_advantages
                advantages=group_relative_advantages(rewards)
        for _ in range(training['update_epochs']):
            dist,logp,entropy = self.policy_statistics(data)
            direction, info = preco_direction(logp-old, advantages, rewards, prefs[0], ref,
                training['clip_epsilon'], training['preco_strength'],training.get('preference_mix',0.))
            ratio = (logp-old).clamp(-20,20).exp()
            clipped = ratio.clamp(1-training['clip_epsilon'], 1+training['clip_epsilon'])
            if self.cost_dim is None:
                penalty = training['cost_coefficient']*torch.maximum(ratio*cost_advantages, clipped*cost_advantages).mean()
            else:
                weights = torch.as_tensor(multipliers, device=self.device, dtype=torch.float32)
                penalty = (weights*torch.maximum(ratio[:,None]*cost_advantages,
                    clipped[:,None]*cost_advantages).mean(0)).sum()
            loss = -(direction*logp).sum()+penalty-self.cfg['entropy_coefficient']*entropy.mean()
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(self.actor_parameters(), training['max_grad_norm'])
            if not torch.isfinite(loss) or not torch.isfinite(norm):
                raise FloatingPointError('Nonfinite size PPO actor update')
            self.optimizer.step()
            values, cost_values = self.critic(features, prefs)
            critic_loss = (values-rewards).square().mean()+(cost_values-costs).square().mean()
            self.critic_optimizer.zero_grad(set_to_none=True)
            critic_loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), training['max_grad_norm'])
            if not torch.isfinite(critic_loss) or not torch.isfinite(norm):
                raise FloatingPointError('Nonfinite size PPO critic update')
            self.critic_optimizer.step()
        self.updates += 1
        with torch.no_grad():
            updated,_,updated_entropy = self.policy_statistics(data)
        by_action = []
        labels=self.cfg['actions'] if self.cfg['version']>=2 else self.cfg['deltas']
        for action, label in enumerate(labels):
            selected = actions == action
            by_action.append(dict(**({'kind':label} if self.cfg['version']>=2 else {'delta':label}), attempts=int(selected.sum()),
                mean_reward_vector=rewards[selected].mean(0).tolist() if selected.any() else None,
                mean_cost_vector=costs[selected].mean(0).tolist() if selected.any() else None))
        return dict(updates=self.updates, optimizer_steps=training['update_epochs'], replay_error=error,
            action_counts=torch.bincount(actions,minlength=self.action_count).tolist(),
            by_action=by_action,
            mean_probabilities_before=initial.probs.mean(0).tolist(),
            mean_probabilities_after=updated.probs.mean(0).tolist(),
            entropy=float(updated_entropy.mean()), loss=float(loss.detach()),
            mean_seed_atoms=float(np.mean([r['seed_atoms'] for r in trace['metadata']])),
            weights=info['weights'].tolist())

    def state_dict(self):
        state=dict(config=self.cfg, frontier_dim=self.frontier_dim, history_dim=self.history_dim,
            reward_dim=self.reward_dim, cost_dim=self.cost_dim, actor=self.actor.state_dict(),
            critic=self.critic.state_dict(), optimizer=self.optimizer.state_dict(),
            critic_optimizer=self.critic_optimizer.state_dict(), updates=self.updates)
        if self.cfg['version']>=5:state['proposed_graphs']=sorted(self.proposed_graphs)
        return state

    def load_state_dict(self, state):
        for name, current in [('config',self.cfg),('frontier_dim',self.frontier_dim),
                ('history_dim',self.history_dim),('reward_dim',self.reward_dim),('cost_dim',self.cost_dim)]:
            if state[name] != current:
                raise ValueError('Size policy checkpoint schema mismatch: '+name)
        self.actor.load_state_dict(state['actor'])
        self.critic.load_state_dict(state['critic'])
        self.optimizer.load_state_dict(state['optimizer'])
        self.critic_optimizer.load_state_dict(state['critic_optimizer'])
        self.updates = state['updates']
        if self.cfg['version']>=5:self.proposed_graphs=set(state['proposed_graphs'])


def new_size_policy(exp, saved=None):
    cfg = exp.protocol.get('size_policy')
    if cfg is None:
        return None
    policy_class=SizePPO
    if cfg['version']==6:
        from utils.molecular_candidate_policy import CandidatePPO
        policy_class=CandidatePPO
    elif cfg['version']==7:
        from utils.molecular_surrogate_policy import SurrogatePPO
        policy_class=SurrogatePPO
    elif cfg['version'] in (8,9):
        from utils.molecular_fragment_policy import FragmentPPO
        policy_class=FragmentPPO
    policy = policy_class(cfg, len(exp.frontier_objectives), exp.history_dim, len(exp.reward_schema),
        len(exp.constraints) if exp.constraints else None, exp.device)
    if saved is not None:
        policy.load_state_dict(saved['size_policy'])
    return policy
