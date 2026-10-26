"""Graph remapping and full-group PPO gradients must survive microbatching."""
import copy
import unittest
from unittest.mock import patch

import torch
from torch import nn

from scripts.run_v18 import replay_microbatches, update
from utils.diffusion_ppo import joint_logp, reference_kl,seed_biased_log_probs
from utils.molecular_objectives import ObjectiveSet
from utils.morl import VectorCritic, preco_direction


class ToyPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.atom = nn.Linear(4, 11)
        self.bond = nn.Linear(6, 5)


def toy_transition(model, context, state, timestep):
    pos, _, _ = state
    batch = context['batch_ligand']
    condition = context['objective_condition']
    signal = pos.new_zeros((len(pos), 1)) if condition is None else condition[batch, :1]
    atom = model.atom(torch.cat([pos, signal], -1))
    src, dst = context['ligand_fc_bond_index']
    bond = model.bond(torch.cat([pos[src], pos[dst]], -1))
    std = context['prior_stds'][context['ligand_decomp_batch']]
    atom_lp=seed_biased_log_probs(atom[:,3:],context.get('seed_atom_types'),context.get('seed_atom_logit_bias',0.))
    bond_lp=seed_biased_log_probs(bond,context.get('seed_bond_types'),context.get('seed_bond_logit_bias',0.))
    return atom[:, :3], std, atom_lp, bond_lp


class CaptureSGD(torch.optim.SGD):
    def step(self, closure=None):
        if not hasattr(self, 'captured'):
            self.captured = []
        self.captured.append([p.grad.clone() for group in self.param_groups for p in group['params']])
        return super().step(closure)


class MicrobatchTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(843)
        sizes = [2, 3, 4, 2, 3]
        batch = torch.repeat_interleave(torch.arange(5), torch.tensor(sizes))
        edges, bond_batches, offset = [], [], 0
        for i, n in enumerate(sizes):
            pairs = torch.combinations(torch.arange(n), r=2).T
            pairs = torch.cat([pairs, pairs.flip(0)], 1) + offset
            edges.append(pairs)
            bond_batches.append(torch.full((pairs.shape[1],), i))
            offset += n
        edge = torch.cat(edges, 1)
        self.context = dict(protein_pos=torch.randn(10, 3, dtype=torch.double),
            protein_v=torch.randn(10, 29, dtype=torch.double), batch_protein=torch.arange(5).repeat_interleave(2),
            batch_ligand=batch, batch_ligand_bond=torch.cat(bond_batches),
            ligand_fc_bond_index=edge, ligand_v_aux=torch.randn(sum(sizes), 2, dtype=torch.double),
            prior_stds=torch.rand(10, 3, dtype=torch.double)+.7,
            ligand_decomp_batch=batch*2+1,
            objective_condition=torch.tensor([[.6, .3, .1]], dtype=torch.double).repeat(5, 1))
        state = (torch.randn(sum(sizes), 3, dtype=torch.double), torch.zeros(sum(sizes), dtype=torch.long),
                 torch.zeros(edge.shape[1], dtype=torch.long))
        self.model = ToyPolicy().double()
        self.reference = copy.deepcopy(self.model).requires_grad_(False)
        with torch.no_grad():
            self.reference.atom.weight.add_(.05)
            params = toy_transition(self.model, self.context, state, 500)
            mean, std, atom, bond = params
            action = (mean+std*torch.randn_like(mean), atom.argmax(-1), bond.argmax(-1))
            drift = torch.randn_like(mean)*.02
            self.step = dict(state=state, action=action, drift=drift, t=500,
                old_logp=joint_logp(params, action, drift, self.context),
                features=torch.randn(5, 18, dtype=torch.double))
        self.critic = VectorCritic(18, 3, 8).double()
        self.scores = [dict(objectives=v, cost=c) for v, c in zip(
            [[.8,.2,-.3], [.1,.7,.5], [-.4,.6,.3], [.2,-.3,.8], [.4,.5,-.2]], [.1, 1., .4, .2, 2.])]
        self.objectives = ObjectiveSet([dict(name=n, evaluator='qed') for n in ('a', 'b', 'c')])
        self.cfg = dict(update_epochs=1, clip_epsilon=.2, preco_strength=.1, cost_coefficient=.1,
                        reference_kl_coefficient=.01, max_grad_norm=1e6)

    def test_sliced_graphs_preserve_probabilities_kl_and_nonlocal_prior_indices(self):
        with torch.no_grad():
            full = toy_transition(self.model, self.context, self.step['state'], 500)
            ref = toy_transition(self.reference, self.context, self.step['state'], 500)
            expected_kl = reference_kl(full, ref, self.context)
            chunks = replay_microbatches(self.context, self.step, ref, 2)
            self.assertEqual([len(s['old_logp']) for _, _, s, _ in chunks], [2, 2, 1])
            for part, c, s, r in chunks:
                params = toy_transition(self.model, c, s['state'], 500)
                torch.testing.assert_close(joint_logp(params, s['action'], s['drift'], c), self.step['old_logp'][part])
                torch.testing.assert_close(reference_kl(params, r, c), expected_kl[part])
                src, dst = c['ligand_fc_bond_index']
                torch.testing.assert_close(c['batch_ligand'][src], c['batch_ligand_bond'])
                torch.testing.assert_close(c['batch_ligand'][dst], c['batch_ligand_bond'])

    def test_accumulated_gradient_matches_full_group_weighted_ppo_objective(self):
        # Independent differentiable PPO surrogate, including clipped negative
        # advantages, cost and KL. Gradients are compared before the SGD step.
        with torch.no_grad():
            returns = torch.tensor([s['objectives'] for s in self.scores], dtype=torch.float32)
            costs = torch.tensor([s['cost'] for s in self.scores], dtype=torch.float32)
            values, cost_values = self.critic(self.step['features'], self.context['objective_condition'])
            advantages, cost_adv = returns-values, costs-cost_values
            ref = toy_transition(self.reference, dict(self.context, objective_condition=None), self.step['state'], 500)
        expected_model = copy.deepcopy(self.model)
        expected_optimizer = CaptureSGD(expected_model.parameters(), lr=.03)
        clipped = False
        for _ in range(2):
            params = toy_transition(expected_model, self.context, self.step['state'], 500)
            logp = joint_logp(params, self.step['action'], self.step['drift'], self.context)
            ratio = (logp-self.step['old_logp']).clamp(-20, 20).exp()
            clipped |= bool(((ratio < .8) | (ratio > 1.2)).any())
            _, info = preco_direction(logp-self.step['old_logp'], advantages, returns,
                self.context['objective_condition'][0], torch.zeros(3), .2, .1)
            surrogate = torch.minimum(ratio[:,None]*advantages, ratio.clamp(.8,1.2)[:,None]*advantages).mean(0)
            loss = -(surrogate*info['effective_weights']).sum()
            loss += .1*torch.maximum(ratio*cost_adv, ratio.clamp(.8,1.2)*cost_adv).mean()
            loss += .01*reference_kl(params, ref, self.context).mean()
            expected_optimizer.zero_grad()
            loss.backward()
            expected_optimizer.step()
        self.assertTrue(clipped)
        for size in (1, 2, 5):
            model, critic = copy.deepcopy(self.model), copy.deepcopy(self.critic)
            actor_optimizer = CaptureSGD(model.parameters(), lr=.03)
            critic_optimizer = torch.optim.SGD(critic.parameters(), lr=.001)
            with patch('scripts.run_v18.transition', side_effect=toy_transition):
                result = update(model, self.reference, actor_optimizer, critic, critic_optimizer,
                    [(self.context, [self.step])], self.scores, self.objectives,
                    dict(self.cfg, update_epochs=2), microbatch_size=size)
            self.assertEqual(result['optimizer_steps'], 2)
            for actual_step, expected_step in zip(actor_optimizer.captured, expected_optimizer.captured):
                for actual, wanted in zip(actual_step, expected_step):
                    torch.testing.assert_close(actual, wanted, atol=1e-10, rtol=1e-8)
            for actual, wanted in zip(model.parameters(), expected_model.parameters()):
                torch.testing.assert_close(actual, wanted, atol=1e-10, rtol=1e-8)

    def test_seed_priors_survive_microbatch_likelihood_kl_and_gradient_replay(self):
        self.context.update(seed_atom_types=torch.arange(len(self.context['batch_ligand']))%8,
            seed_bond_types=torch.arange(len(self.context['batch_ligand_bond']))%5,
            seed_atom_logit_bias=6.,seed_bond_logit_bias=4.)
        with torch.no_grad():
            params=toy_transition(self.model,self.context,self.step['state'],500)
            self.step['old_logp']=joint_logp(params,self.step['action'],self.step['drift'],self.context)
        self.test_sliced_graphs_preserve_probabilities_kl_and_nonlocal_prior_indices()
        self.test_accumulated_gradient_matches_full_group_weighted_ppo_objective()


if __name__ == '__main__':
    unittest.main()
