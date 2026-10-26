"""Preference and history conditioning for diffusion predictions."""
import torch.nn.functional as F


def guided_predictions(model, kwargs, conditions, guidance_scale=2.5):
    """CFG keeps the same protein pocket in conditional and null branches.

    Coordinates are affine-combined. Categorical distributions are combined
    in log space and renormalized, rather than subtracting probabilities.
    """
    conditional = model(**kwargs, objective_condition=conditions)
    if guidance_scale == 1:
        return conditional
    null_kwargs = dict(kwargs)
    if 'history_condition' in null_kwargs:
        null_kwargs['history_condition'] = None
    unconditional = model(**null_kwargs, objective_condition=None)
    result = dict(conditional)
    result['pred_ligand_pos'] = (unconditional['pred_ligand_pos'] + guidance_scale *
                               (conditional['pred_ligand_pos'] - unconditional['pred_ligand_pos']))
    for key in ('pred_ligand_v', 'pred_bond'):
        lc, lu = F.log_softmax(conditional[key], -1), F.log_softmax(unconditional[key], -1)
        result[key] = F.log_softmax(lu + guidance_scale * (lc - lu), -1)
    return result
