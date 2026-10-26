"""Causal frontier-fragment context for the joint guided PPO policy."""
import copy

from utils.molecular_size_policy import edit_catalog
from utils.molecular_surrogate_policy import SurrogatePPO


class FragmentPPO(SurrogatePPO):
    def __init__(self,cfg,frontier_dim,history_dim,reward_dim,cost_dim,device):
        if cfg['version'] not in (8,9):raise ValueError('Frontier-fragment PPO requires structure version 8 or 9')
        super().__init__(cfg,frontier_dim,history_dim,reward_dim,cost_dim,device)
        self.donors={}

    def set_donors(self,pocket,donors):
        # Reconstructed from the saved frontier pool before every rollout,
        # including after resume; not a second independent training archive.
        self.donors[str(pocket)]=copy.deepcopy(donors)

    def catalog(self,parent):
        return edit_catalog(parent,self.cfg,donors=self.donors[parent['pocket']])
