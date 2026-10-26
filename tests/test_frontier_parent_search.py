import copy
import json
import unittest
from unittest import mock

import numpy as np
from rdkit import Chem,RDLogger

from utils.frontier_discovery import molecular_graph
from utils.frontier_parent_search import (choose_parent,greedy_coverage,verify_search_trace,SELECTION,
                                         _cached_edit_identities)
from utils.frontier_seeds import FrontierSeedPool
from utils.frontier_surrogate import SurrogateHistory
from utils.molecular_structure_search import SURROGATE_STRUCTURE_POLICY,FRAGMENT_STRUCTURE_POLICY
from test_molecular_fragment_policy import donor


class ParentSearchTests(unittest.TestCase):
    def setUp(self):
        _cached_edit_identities.cache_clear()
        RDLogger.DisableLog('rdApp.error');RDLogger.DisableLog('rdApp.warning')
        self.settings=dict(version=1,exploration=.2)
        self.cfg=copy.deepcopy(SURROGATE_STRUCTURE_POLICY)
        self.cfg['surrogate'].update(version=2,warmup_molecules=2,samples_power=7)

    def tearDown(self):
        _cached_edit_identities.cache_clear()
        RDLogger.EnableLog('rdApp.error');RDLogger.EnableLog('rdApp.warning')

    def history(self,dimension=5):
        parents=[donor('CCCCCCO'),donor('CCCCCCN')]
        for i,p in enumerate(parents):
            p.update(graph=molecular_graph(p['smiles']),objectives=[.3+.4*((i+j)%2) for j in range(dimension)])
        pool=FrontierSeedPool([dict(name=str(i)) for i in range(dimension)])
        pool.pockets={'0':parents}
        surrogate=SurrogateHistory(self.cfg['surrogate'],[0.]*dimension,[1.]*dimension)
        surrogate.initialize(0,[dict(valid=True,feasible=True,smiles=p['smiles'],frontier_objectives=p['objectives']) for p in parents])
        return pool,surrogate

    def catalog(self,parent,cfg,donors=None):
        candidate='CCCCCCF' if parent['smiles']=='CCCCCCO' else 'CCCCCCCl'
        return Chem.MolFromSmiles(parent['smiles']),[[(Chem.MolFromSmiles(candidate),dict(kind='grow'))]]

    def context(self,pocket,smiles):
        values={'CCCCCCF':[.1,.2,.1,.2],'CCCCCCCl':[.7,.7,.7,.7]}
        return dict(coverage=np.asarray([values[s] for s in smiles],dtype=np.float32))

    def test_greedy_score_counts_one_edit_per_graph(self):
        coverage=np.full((2,4),.5)
        self.assertAlmostEqual(greedy_coverage(coverage,['a','a'],8,4),.5)
        self.assertAlmostEqual(greedy_coverage(coverage,['a','b'],8,4),.75)
        self.assertEqual(greedy_coverage(np.zeros((0,4)),[],8,4),0.)

    def test_parent_weights_preserve_exploration_and_ignore_conformer_multiplicity(self):
        pool,surrogate=self.history()
        before_pool=pool.state_dict();before_history=surrogate.state_dict()
        with mock.patch('utils.frontier_parent_search.edit_catalog',side_effect=self.catalog),mock.patch.object(surrogate,'context',side_effect=self.context):
            parent,trace=choose_parent(pool,0,9,self.settings,self.cfg,surrogate,set(),8)
            self.assertEqual(parent['selection'],SELECTION)
            self.assertTrue(trace['active']);self.assertEqual(trace['unique_candidates'],2)
            self.assertTrue(all(p>=.1 for p in trace['graph_probabilities']))
            self.assertAlmostEqual(sum(trace['graph_probabilities']),1.)
            self.assertEqual(pool.state_dict(),before_pool);self.assertEqual(surrogate.state_dict(),before_history)
            pool.pockets['0'].append(copy.deepcopy(pool.pockets['0'][0]))
            _,duplicate=choose_parent(pool,0,9,self.settings,self.cfg,surrogate,set(),8)
            self.assertEqual(trace['graphs'],duplicate['graphs'])
            np.testing.assert_allclose(trace['graph_probabilities'],duplicate['graph_probabilities'],atol=0,rtol=0)

    def test_blocked_edits_and_warmup_use_only_available_causal_information(self):
        pool,surrogate=self.history()
        with mock.patch('utils.frontier_parent_search.edit_catalog',side_effect=self.catalog),mock.patch.object(surrogate,'context',side_effect=self.context) as context:
            _,trace=choose_parent(pool,0,9,self.settings,self.cfg,surrogate,{'CCCCCCF'},8)
            self.assertEqual(trace['candidate_counts'],[0,1])
            self.assertEqual(trace['predicted_batch_gains'][0],0.)
            parent,empty=choose_parent(pool,0,9,self.settings,self.cfg,surrogate,{'CCCCCCF','CCCCCCCl'},8)
            self.assertEqual(empty['fallback'],'no_novel_edits')
            self.assertEqual(parent,pool.choose(0,9));self.assertEqual(context.call_count,1)
            surrogate.cfg['warmup_molecules']=3
            parent,warmup=choose_parent(pool,0,9,self.settings,self.cfg,surrogate,set(),8)
            self.assertEqual(warmup['fallback'],'surrogate_warmup')
            self.assertEqual(parent,pool.choose(0,9));self.assertEqual(context.call_count,1)

    def test_causal_trace_rejects_different_histories_and_probability_changes(self):
        pool,surrogate=self.history()
        with mock.patch('utils.frontier_parent_search.edit_catalog',side_effect=self.catalog),mock.patch.object(surrogate,'context',side_effect=self.context):
            _,trace=choose_parent(pool,0,9,self.settings,self.cfg,surrogate,set(),8)
        other=copy.deepcopy(trace);other['graph_probabilities'][0]+=1e-9
        verify_search_trace(other,trace)
        other=copy.deepcopy(trace);other['observed_attempts']+=8
        with self.assertRaises(ValueError):verify_search_trace(other,trace)
        other=copy.deepcopy(trace);other['predicted_batch_gains'][0]+=.01
        with self.assertRaises(AssertionError):verify_search_trace(other,trace)

    def test_real_fitted_surrogate_supports_variable_objectives_and_exact_resume(self):
        for dimension in [5,9]:
            pool,surrogate=self.history(dimension)
            blocked={p['graph'] for p in pool.pockets['0']}
            parent,trace=choose_parent(pool,0,9,self.settings,self.cfg,surrogate,blocked,4)
            self.assertTrue(trace['active']);self.assertGreater(trace['unique_candidates'],0)
            self.assertTrue(np.isfinite(trace['predicted_batch_gains']).all())
            p2=FrontierSeedPool.from_state_dict(pool.state_dict())
            s2=SurrogateHistory.from_state_dict(surrogate.state_dict())
            restored,replay=choose_parent(p2,0,9,self.settings,self.cfg,s2,blocked,4)
            self.assertEqual(parent,restored);verify_search_trace(replay,trace)

    def test_cached_catalog_keeps_history_exclusions_and_predictions_current(self):
        pool,surrogate=self.history()
        with mock.patch('utils.frontier_parent_search.edit_catalog',side_effect=self.catalog) as catalog, \
                mock.patch.object(surrogate,'context',side_effect=self.context) as context:
            parent,trace=choose_parent(pool,0,9,self.settings,self.cfg,surrogate,set(),8)
            repeated,replay=choose_parent(pool,0,9,self.settings,self.cfg,surrogate,set(),8)
            self.assertEqual(parent,repeated);self.assertEqual(trace,replay)
            self.assertEqual(catalog.call_count,2);self.assertEqual(context.call_count,2)
            surrogate.observe(0,[dict(valid=True,feasible=True,smiles='CCCCCCF',
                                      frontier_objectives=[.6]*5)])
            changed,current=choose_parent(pool,0,9,self.settings,self.cfg,surrogate,{'CCCCCCF'},8)
            self.assertEqual(catalog.call_count,2);self.assertEqual(context.call_count,3)
            self.assertEqual(current['candidate_counts'],[0,1])
            self.assertEqual(current['observed_batches'],1)
            with mock.patch('utils.frontier_parent_search._cached_edit_identities',
                            side_effect=_cached_edit_identities.__wrapped__):
                uncached,expected=choose_parent(pool,0,9,self.settings,self.cfg,surrogate,{'CCCCCCF'},8)
            self.assertEqual(changed,uncached);self.assertEqual(current,expected)

    def test_cache_keys_include_scored_identity_coordinates_and_topology_rules(self):
        from utils.molecular_size_policy import edit_catalog
        parent=donor('CCc1ccccc1CCO');encoded=json.dumps(self.cfg,sort_keys=True)
        with mock.patch('utils.frontier_parent_search.edit_catalog',wraps=edit_catalog) as catalog:
            original=_cached_edit_identities(parent['mol_block'],parent['smiles'],encoded)
            self.assertEqual(original,_cached_edit_identities(parent['mol_block'],parent['smiles'],encoded))
            self.assertEqual(catalog.call_count,1)
            mol=Chem.MolFromMolBlock(parent['mol_block']);position=mol.GetConformer().GetAtomPosition(0)
            position.x+=.1;mol.GetConformer().SetAtomPosition(0,position)
            shifted=_cached_edit_identities(Chem.MolToMolBlock(mol),parent['smiles'],encoded)
            self.assertEqual(original,shifted);self.assertEqual(catalog.call_count,2)
            with self.assertRaises(ValueError):
                _cached_edit_identities(parent['mol_block'],'CCCCCCN',encoded)
            limited=copy.deepcopy(self.cfg);limited['ring_closure_max_aromatic_overlap']=2
            filtered=_cached_edit_identities(parent['mol_block'],parent['smiles'],json.dumps(limited,sort_keys=True))
            self.assertLess(len(filtered),len(original))

    def test_dynamic_fragment_donors_bypass_static_catalog_cache(self):
        pool,surrogate=self.history();cfg=copy.deepcopy(FRAGMENT_STRUCTURE_POLICY)
        with mock.patch('utils.frontier_parent_search._cached_edit_identities',side_effect=AssertionError('dynamic donor cache')), \
                mock.patch('utils.frontier_parent_search.edit_catalog',side_effect=self.catalog) as catalog, \
                mock.patch.object(surrogate,'context',side_effect=self.context):
            choose_parent(pool,0,9,self.settings,cfg,surrogate,set(),8)
            pool.pockets['0'].append(copy.deepcopy(pool.pockets['0'][0]))
            choose_parent(pool,0,9,self.settings,cfg,surrogate,set(),8)
            self.assertEqual(catalog.call_count,5)


if __name__=='__main__':unittest.main()
