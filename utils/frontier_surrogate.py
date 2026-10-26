"""Causal vector GP and fixed-front expected HV for molecular proposals.

This is a proposal heuristic, not a property oracle or qNEHVI. Only public
measurements and completed batches enter the GP. Actual rewards still come
from the common evaluator. Objectives are independent conditional on a shared
Tanimoto/descriptor kernel; batch overlap uses an independence approximation.
"""
import copy
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from scipy.linalg import cho_factor,cho_solve
from scipy.special import ndtr
from scipy.stats import qmc

from utils.frontier_discovery import molecular_graph
from utils.molecular_candidate_policy import descriptors
from utils.morl import frontier

FP=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048,includeChirality=True)


@lru_cache(maxsize=50000)
def features(smiles):
    # Global parent search revisits catalogs larger than 10k molecules. Cache
    # deterministic graph features only; no objective labels or GP fits enter
    # this cache. 50k vectors occupy about 0.8 GiB at most on the CPU.
    mol=Chem.MolFromSmiles(smiles)
    if mol is None:raise ValueError('Surrogate input is not a valid molecule')
    return np.concatenate([FP.GetFingerprintAsNumPy(mol),descriptors(mol)]).astype(float)


def kernel(a,b):
    af,bf=a[:,:2048],b[:,:2048]
    dot=af@bf.T
    similarity=dot/np.maximum(af.sum(1)[:,None]+bf.sum(1)[None,:]-dot,1e-12)
    distance=((a[:,None,2048:]-b[None,:,2048:])**2).sum(-1)
    return .7*similarity+.3*np.exp(-.5*distance)


def posterior(train_x,train_y,test_x):
    mean=train_y.mean(0);scale=np.maximum(train_y.std(0),.1)
    covariance=kernel(train_x,train_x)+(.05**2+1e-8)*np.eye(len(train_x))
    factor=cho_factor(covariance,lower=True,check_finite=True)
    cross=kernel(test_x,train_x)
    mu=mean+(cross@cho_solve(factor,(train_y-mean)/scale))*scale
    variance=np.maximum(1.-np.sum(cross*cho_solve(factor,cross.T).T,axis=1),1e-10)
    std=np.sqrt(variance[:,None])*scale
    if not np.isfinite(mu).all() or not np.isfinite(std).all():
        raise FloatingPointError('Nonfinite GP prediction')
    return mu,std


def independent_coverage(mu,std,probes):
    """Evaluate the same 32-candidate blocks concurrently, with disjoint writes."""
    coverage=np.empty((len(mu),len(probes)),dtype=np.float32)
    def fill(offset):
        coverage[offset:offset+32]=ndtr((mu[offset:offset+32,None,:]-probes[None,:,:])/
                                      std[offset:offset+32,None,:]).prod(-1)
    offsets=range(0,len(mu),32)
    if len(mu)<=96 or not len(probes):
        for offset in offsets:fill(offset)
    else:
        with ThreadPoolExecutor(max_workers=4) as executor:
            list(executor.map(fill,offsets))
    return coverage


class SurrogateHistory:
    def __init__(self,cfg,reference,upper):
        if cfg['version'] not in (1,2):raise ValueError('Unsupported surrogate model version')
        self.cfg=copy.deepcopy(cfg)
        self.reference=np.asarray(reference,dtype=float)
        self.upper=np.asarray(upper,dtype=float)
        if (self.reference.ndim!=1 or not len(self.reference) or self.upper.shape!=self.reference.shape
                or not np.isfinite(self.reference).all() or not np.isfinite(self.upper).all()
                or np.any(self.upper<=self.reference)):
            raise ValueError('Surrogate requires a finite fixed HV box')
        self.dimension=len(self.reference)
        self.probes=qmc.Sobol(self.dimension,scramble=True,seed=cfg['seed']).random_base2(cfg['samples_power'])
        self.pockets={}

    def initialize(self,pocket,rows,excluded_smiles=()):
        key=str(pocket)
        if key in self.pockets:raise ValueError('Surrogate pocket already initialized')
        excluded={molecular_graph(s) for s in excluded_smiles}
        excluded.update(molecular_graph(r['smiles']) for r in rows if r['valid'])
        self.pockets[key]=dict(training={},front=[],excluded=sorted(excluded),observed_attempts=0,observed_batches=0)
        self.observe(key,rows,public=True)

    def observe(self,pocket,rows,public=False):
        state=self.pockets[str(pocket)];points=list(state['front']);excluded=set(state['excluded'])
        for row in rows:
            if not row['valid']:continue
            point=np.asarray(row['frontier_objectives'],dtype=float)
            if point.shape!=(self.dimension,) or not np.isfinite(point).all():
                raise ValueError('Surrogate observation has the wrong objective schema')
            scaled=(point-self.reference)/(self.upper-self.reference)
            if np.any(scaled>1+1e-12):raise ValueError('Observation exceeds the fixed HV box')
            smiles=Chem.MolToSmiles(Chem.MolFromSmiles(row['smiles']))
            state['training'][smiles]=point.tolist()
            if public or (row['feasible'] and molecular_graph(smiles) not in excluded):points.append(point.tolist())
        state['front']=frontier(points,self.dimension).tolist()
        if not public:
            state['observed_attempts']+=len(rows);state['observed_batches']+=1

    def context(self,pocket,smiles):
        state=self.pockets[str(pocket)];training=state['training']
        active=len(training)>=self.cfg['warmup_molecules']
        meta=dict(pocket=str(pocket),training_molecules=len(training),observed_attempts=state['observed_attempts'],
                  observed_batches=state['observed_batches'],active=active,candidate_smiles=list(smiles))
        if not active:return dict(meta=meta,coverage=None,remaining=None)
        covered=np.zeros(len(self.probes),dtype=bool)
        for point in state['front']:
            scaled=(np.asarray(point)-self.reference)/(self.upper-self.reference)
            covered|=np.all(self.probes<=scaled,axis=1)
        probes=self.probes[~covered]
        x=np.stack([features(s) for s in training])
        y=(np.asarray(list(training.values()))-self.reference)/(self.upper-self.reference)
        test=np.stack([features(s) for s in smiles])
        if self.cfg['version']==2:
            from utils.frontier_gp import fitted_posterior
            mu,std,_=fitted_posterior(x,y,test)
        else:
            mu,std=posterior(x,y,test)
        coverage=independent_coverage(mu,std,probes)
        return dict(meta=meta,coverage=coverage,remaining=np.ones(len(probes)))

    def bias(self,context,available):
        count=len(available)
        if context['coverage'] is None or not len(context['remaining']):return np.zeros(count,dtype=np.float32)
        acquisition=context['coverage']@context['remaining']/len(self.probes)
        maximum=float(np.max(acquisition[np.asarray(available,dtype=bool)]))
        if maximum<=0:return np.zeros(count,dtype=np.float32)
        relative=np.maximum(acquisition/maximum,self.cfg['relative_floor'])
        return np.clip(self.cfg['strength']*np.log(relative),-self.cfg['max_log_span'],0).astype(np.float32)

    def advance(self,context,choice):
        if context['coverage'] is not None:context['remaining']*=1-context['coverage'][choice]

    def state_dict(self):
        return copy.deepcopy(dict(config=self.cfg,reference=self.reference.tolist(),upper=self.upper.tolist(),pockets=self.pockets))

    @classmethod
    def from_state_dict(cls,state):
        obj=cls(state['config'],state['reference'],state['upper'])
        obj.pockets=copy.deepcopy(state['pockets'])
        return obj
