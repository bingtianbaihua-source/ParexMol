"""Molecular property scoring service for ParexMol V18."""
import argparse
import json
import math
import os
import sys
from pathlib import Path
import subprocess
import signal
import threading
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

from scripts.run_v18 import MORLExperiment
from utils.runtime import ROOT, save_json
from utils.evaluation.docking_vina import VinaDockingTask
from utils.molecular_constraints import ConstraintSet
from utils.geometry_score import geometry_score
from utils.evaluation.scoring_func import get_chem


def save_oracle_state(path, value):
    # Reports read these files while other pockets continue scoring.
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


class Oracle:
    def prepare_context(self, protocol, folder, context_name='context'):
        self.protocol, self.folder = protocol, folder
        folder.mkdir(parents=True, exist_ok=True)
        case = json.loads((ROOT/protocol['case_config']).read_text())
        self.exp = MORLExperiment('cpu', folder/context_name, case)
        self.target = self.exp.targets[0]
        self.parent = self.exp.original[self.target['target_id']]['ligand_rdmol']
        self.parent = self.exp.stereochemical_records([dict(mol=self.parent)])[0]['mol']
        self.center = np.asarray(self.parent.GetConformer().GetPositions()).mean(0)
        self.parent_fp = AllChem.GetMorganFingerprintAsBitVect(self.parent, 2, nBits=2048)
        return case

    def __init__(self, protocol, folder, workers=8):
        case=self.prepare_context(protocol,folder)
        self.measurements = json.loads((folder/'measurements.json').read_text()) if (folder/'measurements.json').exists() else {}
        self.requests = json.loads((folder/'requests.json').read_text()) if (folder/'requests.json').exists() else {}
        specifications = list(case['constraints'])
        if protocol.get('hard_similarity', True):
            specifications.append(dict(name='lead_similarity', evaluator='validity', relation='ge',
                threshold=protocol['similarity']['threshold'], limit=.2, cost_type='hinge', scale=.4))
        self.constraints = ConstraintSet(specifications)
        self.worker_log = (folder/'admet_worker.log').open('a')
        self.worker = subprocess.Popen([os.environ.get('PAREXMOL_ADMET_PYTHON', sys.executable), '-u',
            str(ROOT/'scripts/admet_worker.py')], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=self.worker_log, text=True, bufsize=1,
            env=dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2'))
        self.lock = threading.Lock()
        self.pool=ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),
            initializer=initialize_preparation_worker,initargs=(protocol,str(folder)))

    def close(self):
        self.pool.shutdown(wait=True,cancel_futures=True)
        self.worker.stdin.close()
        self.worker.wait(timeout=30)
        self.worker_log.close()

    def properties(self, smiles, members):
        if not smiles:
            return {}
        fields = self.protocol.get('admet_properties',
            dict(logs_reward='Solubility_AqSolDB', herg_reward='hERG'))
        self.worker.stdin.write(json.dumps(dict(smiles=smiles, members=members,
            properties=list(dict.fromkeys(fields.values()))))+'\n')
        self.worker.stdin.flush()
        line = self.worker.stdout.readline()
        if not line:
            raise RuntimeError('ADMET worker stopped; inspect admet_worker.log')
        result = json.loads(line)
        if set(result) != set(smiles):
            raise ValueError('ADMET output does not match requested structures')
        return result

    def prepare(self, smiles, index):
        mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
        if mol is None or '.' in Chem.MolToSmiles(mol):
            return dict(valid=False, error='invalid_or_disconnected_smiles')
        if not 6 <= mol.GetNumHeavyAtoms() <= 60:
            return dict(valid=False, error='outside_shared_6_to_60_heavy_atom_domain')
        try:
            mol = Chem.AddHs(mol)
            settings = AllChem.ETKDGv3()
            settings.randomSeed = self.protocol['docking']['seed']
            settings.numThreads = 1
            embedded = AllChem.EmbedMolecule(mol, settings)
            if embedded < 0:
                return dict(valid=False, error='common_conformer_failed')
            relaxation = 'ETKDG_only'
            if AllChem.MMFFHasAllMoleculeParams(mol):
                AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
                relaxation = 'MMFF'
            elif AllChem.UFFHasAllMoleculeParams(mol):
                AllChem.UFFOptimizeMolecule(mol, maxIters=200)
                relaxation = 'UFF'
            mol = Chem.RemoveHs(mol)
            Chem.AssignStereochemistryFrom3D(mol, replaceExistingTags=True)
        except Chem.rdchem.MolSanitizeException as error:
            # Some graphs parse as SMILES but fail stricter sanitization during
            # embedding/relaxation. This is a counted preparation failure.
            return dict(valid=False, error='common_conformer_failed', detail=str(error))
        coords = mol.GetConformer().GetPositions()
        shift = self.center-coords.mean(0)
        for i, xyz in enumerate(coords+shift):
            mol.GetConformer().SetAtomPosition(i, xyz)
        normalized = Chem.MolToSmiles(mol)
        folder = self.folder/'poses'/f'{index:06d}'
        folder.mkdir(parents=True, exist_ok=True)
        task = VinaDockingTask.from_generated_mol(mol, self.target['source_ligand'],
            protein_root=str(ROOT/'data/crossdocked_v1.1_rmsd1.0'), tmp_dir=str(folder),
            center=self.center, size_factor=None)
        task.size_x, task.size_y, task.size_z = self.protocol['docking']['box_size']
        try:
            docking = task.run(mode='dock', exhaustiveness=self.protocol['docking']['exhaustiveness'],
                               seed=self.protocol['docking']['seed'], cpu=self.protocol['docking']['cpu'])[0]
        except TypeError as error:
            message = str(error)
            if 'PDBQT parsing error: Atom type ' not in message or 'is not a valid AutoDock type' not in message:
                raise
            # For example, RxnFlow's building blocks include boron, which the
            # fixed Vina scorer cannot evaluate. Retain this failed attempt in
            # the ordinary measurement cache and request ledger; other members
            # of the batch still receive their original scores.
            return dict(valid=False, error='common_docking_unsupported_atom_type',
                        detail=message, smiles=normalized)
        # Preserve the standardized input and the returned docking pose for audit.
        with Chem.SDWriter(str(folder/'standardized.sdf')) as writer:
            writer.write(mol)
        (folder/'docked.pdbqt').write_text(docking['pose'])
        geometry,coverage=geometry_score(self.exp.amber,mol)
        chem = get_chem(mol)
        return dict(valid=True, smiles=normalized, qed=float(chem['qed']), sa=float(chem['sa']),
                    geometry_cost=geometry, geometry_statistics=coverage, vina_dock=float(docking['affinity']),
                    relaxation=relaxation, similarity=float(DataStructs.TanimotoSimilarity(self.parent_fp,
                        AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))), pose_dir=str(folder))

    def measure(self, smiles):
        keys = []
        for value in smiles:
            mol = Chem.MolFromSmiles(value) if isinstance(value, str) else None
            keys.append(Chem.MolToSmiles(mol) if mol is not None else str(value))
        new = list(dict.fromkeys(k for k in keys if k not in self.measurements))
        values=self.pool.map(prepare_in_worker,[(k,len(self.measurements)+i) for i,k in enumerate(new)])
        prepared=dict(zip(new,values))
        normalized = sorted({r['smiles'] for r in prepared.values() if r['valid']})
        predictions = self.properties(normalized, [0, 1, 2])
        for key, row in prepared.items():
            if row['valid']:
                values = predictions[row['smiles']]
                fields = self.protocol.get('admet_properties',
                    dict(logs_reward='Solubility_AqSolDB', herg_reward='hERG'))
                row.update({key: float(values[prop]) for key, prop in fields.items()})
                if not all(math.isfinite(row[k]) for k in
                           ('geometry_cost', 'vina_dock', 'similarity', *fields)):
                    raise ValueError('Nonfinite common scoring result')
            self.measurements[key] = row
        save_oracle_state(self.folder/'measurements.json', self.measurements)
        return [self.to_record(self.measurements[k]) for k in keys]

    def to_record(self, raw):
        transforms = self.protocol['objective_transforms']
        dimension = len(transforms)
        if not raw['valid']:
            return dict(valid=False, error=raw['error'], objectives=[-1.]*dimension, frontier_objectives=[-1.]*dimension,
                        raw_properties={}, raw_frontier_properties={}, cost=10., **self.constraints.failed())
        vector = [float(np.clip((raw[s['property']]-s['origin'])/s['scale'] *
            (1 if s['direction']=='max' else -1), 0., 1.)) for s in transforms]
        values = dict(valid_generation=1., geometry=raw['geometry_cost'], lead_similarity=raw['similarity'])
        costs, violations = zip(*(c.cost(values[c.name]) for c in self.constraints.items))
        properties = dict(zip([s['name'] for s in transforms], vector))
        return dict(raw, objectives=vector, frontier_objectives=vector, raw_properties=properties,
                    raw_frontier_properties=properties, constraint_values=values,
                    constraint_costs=list(costs), constraint_violations=list(violations),
                    feasible=not any(violations), cost=raw['geometry_cost'])

    def score(self, request):
        arm, stage, request_id = request['arm'], request['stage'], request['request_id']
        if arm not in self.protocol['methods']+['initial', 'smoke'] or stage not in ('search', 'initial', 'smoke'):
            raise ValueError('Unknown arm or stage')
        key = json.dumps([arm, stage, request_id])
        with self.lock:
            if key in self.requests:
                previous = self.requests[key]
                if previous['smiles'] != request['smiles']:
                    raise ValueError('Request ID reused with different molecules')
                return previous
            used = sum(len(r['smiles']) for r in self.requests.values() if r['arm']==arm and r['stage']=='search')
            if stage=='search' and used+len(request['smiles'])>self.protocol['attempts_per_method']:
                raise ValueError('Method proposal budget exceeded')
            records = self.measure(request['smiles'])
            result = dict(request, offset=used, records=records)
            self.requests[key] = result
            save_oracle_state(self.folder/'requests.json', self.requests)
            print(json.dumps(dict(arm=arm, stage=stage, offset=used, attempts=len(records),
                                  valid=sum(r['valid'] for r in records), feasible=sum(r['feasible'] for r in records))), flush=True)
            return result


def initialize_preparation_worker(protocol,folder):
    global PREPARATION_WORKER
    torch.set_num_threads(2)
    PREPARATION_WORKER=object.__new__(Oracle)
    PREPARATION_WORKER.prepare_context(protocol,Path(folder),f'context_{os.getpid()}')


def prepare_in_worker(item):
    return PREPARATION_WORKER.prepare(*item)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/scoring_5obj.json'))
    parser.add_argument('--port', type=int, default=18764)
    parser.add_argument('--workers',type=int,default=8,help='Independent molecule preparation processes; Vina still uses its fixed per-molecule CPU count')
    args = parser.parse_args()
    torch.set_num_threads(2)
    protocol = json.loads(args.config.read_text())
    folder = ROOT/protocol['outdir']/'oracle'
    oracle = Oracle(protocol, folder,args.workers)
    def stop(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            try:
                if self.path != '/score':
                    raise ValueError('Unknown endpoint')
                request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                result = oracle.score(request)
                payload = json.dumps(result, allow_nan=False).encode()
                self.send_response(200)
            except Exception as error:
                import traceback
                traceback.print_exc()
                payload = json.dumps(dict(error=f'{type(error).__name__}: {error}')).encode()
                self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    save_json(folder/'service.json', dict(pid=os.getpid(), url=f'http://127.0.0.1:{args.port}',workers=args.workers))
    try:
        server.serve_forever()
    finally:
        server.server_close()
        oracle.close()


if __name__ == '__main__':
    main()
