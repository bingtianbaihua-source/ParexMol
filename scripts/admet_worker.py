"""Persistent CPU ADMET worker; one JSON request/response per line."""
import contextlib
import json
import sys
import torch
from admet_ai import ADMETModel


def main():
    if torch.cuda.is_available():
        raise RuntimeError('ADMET must run on CPU')
    torch.set_num_threads(2)
    with contextlib.redirect_stdout(sys.stderr):
        model = ADMETModel(include_physchem=False, drugbank_path=None, num_workers=0,
                           fingerprint_multiprocessing_min=1000000)
    original_models, original_scalers = model.model_lists, model.scaler_lists
    for line in sys.stdin:
        request = json.loads(line)
        members = request['members']
        model.model_lists = [[g[i] for i in members] for g in original_models]
        model.scaler_lists = [[g[i] for i in members] for g in original_scalers]
        with contextlib.redirect_stdout(sys.stderr):
            frame = model.predict(smiles=request['smiles'])
        properties = request.get('properties', ['Solubility_AqSolDB', 'hERG'])
        data = frame[properties].to_dict(orient='index')
        print(json.dumps(data, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
