"""Shared local scoring client for the native molecular-optimization baselines."""
import json
import urllib.request
import urllib.error
import time


def read_property(mol, context, property_name):
    return context['mainstream_properties'][property_name]


def score_request(url, arm, request_id, smiles, stage='search'):
    body = json.dumps(dict(arm=arm, request_id=request_id, smiles=smiles, stage=stage)).encode()
    request = urllib.request.Request(url+'/score', data=body, headers={'Content-Type': 'application/json'})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    # A scoring-service restart must not lose an emitted batch or spend its
    # allowance twice. The server keys requests by arm/stage/request_id.
    for attempt in range(11):
        try:
            with opener.open(request, timeout=1800) as response:
                return json.load(response)['records']
        except urllib.error.HTTPError as error:
            if error.code not in (502,503,504) or attempt==10:
                raise
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            if attempt==10:
                raise
        time.sleep(5)


def scalar_reward(row):
    """Smooth positive constraint-aware utility for native scalar-reward methods."""
    if not row['valid']:
        return 0.
    import math
    positive = [max(v, 1e-6) for v in row['frontier_objectives']]
    utility = sum(positive)/len(positive)
    return utility/(1+utility)*math.exp(-sum(row['constraint_costs']))
