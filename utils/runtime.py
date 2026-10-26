"""Paths and JSON output for ParexMol experiments."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB_STEM = 'crossdocked_v1.1_rmsd1.0_processed_full_ref_prior_aromatic'


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
