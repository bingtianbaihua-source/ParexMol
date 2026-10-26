# ParexMol

Although generative models hold promise for discovering molecules that balance multiple drug-relevant properties, they often fail to suggest molecules that improve upon the properties of structures represented in the training distribution. We suggest that this limitation arises in part from using molecular property predictors to evaluate candidates in isolation, without directly assessing their improvement over existing molecules. We address this challenge by introducing ParexMol (PAReto-frontier EXpansion for MOLecular design), a molecular generation framework that uses a dynamically updated empirical Pareto front to guide generation and learning. The framework selects objective preferences based on the historical front and incorporates candidate hypervolume gains into reinforcement learning feedback, allowing a generative model to adapt subsequent generation as discoveries accumulate. In seven- and nine-objective experiments across 20 protein pockets, ParexMol achieves the highest mean success rate among the evaluated methods. Its mean cumulative hypervolume gain exceeds that of PMMG, the best-performing baseline for this metric, by 49.43\% and 58.58\%, respectively. Furthermore, in the GSK3β lead optimization case, the hypervolume increased by 110.4\% relative to the initial set.

## Objective configurations

| Configuration | Objectives | Reward channels | Scoring port |
| --- | --- | ---: | ---: |
| [5 objectives](configs/v18_5obj.json) | Docking affinity, QED, normalized SA, predicted solubility, and inverted hERG risk | 11 | 18764 |
| [7 objectives](configs/v18_7obj.json) | The five objectives above, plus Caco-2 permeability and inverted CYP3A4 inhibition risk | 15 | 18765 |
| [9 objectives](configs/v18_9obj.json) | The seven objectives above, plus inverted AMES risk and metabolic stability | 19 | 18766 |

Each setup uses three files: an optimization configuration, a `scoring_*obj.json` file defining scoring and attempt budgets, and a `targets_*obj.json` manifest identifying the pocket and public sample indices. Each supplied target manifest contains one pocket.

Properties are transformed using fixed directions and scales so that all coordinates are maximized. Hypervolume uses coordinates clipped to `[0, 1]`; soft ranking uses unclipped coordinates to preserve differences beyond those boundaries. Molecular validity and geometry quality are separate constraints. Failed and duplicate proposals both count toward the attempt budget.

## Code navigation

```text
ParexMol/
├── configs/          Optimization settings, scoring protocols, and target pockets
├── datasets/         Protein and ligand graph data structures
├── models/           Diffusion networks, graph encoders, and conditional predictions
├── scripts/          Optimization entry point, scoring service, and ADMET worker
├── utils/            PPO, frontier feedback, surrogates, and molecular edits
├── tests/            Core algorithm and replay consistency tests
├── requirements.txt  Python dependencies
└── LICENSE
```

| Module | Purpose |
| --- | --- |
| [Optimization entry point](scripts/run_v18.py) | Preparation, online training, sampling, scoring, and reporting |
| [utils/morl.py](utils/morl.py) | Vector value networks, PreCo direction combination, preference scheduling, and constraint multipliers |
| [utils/diffusion_ppo.py](utils/diffusion_ppo.py) | Diffusion trajectories, joint action probabilities, and reference KL |
| [utils/local_diffusion.py](utils/local_diffusion.py) | Parent reconstruction, partial noising, and local denoising |
| [utils/frontier_discovery.py](utils/frontier_discovery.py) | Historical archives, discovery eligibility, and shared hypervolume credit |
| [utils/frontier_rank.py](utils/frontier_rank.py) | Per-objective rank feedback against the frozen historical frontier |
| [utils/frontier_parent_search.py](utils/frontier_parent_search.py) | Parent selection guided by the surrogate |
| [utils/frontier_surrogate.py](utils/frontier_surrogate.py) and [utils/frontier_gp.py](utils/frontier_gp.py) | Observation history, vector GP fitting, and candidate guidance |
| [utils/molecular_fragment_policy.py](utils/molecular_fragment_policy.py) and [utils/molecular_fragment_search.py](utils/molecular_fragment_search.py) | Fragment editing policies and enumeration of legal edits |
| [scripts/scoring_server.py](scripts/scoring_server.py) | Conformer preparation, Vina docking, property normalization, and scoring records |

To follow the optimization loop, start with `train()`, then read `choose_parent()`, the structure policy's `propose()`, `sample_lead()`, `DiscoveryArchive.credit()`, and `update()`.

## Main settings

All three objective configurations share the following core settings:

| Parameter | Default |
| --- | ---: |
| Candidates per round | 8 |
| Attempt budget per pocket | 128 |
| Rounds to complete the budget | 16 |
| Starting timestep for local diffusion | 100 |
| Training transitions retained per trajectory | 16 |
| PPO update epochs per round | 2 |
| PPO clipping coefficient | 0.2 |
| Preference mass for properties / HV / ranks | 0.2 / 0.5 / 0.3 |
| Heavy atom count | 6–60 |
| Donor fragment heavy atom count | 4–8 |
| Maximum donor fragments | 32 |
| Random seed | 7101 |

`--update-microbatch-size` controls the number of molecules processed in each backward pass. Rewards and update directions are still computed over the complete candidate group.

## Setup and execution

Run the following commands from the `ParexMol` directory in the project Python environment. Dependencies include PyTorch, PyTorch Geometric, RDKit, Open Babel, Vina, and ADMET-AI. The PyTorch graph extensions must match the installed PyTorch and CUDA builds.

```bash
python -m pip install -r requirements.txt
```

Target pockets are specified in `configs/targets_*obj.json`, and `baseline_dataset` identifies the public sample records. Runtime data uses the following layout:

```text
data/
├── crossdocked_v1.1_rmsd1.0/                                Receptor structures
├── crossdocked_v1.1_rmsd1.0_processed_full_ref_prior_aromatic.lmdb
├── public_preferences.pt                                  Public sample records
├── bond_stat.pt                                            Bond length statistics
└── angle_stat.pt                                           Bond angle statistics
```

For the five-objective configuration, start the CPU scoring service in one terminal and leave it running:

```bash
CUDA_VISIBLE_DEVICES='' python -m scripts.scoring_server \
  --config configs/scoring_5obj.json \
  --port 18764 \
  --workers 8
```

The ADMET worker uses the current Python interpreter by default. Set `PAREXMOL_ADMET_PYTHON` to use a separate interpreter for that worker.

In another terminal, prepare the public baseline and start online optimization:

```bash
python -m scripts.run_v18 prepare \
  --config configs/v18_5obj.json \
  --device cpu

CUDA_VISIBLE_DEVICES=0 python -m scripts.run_v18 train \
  --config configs/v18_5obj.json \
  --device cuda:0 \
  --seed 7101 \
  --max-rounds 16 \
  --max-seconds 14400 \
  --update-microbatch-size 2
```

To continue an interrupted run, keep the same configuration and add `--resume`. Set `--max-rounds` to the number of additional rounds required by the remaining budget. The output directory's `run.json` records the completed round count. For seven or nine objectives, switch both configuration files and use the corresponding scoring port.

The entry point also provides `sample`, `score`, and `report` stages for separate generation and evaluation. The scoring service accumulates search attempts per method, so independent evaluation should use its own scoring output directory and budget. To inspect all command-line options:

```bash
python -m scripts.run_v18 --help
python -m scripts.scoring_server --help
```

## Outputs

The optimization output directory is set by `outdir` in the selected configuration. For five objectives, scoring records are stored in `outputs/scoring_5obj/oracle/`; the other configurations use the corresponding objective count.

| Output | Contents |
| --- | --- |
| `protocol.json` | The fixed configuration for the run |
| `baseline.json` | Property and constraint scores for public samples |
| `run.json` | Run status, completed rounds, and optimizer update counts |
| `archive.json` | The Pareto archive |
| `final.pt` | Policy, optimizer, and historical state for resuming the run |
| `rollouts/<round>/` | Candidate molecules, scores, parent selection, structure actions, and discovery rewards |
| `oracle/requests.json` | Scoring requests and cumulative attempt records |
| `oracle/measurements.json` | Cached molecular property measurements |
| `evaluation/<tag>/report.json` | Hypervolume, validity, and feasibility metrics from evaluation |

## Tests

```bash
python -m pytest -q tests
```

Tests cover GP fitting, parent selection, rank feedback, structure and fragment edits, ring topology constraints, policy probability replay, state restoration, and consistency of PPO updates across microbatches.

## License

Licensed under the [Apache License 2.0](LICENSE).
