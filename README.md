# More Test Users, More Overconfidence

This repository contains code for the paper
"More Test Users, More Overconfidence: Training-Seed Variance in Single-Run Sequential Recommender Tests."

Offline recommender papers often compare two trained checkpoints with a per-user paired test and then make an algorithm-level claim. The experiments here isolate the missing replication unit: the trained run. With fixed splits, fixed users, and fixed metric code, the same stochastic training procedure can be declared significantly different from itself by user-level tests, while run-level tests remain calibrated.

## Environment

The experiments were run with Python 3.10, PyTorch, PyTorch Lightning, Hydra, SciPy, pandas, and NumPy. A fresh environment can be prepared with:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

If CUDA-specific PyTorch wheels are needed, install the appropriate PyTorch build first, then install the remaining requirements.

## Data Layout

Training expects split CSV files under `data/`:

```text
data/
  Movielens-1m/
    leave-one-out-no_cold_items/
      train.csv
      validation_input.csv
      validation_target.csv
      test_input.csv
      test_target.csv
  Beauty/
    leave-one-out-no_cold_items/
      ...
  Movielens-20m/
    leave-one-out-no_cold_items/
      ...
```

The required columns are `user_id`, `item_id`, and `timestamp` after preprocessing. Dataset-specific raw column mappings are in `splitlight/runs/configs/dataset/`. Public raw datasets are not redistributed here.


## Running Experiments

Set the data root if the splits are not under `./data`:

```bash
export SEQ_SPLITS_DATA_PATH=$PWD/data
```

Run the full SASRec matrix from the paper:

```bash
python experiments/run_seed_matrix.py \
  --datasets Movielens-1m Beauty Movielens-20m \
  --configs C0 C1 C2 \
  --gpus 0 1
```

Run the GRU4Rec same-procedure null check:

```bash
python experiments/run_gru4rec_null.py \
  --datasets Movielens-20m \
  --gpus 0 1
```

Both launchers skip completed per-user metric files unless `--force` is provided. Outputs are written to `outputs/seed_matrix/`.

After training finishes, regenerate the analysis artifacts:

```bash
python experiments/analyze_seed_matrix.py \
  --input-root outputs/seed_matrix \
  --analysis-dir outputs/analysis \
  --figure-dir outputs/figures \
  --table-dir outputs/tables
```

To analyze a copied result matrix instead of the default output directory:

```bash
python experiments/analyze_seed_matrix.py \
  --input-root /path/to/seed_matrix \
  --analysis-dir /path/to/analysis \
  --figure-dir /path/to/figures \
  --table-dir /path/to/tables
```

## Optional Split Generation

If preprocessed interactions are already available as `data/<dataset>/preprocessed.csv`, the leave-one-out split used in the paper can be regenerated with:

```bash
cd splitlight/runs
SEQ_SPLITS_DATA_PATH=../../data python split.py \
  dataset=Movielens-1m \
  split_type=leave-one-out \
  split_params.remove_cold_items=True
```

Run the analogous command for `Beauty` and `Movielens-20m`. The study conditions on this fixed evaluator; split variance is intentionally outside the main experiment.


## Citation

If you find our work helpful, please consider citing the paper:

```bibtex
@inproceedings{gusak2026overconfidence,
  title={More Test Users, More Overconfidence: Seed Variance in Sequential Recommender Tests},
  author={Gusak, Danil and Volodkevich, Anna and Frolov, Evgeny},
  booktitle={Proceedings of the 20th ACM Conference on Recommender Systems},
  doi={10.1145/3773078.3841281},
  year={2026}
}
```