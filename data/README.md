# Project Data

This directory contains DiBO's project-prepared arrays, model-ready task
bundles, prompt templates, morphology relabels, evaluation ranges, and reward
statistics. It is not the Design-Bench data cache. The exact Design-Bench data
snapshot used for DiBO is distributed separately as `design_bench_data.zip`;
its contents are installed by `scripts/prepare_design_bench_data.py` into the
active Design-Bench Python environment, not into this directory.

## Layout

| Path | Contents |
|---|---|
| `raw/TFBind8-Exact-v0.npz` | TFBind8 designs and scores, 65,792 rows |
| `raw/TFBind10-Exact-v0.npz` | Corrected higher-is-better lookup over all `4^10` DNA sequences |
| `raw/AntMorphology-Exact-v0.npz` | Ant designs and source scores, 25,009 rows |
| `raw/DKittyMorphology-Exact-v0.npz` | D'Kitty designs and source scores, 25,009 rows |
| `relabeled/` | Project oracle scores aligned to the 10,004 visible Ant and D'Kitty designs |
| `task_bundles/` | Preprocessed, checkpoint-compatible training inputs |
| `prompts/` | Eight training and two validation prompt templates per task |
| `normalization_ranges.json` | Per-task minimum and maximum scores used by evaluation |
| `reward_stats/reward_stats.json` | Reward-distribution statistics used to scale RL advantages |

The files in `relabeled/` contain the same visible morphology designs as the
source data, paired with precomputed exact-oracle scores. The source task
labels and exact-oracle scores are not always identical, so the released Ant
and D'Kitty bundles use the exact-oracle scores as their training labels. The
source labels remain in the bundles for alignment and inspection. Each
relabeled score is matched to its raw design before the corresponding bundle
is sorted. The oracle evaluates a supplied morphology using a fixed controller
policy in MuJoCo; it does not generate additional designs.

All project NPZ files use named arrays and are loaded with
`allow_pickle=False`.

## Design-Bench Snapshot

[DiBO-DesignBench-Snapshot](https://huggingface.co/datasets/zpointsun/DiBO-DesignBench-Snapshot)
contains the exact `design_bench_data.zip` snapshot used in the DiBO
experiments. It is separate from the files in this directory and is needed for
the Design-Bench oracles used by TFBind8, Ant, and D'Kitty evaluation; TFBind10
evaluation uses `raw/TFBind10-Exact-v0.npz`. The snapshot also contains the
fixed Ant and D'Kitty controller policies used while MuJoCo scores a supplied
morphology.

The ZIP is a raw source archive, not a complete final DiBO `data/` directory:
unpacking it alone does not create task bundles, relabeled scores, prompts,
reward statistics, or normalization metadata. Download and install it into
the active Design-Bench environment with:

```bash
hf download zpointsun/DiBO-DesignBench-Snapshot design_bench_data.zip \
  --repo-type dataset --revision v1.0.0 --local-dir data/downloads
python scripts/prepare_design_bench_data.py \
  --archive data/downloads/design_bench_data.zip
```

For normal oracle use, omit `--target` so the script installs into the
`design_bench_data` location used by Design-Bench, rather than this project
`data/` directory. The option is available for staging or inspecting an archive
in a separate directory. Identical files are reused; differing files are
refused unless `--force` is supplied.

## Task Bundles

A task bundle is a model-ready `.npz` cache, not an upstream Design-Bench
format. It stores:

- `X_raw_sorted`: raw designs ordered by oracle score;
- `y_task_sorted`: task labels aligned to that ordering where available;
- `y_oracle_sorted`: oracle labels used by the training objective;
- `X_normalized_sorted`: normalized design representations used for
  similarity calculations;
- `y_task_normalized_sorted` and `y_oracle_normalized_sorted`: normalized
  labels where available;
- `task_name`: the task identifier.

The bundles preserve the benchmark's visible-data filtering and the project's
normalization and relabel alignment. Reading them directly avoids repeated
Design-Bench initialization, filtering, normalization, and oracle relabeling
before each training run.

Validate the included bundles:

```bash
python -m src.dataset.task_bundle --stage validate
```

TFBind8, Ant, and D'Kitty bundles can be rebuilt into a separate directory:

```bash
python -m src.dataset.task_bundle \
  --stage build \
  --tasks TFBind8-Exact-v0 AntMorphology-Exact-v0 DKittyMorphology-Exact-v0 \
  --output_dir outputs/task_bundles
```

The preprocessing uses the same visible-data percentile cutoffs as
Design-Bench 2.0.20: 50 percent for TFBind8 and 40 percent for each morphology
task. TFBind8 sequences are converted to soft categorical logits before
standardization; continuous designs are standardized directly. The
morphology rows are then aligned to `relabeled/` and sorted by the relabeled
score.

To compare a rebuilt bundle with the included version:

```bash
python scripts/compare_task_bundles.py \
  --reference data/task_bundles/TFBind8-Exact-v0.npz \
  --candidate outputs/task_bundles/TFBind8-Exact-v0.npz
```

## Reward Statistics

`reward_stats/reward_stats.json` contains statistics computed from 100,000
reward-only samples per task. Training reads its per-task reward standard
deviation to scale the RL advantage. The file also records distribution
summaries and clipping diagnostics.

The included statistics were sampled with the same `0.70-0.95` anchor band
used by the training dataset.

Generate a new statistics file with the recorded sampling settings:

```bash
python compute_reward_stats.py \
  --tasks TFBind8-Exact-v0 TFBind10-Exact-v0 \
          AntMorphology-Exact-v0 DKittyMorphology-Exact-v0 \
  --few_shot 7 7 7 7 \
  --n_pool 500 \
  --k_pool 50 \
  --ratio 0.8 \
  --lower_ratio 0.70 \
  --upper_ratio 0.95 \
  --split train \
  --seed <SEED> \
  --n_samples_per_task 100000 \
  --clip_lo_q 1 \
  --clip_hi_q 99 \
  --out_json outputs/reward_stats.json
```
