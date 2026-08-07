# Installation

DiBO supports Python 3.10 in a `virtualenv` environment. The tested
environment uses the following core versions:

| Component | Version |
|---|---|
| Python | `3.10.12` |
| virtualenv | `20.13.0` |
| pip | `24.0` |
| setuptools | `65.5.1` |
| wheel | `0.38.4` |
| PyTorch | `2.6.0+cu118` |
| Transformers | `4.44.2` |
| Tokenizers | `0.19.1` |
| NumPy | `1.26.4` |
| Accelerate | `1.13.0` |
| BitsAndBytes | `0.49.0` |
| OmegaConf | `2.3.0` |
| Weights & Biases | `0.21.2` |

## Core Environment

Create the environment from the repository root:

```bash
python3.10 -m pip install --user "virtualenv==20.13.0"
python3.10 -m virtualenv .venv
source .venv/bin/activate

python -m pip install --upgrade \
  "pip==24.0" \
  "setuptools==65.5.1" \
  "wheel==0.38.4"
python -m pip install \
  --index-url https://download.pytorch.org/whl/cu118 \
  "torch==2.6.0+cu118"
python -m pip install -r requirements.txt
```

The CUDA wheel includes the CUDA 11.8 user-space runtime. The host must still
provide a compatible NVIDIA driver. The model is loaded in BF16, so training
and evaluation require a BF16-capable NVIDIA GPU; the experiments used NVIDIA
H100 GPUs.

Check the installation:

```bash
python -m pip check
python - <<'PY'
import platform

import numpy
import tokenizers
import torch
import transformers

print("Python:", platform.python_version())
print("PyTorch:", torch.__version__)
print("PyTorch CUDA runtime:", torch.version.cuda)
print("Transformers:", transformers.__version__)
print("Tokenizers:", tokenizers.__version__)
print("NumPy:", numpy.__version__)
print("CUDA available:", torch.cuda.is_available())
PY
```

Training uses the preprocessed task bundles in `data/task_bundles`. The bundles
cache the benchmark filtering, score alignment, sorting, and normalization
steps. Design-Bench is therefore not required for training, and the slower
benchmark initialization and preprocessing do not run again for every job.

The model loader downloads the pinned LLaDA snapshot from Hugging Face. For an
offline compute node, download the snapshot to shared storage first and pass
its path with `--model_name_or_path`, or set `MODEL_NAME_OR_PATH` when using
`scripts/run_pipeline.sh`.

## Oracle Environment

TFBind10 evaluation reads `data/raw/TFBind10-Exact-v0.npz` directly. Its
scores already use DiBO's higher-is-better `-ddG` convention, so evaluation
does not negate them again. TFBind8, Ant, and D'Kitty evaluation use
Design-Bench. Ant and D'Kitty additionally require MuJoCo 2.0 and OpenGL
libraries.

### System Packages

On Ubuntu 22.04:

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential python3.10-dev patchelf curl unzip \
  libosmesa6-dev libosmesa6 mesa-common-dev libglew-dev \
  libglu1-mesa-dev libglu1-mesa libgl1-mesa-dev \
  libglvnd-dev libegl-dev libgles-dev libopengl-dev
```

On a managed system without administrator access, install or load equivalent
compiler and OpenGL development libraries before building `mujoco-py`.

### MuJoCo 2.0

Install MuJoCo 2.0 in its conventional user directory:

```bash
mkdir -p "$HOME/.mujoco"
curl -L \
  https://www.roboti.us/download/mujoco200_linux.zip \
  -o "$HOME/.mujoco/mujoco200_linux.zip"
unzip -o "$HOME/.mujoco/mujoco200_linux.zip" -d "$HOME/.mujoco"

export MUJOCO_PY_MUJOCO_PATH="$HOME/.mujoco/mujoco200_linux"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export LD_LIBRARY_PATH="$MUJOCO_PY_MUJOCO_PATH/bin:${LD_LIBRARY_PATH:-}"
```

### Python Dependencies

Install the build-time packages before `mujoco-py`, then install the complete
oracle dependency set:

```bash
python -m pip install \
  "Cython==0.29.36" \
  "cffi==1.17.1" \
  "fasteners==0.19" \
  "glfw==2.9.0" \
  "imageio==2.37.0" \
  "patchelf==0.17.2.1" \
  "PyOpenGL==3.1.9"
python -m pip install \
  --index-url https://download.pytorch.org/whl/cu118 \
  "torchvision==0.21.0+cu118"
python -m pip install \
  --no-build-isolation \
  --no-use-pep517 \
  "mujoco-py==2.0.2.13"
python -m pip install -r requirements-oracle.txt
```

The oracle package versions are:

| Component | Version |
|---|---|
| Design-Bench | `2.0.20` |
| TensorFlow | `2.17.0` |
| Keras | `3.12.0` |
| DeepChem | `2.8.0` |
| Gym | `0.13.1` |
| MuJoCo Python bindings | `2.0.2.13` |
| Morphing Agents | `1.5.1` |
| ROBEL | commit `5b0fd3704629931712c6e0f7268ace1c2154dc83` |

The exact Design-Bench data snapshot used for the DiBO experiments is available
from [DiBO-DesignBench-Snapshot](https://huggingface.co/datasets/zpointsun/DiBO-DesignBench-Snapshot).
It is a raw source archive for the Design-Bench cache, not a ready-made DiBO
`data/` directory. Install it into the active environment with:

```bash
hf download zpointsun/DiBO-DesignBench-Snapshot design_bench_data.zip \
  --repo-type dataset --revision v1.0.0 --local-dir data/downloads
python scripts/prepare_design_bench_data.py \
  --archive data/downloads/design_bench_data.zip
```

The script validates the archive and installs its files into the
`design_bench_data` directory used by Design-Bench. The optional `--target`
argument is intended for staging or inspection, not automatic oracle discovery.
See [`../data/README.md`](../data/README.md) for the distinction between this
complete cache and the preprocessed task bundles.

Verify the Python and simulator layers:

```bash
python -m pip check
python - <<'PY'
import src.compat.compat_patches  # Apply NumPy compatibility before Design-Bench imports.
import design_bench
import mujoco_py

print("mujoco-py:", mujoco_py.__version__)
for name in (
    "TFBind8-Exact-v0",
    "AntMorphology-Exact-v0",
    "DKittyMorphology-Exact-v0",
):
    task = design_bench.make(name)
    print(name, task.x.shape)
PY
```
