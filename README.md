# EBU6307 Stereo Depth Mini Project

Minimal bootstrap for the EBU6307 stereo depth assignment. This repository is organized to keep the required `codes/` and `results/` outputs obvious from the start, while leaving room for `docs/` and `scripts/` that support development.

## Layout

```text
.
├── codes/                 # Source, tests, packaging
├── configs/               # Example path configuration
├── docs/                  # Milestone notes / TODOs
├── results/               # Assignment outputs for O1 -> O4
├── scripts/               # Local / remote bootstrap helpers
├── environment.yml        # Minimal conda environment spec
└── README.md
```

## Local bootstrap

```bash
cd /Users/nemoyu/Desktop/openclaw-operate
conda env create -f environment.yml
conda activate ebu6307-stereo
python -m pip install -e .
python -m ebu6307_stereo
```

## Remote bootstrap

Remote host details for this project:

```bash
ssh -p 40043 root@14.103.233.39
cd /root/code/new_folder
```

Then either copy this repository into `/root/code/new_folder/openclaw-operate`, or clone/upload it there, and run:

```bash
bash scripts/setup_remote_conda.sh
```

The script only:
- creates a conda env from `environment.yml`
- attempts to install PyTorch for O4 (`cu121` wheels when `nvidia-smi` is present, CPU torch otherwise)
- installs this repository in editable mode inside that env
- creates safe working folders under `/root/code/new_folder`
- creates result skeletons under the project directory
- avoids delete commands entirely

## Intended milestone order

1. `O1`: synthetic stereo pair generation and SSIM evaluation
2. `O2`: SIFT feature extraction, repeatability evaluation, visualizations
3. `O3`: SIFT-based stereo matching and disparity error reporting
4. `O4`: trainable token-matching stereo baseline with a PyTorch/CUDA-capable route under the 16 GB GPU limit

## Dataset path setup

Configure dataset and result roots in [`configs/dataset_paths.example.yaml`](/Users/nemoyu/Desktop/openclaw-operate/configs/dataset_paths.example.yaml).

The official assignment target is Middlebury-style stereo scenes. Do not mix temporary remote stereo-video fallback data with final assignment data or default result folders.

The expected Middlebury scene layout is documented in [`docs/dataset_paths.md`](/Users/nemoyu/Desktop/openclaw-operate/docs/dataset_paths.md). For the verified remote environment, prefer `/root/code/new_folder/openclaw-operate/workspace/data` as the writable dataset base.
If remote stereo-video datasets are used as a temporary O1 engineering fallback, use [`configs/dataset_paths.fallback.example.yaml`](/Users/nemoyu/Desktop/openclaw-operate/configs/dataset_paths.fallback.example.yaml) and follow [`docs/fallback_data_policy.md`](/Users/nemoyu/Desktop/openclaw-operate/docs/fallback_data_policy.md).
The verified extraction path for that temporary input source remains documented in [`docs/remote_stereo_fallback.md`](/Users/nemoyu/Desktop/openclaw-operate/docs/remote_stereo_fallback.md).
The standalone fallback extractor can be invoked without touching the main O1 CLI: `PYTHONPATH=codes/src python3 -m ebu6307_stereo.fallback_extract --left-video ... --right-video ... --output-dir workspace/data/tmp_remote_stereo_fallback`.

Run the minimal O1 baseline with:

```bash
python -m ebu6307_stereo --config configs/dataset_paths.example.yaml --profile local --dry-run
```

Remove `--dry-run` to write O1 outputs into `results/O1b_synthetic_data/` and the SSIM summary into `results/O1c_synthetic_data/SSIM.csv`.

Run the minimal O2 baseline with:

```bash
python -m ebu6307_stereo --config configs/dataset_paths.example.yaml --profile local --objective o2 --dry-run
```

Run the minimal O3 baseline with:

```bash
python -m ebu6307_stereo --config configs/dataset_paths.example.yaml --profile local --objective o3 --dry-run
```

Run the minimal O4 baseline with:

```bash
python -m ebu6307_stereo --config configs/dataset_paths.example.yaml --profile local --objective o4 --dry-run
```

The O4 config block now exposes backend selection and model/training capacity directly:
- `backend`: `auto`, `torch`, or `numpy`
- `device`: `auto`, `cuda`, or `cpu`
- `execution_mode`: `baseline` or `dinov2_cost_volume`
- `disparity_regression`: `quadratic` or `soft_argmax`
- `model_dim`, `encoder_hidden_dim`, `encoder_layers`
- `training_epochs`, `training_learning_rate`, `training_batch_size`, `inference_batch_size`

For the coursework submission path, keep `execution_mode: baseline` as the default. This is the original ViT-style trainable token-projection route and is the path intended to stay compatible with the restricted pip whitelist. `backend: auto` prefers the torch path and moves to CUDA automatically when `torch.cuda.is_available()` is true; the same baseline can still run through the NumPy path when torch is unavailable.

`dinov2_cost_volume` is retained only as an explicit backup/experimental path. It is not the default submission route, requires torch, and depends on extra local DINOv2 assets (`o4.dinov2_repo_path`, `o4.dinov2_checkpoint_path`) that are outside the restricted submission dependency set. It does not silently fall back to the baseline when DINOv2 loading fails.

Only use the DINOv2 backup path when you intentionally want the non-default experimental route. Example:

```bash
python -m ebu6307_stereo \
    --config configs/dataset_paths.example.yaml \
    --profile remote \
    --objective o4 \
    --scene-name chess3 \
    --o4-execution-mode dinov2_cost_volume \
    --o4-dinov2-model facebook/dinov2-base \
    --o4-dinov2-repo /root/code/new_folder/openclaw-operate/workspace/external/dinov2 \
    --o4-dinov2-checkpoint /limx_embop/tos/users/Nemo/self-work/models/dinov2_vitb14_reg4_pretrain.pth \
    --o4-regression-mode quadratic
```

Target one exact scene directory deterministically:

```bash
python -m ebu6307_stereo --config configs/dataset_paths.example.yaml --profile local --scene-name sample_scene --dry-run
```

Validate previously written O1 synthetic outputs without changing anything:

```bash
python -m ebu6307_stereo --config configs/dataset_paths.example.yaml --profile local --validate-results
```

Validate previously written O2 outputs without changing anything:

```bash
python -m ebu6307_stereo --config configs/dataset_paths.example.yaml --profile local --objective o2 --validate-results
```

Validate previously written O3 outputs without changing anything:

```bash
python -m ebu6307_stereo --config configs/dataset_paths.example.yaml --profile local --objective o3 --validate-results
```

Validate previously written O4 outputs without changing anything:

```bash
python -m ebu6307_stereo --config configs/dataset_paths.example.yaml --profile local --objective o4 --validate-results
```

When run without `--dry-run`, O1 writes one scene-like folder per processed sample under `results/O1b_synthetic_data/` and the summary CSV under `results/O1c_synthetic_data/SSIM.csv`:

```text
results/O1b_synthetic_data/
└── <scene_name>/
    ├── im0.png
    ├── im1.png
    ├── disp0.pfm
    ├── calib.txt      # copied when present in the source scene
    └── README.txt
```

`im0.png` is copied from the source left image, `im1.png` is a simple horizontally shifted synthetic right image, and `disp0.pfm` is the matching constant-disparity baseline for the left view with exposed columns written as zero.

When run without `--dry-run`, O2 writes per-scene SIFT outputs under `results/O2a_sift/` and `results/O2b_sift/`, plus a summary CSV under `results/O2c_sift/metrics.csv`:

```text
results/O2a_sift/
└── <scene_name>/
    ├── im0_keypoints.png
    ├── im1_keypoints.png
    └── README.txt

results/O2b_sift/
└── <scene_name>/
    ├── sift_matches.png
    └── README.txt
```

The O2 baseline uses OpenCV SIFT on `im0.png` and `im1.png`, applies a relaxed Lowe ratio cutoff, keeps bidirectional agreement as a conditional filter, then runs fundamental-matrix RANSAC when enough candidates remain, and reports a simple repeatability proxy as `ratio_test_matches / min(left_keypoints, right_keypoints)`.

When run without `--dry-run`, O3 writes per-scene disparity outputs under `results/O3a_disparity/` and error analysis outputs under `results/O3b_disparity/`, plus a summary CSV under `results/O3c_disparity/metrics.csv`:

```text
results/O3a_disparity/
└── <scene_name>/
    ├── disp0.pfm
    ├── disp0.png
    └── README.txt

results/O3b_disparity/
└── <scene_name>/
    ├── error_map.png
    └── README.txt
```

The O3 baseline now derives disparity from SIFT feature correspondences on `im0.png` and `im1.png`, filters matches using the O2-style ratio/mutual/RANSAC path plus a rectified stereo constraint, then interpolates those feature disparities into a runnable dense left-view estimate before reporting `MAE`, `RMSE`, and `bad_1px` when the source scene includes `disp0.pfm`.

When run without `--dry-run`, O4 writes per-scene token-matching outputs under `results/O4a_transformer/` and analysis outputs under `results/O4b_transformer/`, plus per-scene and per-fold CSV summaries under `results/O4c_transformer/`:

```text
results/O4a_transformer/
└── <scene_name>/
    ├── disp0.pfm
    ├── disp0.png
    └── README.txt

results/O4b_transformer/
└── <scene_name>/
    ├── confidence.png
    ├── error_map.png
    └── README.txt

results/O4c_transformer/
├── metrics.csv
└── fold_summary.csv
```

O4 exposes two explicit execution modes. `baseline` is the default submission path and keeps the original trainable token-projection route. `dinov2_cost_volume` is retained only as a documented backup/experimental route that uses pretrained DINOv2 dense descriptors from an explicit local checkpoint and an explicit disparity cost volume with optional `soft_argmax` regression. The selected mode is controlled by config or CLI, and the DINOv2 path does not silently fall back to the baseline. A local checkout can be wired in explicitly through `o4.dinov2_repo_path` or `--o4-dinov2-repo`; remote `torch.hub` loading remains disabled.

## Current formal baseline state

The current O1-O4 formal baseline uses `configs/dataset_paths.example.yaml` as the documented path map:
- O1 synthetic scenes: `results/O1b_synthetic_data/`
- O1 metrics: `results/O1c_synthetic_data/SSIM.csv`
- O2 outputs: `results/O2a_sift/`, `results/O2b_sift/`, `results/O2c_sift/metrics.csv`
- O3 outputs: `results/O3a_disparity/`, `results/O3b_disparity/`, `results/O3c_disparity/metrics.csv`
- O4 outputs: `results/O4a_transformer/`, `results/O4b_transformer/`, `results/O4c_transformer/metrics.csv`
- O4 fold summary: `results/O4c_transformer/fold_summary.csv`
