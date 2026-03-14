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
- installs this repository in editable mode inside that env
- creates safe working folders under `/root/code/new_folder`
- creates result skeletons under the project directory
- avoids delete commands entirely

## Intended milestone order

1. `O1`: synthetic stereo pair generation and SSIM evaluation
2. `O2`: SIFT feature extraction, repeatability evaluation, visualizations
3. `O3`: SIFT-based stereo matching and disparity error reporting
4. `O4`: transformer-based stereo baseline with 5-fold evaluation under the 16 GB GPU limit

## Dataset path setup

Configure dataset and result roots in [`configs/dataset_paths.example.yaml`](/Users/nemoyu/Desktop/openclaw-operate/configs/dataset_paths.example.yaml).

The expected Middlebury scene layout is documented in [`docs/dataset_paths.md`](/Users/nemoyu/Desktop/openclaw-operate/docs/dataset_paths.md). For the verified remote environment, prefer `/root/code/new_folder/openclaw-operate/workspace/data` as the writable dataset base.
If remote stereo-video datasets are used as a temporary O1 engineering fallback, keep them separate from final assignment data as described in [`docs/remote_stereo_fallback.md`](/Users/nemoyu/Desktop/openclaw-operate/docs/remote_stereo_fallback.md).

Run the minimal O1 baseline with:

```bash
python -m ebu6307_stereo --config configs/dataset_paths.example.yaml --profile local --dry-run
```

Validate previously written O1 synthetic outputs without changing anything:

```bash
python -m ebu6307_stereo --config configs/dataset_paths.example.yaml --profile local --validate-results
```

When run without `--dry-run`, O1 writes one scene-like folder per processed sample under `results/O1b_synthetic_data/`:

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

## Immediate next step

Start with O1-first development:
- wire dataset path handling
- implement a tiny synthetic warping baseline
- write outputs into `results/O1*`
- keep all scripts terminal-runnable
