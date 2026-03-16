# Fallback Data Policy

The official EBU6307 target remains Middlebury-style stereo scenes. Remote stereo-video inputs are allowed only as a temporary engineering fallback while no confirmed official dataset is available.

## When fallback is allowed

- use it only to validate O1 pipeline wiring, file I/O, and result writing
- use it only with extracted left/right frame pairs kept outside the official `middlebury/` tree
- stop using it for assignment reporting as soon as confirmed Middlebury-style data is available

## Required labeling

- use the dedicated fallback config: [`configs/dataset_paths.fallback.example.yaml`](/Users/nemoyu/Desktop/openclaw-operate/configs/dataset_paths.fallback.example.yaml)
- keep fallback inputs under a separate root such as `workspace/data/tmp_remote_stereo_fallback`
- keep fallback outputs under fallback-labeled paths such as `results/O1b_synthetic_data_fallback` and `fallback_SSIM.csv`
- describe any derived results as `fallback`, `temporary`, or `engineering-only`; do not present them as final assignment outputs

## What must be redone later

Once official Middlebury-style data is found, rerun the workflow with the standard config and redo:

- scene discovery against the official dataset root
- O1 synthetic output generation
- SSIM or other reported metrics
- any tables, screenshots, or writeups that previously used fallback-derived outputs

See [`docs/remote_stereo_fallback.md`](/Users/nemoyu/Desktop/openclaw-operate/docs/remote_stereo_fallback.md) for the currently verified extraction path.
