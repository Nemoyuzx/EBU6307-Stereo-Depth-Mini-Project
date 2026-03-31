# O1/O2 Closure Progress

## 2026-03-31 Phase 1 - audit and plan

### What was verified
- Local repo: `/Users/nemoyu/Desktop/openclaw-operate`
- Remote host reachable: `ssh -p 40043 root@14.103.233.39`
- Remote machine: `di-20260314150230-22g2l`
- Remote GPU present: `NVIDIA A100-SXM4-80GB (81920 MiB)`
- Remote project path exists: `/root/code/new_folder/openclaw-operate`
- Current remote `workspace/data` only keeps temporary fallback stereo frames; formal Middlebury source scenes are not currently present under the configured data root.
- Remote `results/` already contains earlier O1/O2/O3/O4 outputs for `chess2`, `chess3`, `curule1`, but these need to be treated as prior artifacts until formal rerun is completed against the verified official dataset input.

### PDF requirements extracted for O1/O2
- O1 pipeline image: `results/O1a_synthetic_data/syn_pipeline.jpg`
- O1 synthetic outputs in dataset-like scene folders under `results/O1b_synthetic_data/`
- O1 metric table: `results/O1c_synthetic_data/SSIM.csv`
- O2 pipeline image: `results/O2a_sift/sift_pipeline.jpg`
- O2 three example images: `results/O2b_sift/example_[#number].jpg`
- O2 metric table: `results/O2c_sift/Reapitability.csv` (keep the PDF spelling exactly for submission compatibility)
- Dataset requirement: use Middlebury 2021 mobile stereo data, original PNG/PFM assets, no manual compression/distortion/reshaping of the source images.

### Official dataset source confirmed
- Middlebury 2021 page: `https://vision.middlebury.edu/stereo/data/scenes2021/`
- Zip index: `https://vision.middlebury.edu/stereo/data/scenes2021/zip/`
- `all.zip` available (~404M) and contains the 24 default scene folders without the optional ambient subdirectories.

### Execution plan
1. Sync the current local repo to the remote project path.
2. Download and extract official Middlebury `all.zip` into remote `workspace/data/middlebury` without altering image geometry.
3. Run formal O1 and O2 on the remote machine against the official dataset root.
4. Add local helper(s) to generate PDF-required exact-name artifacts for O1/O2, including:
   - pipeline figures with nano-banana / local generation path
   - O2 example images `example_1.jpg` to `example_3.jpg`
   - duplicate/export metrics files using exact PDF names where needed
5. Sync remote formal results back to local.
6. Validate local outputs and prepare O1/O2 submission-oriented structure/naming.
7. Commit after each finished stage.

### Open items to resolve during execution
- Whether to keep both `metrics.csv` and `Reapitability.csv` for O2 export compatibility.
- Whether report text should be produced as Markdown only or mirrored into a final submission PDF draft later.
