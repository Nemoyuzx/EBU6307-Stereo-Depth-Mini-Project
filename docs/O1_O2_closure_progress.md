# O1/O2 Closure Progress

## 2026-03-31 Phase 2 - dataset root correction

### What was verified
- Remote `workspace/data/middlebury` already exists as a symlink to `/limx_embop/tos/users/Nemo/self-work/middlebury_scenes2021/extracted/data`.
- That linked root contains the 24 required official scene folders plus two extra entries that must be excluded from formal O1/O2 discovery: `o4_tiny_scene` and `data/`.
- Existing remote O1/O2 outputs and CSVs currently include `o4_tiny_scene`, so a clean rerun is still required.

### Changes prepared locally
- Tightened `codes/common.py` scene discovery to allow only the official 24 scene names.
- Explicitly excluded `o4_tiny_scene` and `data` from automatic scene discovery.
- Next step: sync the patched code to the remote host, clean O1/O2 formal output folders there, rerun O1/O2, then sync the refreshed results back locally.

## 2026-03-31 Phase 3 - O2 constraint rewrite

### New hard constraints received from user
- O2 must compute SIFT on the original dataset image first, then apply a randomly generated transformation/warping, then evaluate repeatability on the transformed image.
- O2 repeatability must measure whether the same SIFT features remain consistent under transformation; plain left/right stereo matching is no longer acceptable as the final O2 definition.
- The three O2 example groups must be clearly different in dominant transformation type.
- Allowed transformation families include affine / scaling / rotation / intensity variation, but every example group must show a dominant qualitative difference.
- All transformation parameters must be randomly generated, not hand-fixed.
- Final code and output naming must remain executable and PDF-compatible.

### Implementation changes in progress
- Replaced the old O2 left/right stereo matching interpretation with an original-image repeatability pipeline.
- New O2 flow: detect SIFT on original `im0.png` → generate a scene-specific random transform → detect SIFT on the transformed image → keep only descriptor matches that are consistent with the known random transform geometry/intensity case → compute repeatability from repeatable matches.
- Each scene now stores its transform family, random seed, and random parameter JSON in both metrics and per-scene README files.
- `scripts/prepare_o1_o2_submission_assets.py` is being updated to export the PDF-named `Reapitability.csv` from the new O2 metric schema and to force the three examples to come from distinct dominant transform families when available.

### Required rerun
- The previously generated O2 results are now obsolete under the updated definition and must be discarded/replaced.
- After the code sync, rerun remote O2 formally on the 24 official scenes and regenerate O2 examples/CSV exports.

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
