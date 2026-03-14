# Dataset Path Convention

This project uses a minimal, explicit dataset path convention for EBU6307.

## Current verified roots

- Local repo: `/Users/nemoyu/Desktop/openclaw-operate`
- Remote repo: `/root/code/new_folder/openclaw-operate`
- Remote writable data root: `/root/code/new_folder/openclaw-operate/workspace/data`

Do not assume `/limx_embop/tos/users/Nemo/self-work` contains usable datasets. It is currently empty and is not a guaranteed source.

## Expected Middlebury layout

Store Middlebury data under a dataset root chosen in config. Each scene should be a folder containing the standard files needed by the project, for example:

```text
<dataset_root>/
└── middlebury/
    ├── Adirondack/
    │   ├── im0.png
    │   ├── im1.png
    │   ├── calib.txt
    │   ├── disp0.pfm
    │   ├── disp1.pfm
    │   └── mask0nocc.png
    └── ArtL/
        ├── im0.png
        ├── im1.png
        ├── calib.txt
        ├── disp0.pfm
        └── ...
```

Minimum assumption for O1:

- scene directory exists per sample
- left and right images are `im0.png` and `im1.png`
- calibration metadata is `calib.txt`
- reference disparity is available as `disp0.pfm`

Additional Middlebury files may be present and should remain alongside the scene.

## Practical rule

- Configure dataset roots in [`configs/dataset_paths.example.yaml`](/Users/nemoyu/Desktop/openclaw-operate/configs/dataset_paths.example.yaml).
- Keep datasets outside tracked source folders when practical.
- On the verified remote environment, prefer `/root/code/new_folder/openclaw-operate/workspace/data` as the writable base for copied or prepared datasets.
