# O1 Baseline Status

The minimal O1 baseline is already part of the current formal O1-O4 project state.

- Run with `python codes/o1.py --config configs/dataset_paths.example.yaml --profile local` to write outputs.
- Synthetic scene outputs are written under `results/O1b_synthetic_data/`.
- The SSIM summary is written to `results/O1c_synthetic_data/SSIM.csv`.
- Validate existing O1 outputs with `python codes/o1.py --config configs/dataset_paths.example.yaml --profile local --validate-results`.
