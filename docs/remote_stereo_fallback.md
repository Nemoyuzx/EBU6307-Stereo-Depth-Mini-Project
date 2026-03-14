# Remote Stereo Fallback for O1 Validation

We found remote stereo-video datasets such as `/limx_embop/tos/limx_data/community_dataset_v1/Chojins/chess_game_001_blue_stereo` and `/limx_embop/tos/limx_data/community_dataset_v1/Chojins/chess_game_001_red_stereo`.

Their `meta/info.json` indicates stereo image streams like `observation.images.image` and `observation.images.image2`, so they are potentially useful as a temporary engineering input source.

## Why this is not the assignment dataset format

These paths are not Middlebury-style scene folders. They do not provide the expected per-scene files such as `im0.png`, `im1.png`, `calib.txt`, and `disp0.pfm`, so they must not be treated as final assignment-ready stereo data.

## Temporary fallback use

For O1 pipeline validation only, one practical fallback is:

- extract one synchronized left/right frame pair from the remote stereo video
- place that pair into a clearly temporary synthetic scene folder
- generate any temporary synthetic artifacts needed for engineering checks
- use the result only to validate basic pipeline wiring, file I/O, and result writing

## Separation rule

Keep this fallback strictly separate from final assignment datasets and results. Temporary scenes, configs, and outputs derived from these remote videos should be labeled clearly and should not be mixed with Middlebury-format inputs or final reported metrics.
