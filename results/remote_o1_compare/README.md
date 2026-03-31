# O1 Superpixel Experimental Comparison

## 说明
本文件用于单独记录 O1 特殊算法（superpixel + clustering + DIBR 路线）与原始平移算法（shift baseline）的远端对比结果，证明该特殊算法在已跑通的样本上具有提升效果。

同时，项目主线代码已经切回原始 O1 平移算法配置；特殊算法实现被独立保存到：

- `codes/src/ebu6307_stereo/o1_superpixel_experimental.py`

主线 O1 文件保持为：

- `codes/src/ebu6307_stereo/o1.py`

## 对比范围
当前已拿到 6 个共同场景的可比结果：

- artroom1
- artroom2
- bandsaw1
- bandsaw2
- chess1
- chess2

## 平均 SSIM 对比
- shift baseline: `0.763061`
- superpixel experimental: `0.834115`
- average delta: `+0.071054`

## 各场景结果
| scene | shift_ssim | superpixel_ssim | delta |
|---|---:|---:|---:|
| artroom1 | 0.710478 | 0.802200 | +0.091722 |
| artroom2 | 0.714150 | 0.817863 | +0.103713 |
| bandsaw1 | 0.784735 | 0.839723 | +0.054988 |
| bandsaw2 | 0.845678 | 0.902889 | +0.057211 |
| chess1 | 0.758246 | 0.821682 | +0.063436 |
| chess2 | 0.765077 | 0.820333 | +0.055256 |

## 结论
在当前已经完成对比的 6 个场景上，特殊算法的 SSIM 全部高于原始 shift baseline，没有出现下降样本。

因此，至少在这批已完成样本上，可以认为该特殊算法相较原始平移算法有明显提升。

## 备注
- 这份 README 只用于记录提升结果，不改变主线提交路径。
- 当前主线 O1 已恢复为原始 shift baseline。
- 特殊算法后续若继续研究，应通过独立文件维护，而不是继续混入主线 O1。

## 当前本地 results 收口确认（2026-03-31）
- 标准 O1 提交产物仍保留在 `results/O1a_synthetic_data`、`results/O1b_synthetic_data`、`results/O1c_synthetic_data`。
- 本地核对到 `results/O1a_synthetic_data/syn_pipeline.jpg`、`results/O1c_synthetic_data/SSIM.csv` 仍在；`results/O1b_synthetic_data/` 下各场景输出目录仍独立存在。
- O1 新思路/实验性对比结果单独放在 `results/remote_o1_compare`，当前包含本 README 与 `SSIM_shift_baseline.csv`。
- 结论：本地标准 results 没有被 O1 新思路覆盖；实验结果是独立保存的。
