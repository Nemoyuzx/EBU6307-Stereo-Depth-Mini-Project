# EBU6307 Stereo Depth Mini Project

本仓库已整理为课程提交友好的扁平结构：`codes/` 就是程序执行目录，`results/` 保持原有结果接口不变。

## Layout

```text
.
├── codes/                 # 主线代码与直接入口（o1.py ~ o4.py）
├── configs/               # 示例路径配置
├── docs/                  # 说明文档
├── results/               # Assignment outputs for O1 -> O4
├── scripts/               # 环境/辅助脚本
└── README.md
```

## 直接运行方式

### 从项目根目录运行

```bash
python codes/o1.py --dry-run --max-scenes 0
python codes/o2.py --dry-run --max-scenes 0
python codes/o3.py --dry-run --max-scenes 0
python codes/o4.py --dry-run --max-scenes 0
```

### 进入 `codes/` 目录运行

```bash
cd codes
python o1.py --dry-run --max-scenes 0
python o2.py --dry-run --max-scenes 0
python o3.py --dry-run --max-scenes 0
python o4.py --dry-run --max-scenes 0
```

这四个入口默认读取 `configs/dataset_paths.example.yaml`、使用 `local` profile，并支持常用参数：
- `--config`
- `--profile`
- `--scene-name`
- `--max-scenes`
- `--dry-run`
- `--validate-results`
- O4 专用覆盖参数（`--o4-execution-mode` 等）

## 常见示例

### O1

```bash
python codes/o1.py --dry-run
python codes/o1.py --validate-results
```

### O2

```bash
python codes/o2.py --dry-run
python codes/o2.py --validate-results
```

### O3

```bash
python codes/o3.py --dry-run
python codes/o3.py --validate-results
```

### O4

```bash
python codes/o4.py --dry-run
python codes/o4.py --validate-results
```

指定单个场景：

```bash
python codes/o1.py --scene-name artroom1 --dry-run
```

## 结果目录接口

本轮整理**没有修改** `results/` 的既有输出接口：
- O1 synthetic scenes: `results/O1b_synthetic_data/`
- O1 metrics: `results/O1c_synthetic_data/SSIM.csv`
- O2 outputs: `results/O2a_sift/`, `results/O2b_sift/`, `results/O2c_sift/metrics.csv`
- O3 outputs: `results/O3a_disparity/`, `results/O3b_disparity/`, `results/O3c_disparity/metrics.csv`
- O4 outputs: `results/O4a_transformer/`, `results/O4b_transformer/`, `results/O4c_transformer/metrics.csv`
- O4 fold summary: `results/O4c_transformer/fold_summary.csv`

## 数据与配置

配置文件见：`configs/dataset_paths.example.yaml`

Middlebury 场景布局说明见：`docs/dataset_paths.md`

## 备用工具

独立 fallback extractor 现在也从 `codes/` 执行目录直接调用：

```bash
python codes/fallback_extract.py --left-video ... --right-video ... --output-dir workspace/data/tmp_remote_stereo_fallback
```

或：

```bash
cd codes
python fallback_extract.py --left-video ... --right-video ... --output-dir ../workspace/data/tmp_remote_stereo_fallback
```

## O4 说明

O4 仍保留两种执行模式：
- `baseline`：默认提交路径
- `dinov2_cost_volume`：实验/备份路径

例如：

```bash
python codes/o4.py \
  --profile remote \
  --scene-name chess3 \
  --o4-execution-mode dinov2_cost_volume \
  --o4-dinov2-model facebook/dinov2-base \
  --o4-dinov2-repo workspace/external/dinov2 \
  --o4-dinov2-checkpoint workspace/checkpoints/dinov2_vitb14_reg4_pretrain.pth \
  --o4-regression-mode quadratic
```

## 环境准备

```bash
conda env create -f environment.yml
conda activate ebu6307-stereo
```

如需远端初始化，可继续使用：

```bash
bash scripts/setup_remote_conda.sh
```


## 记录

像素级极值 + 对比度阈值 + Hessian 边缘剔除  换为 3 维泰勒展开迭代定位 + 反向验证剔除 效果提升明显
关键点位置更准，导致可视化上的点更贴结构中心。
主方向更稳，因为方向直方图围绕的中心和尺度更准。
描述子更稳，因为采样窗口和梯度统计区域更准。
