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

O3 使用 SIFT+SGM 混合路径：先用 manual SIFT 描述符做 ratio/mutual/stereo geometry 筛选，记录可靠立体特征支持并生成 SIFT 种子视差先验；正式 `disp0.pfm` 再由 census+gradient 像素代价体和四方向 SGM 生成，并用 SIFT 匹配数量与种子先验调节稀疏/重复纹理场景的置信过滤和局部恢复。`disp0.png` 是 RGB 深度伪彩色预览，黑色表示无效视差。

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

`results/` 按 PDF 的 a/b/c 分工整理：
- `a` folders: pipeline / architecture assets only.
- `b` folders: scene-level result images, PFM outputs, and visual examples.
- `c` folders: metric CSV files and model/checkpoint summaries.
- O1 pipeline: `results/O1a_synthetic_data/`; synthetic scenes: `results/O1b_synthetic_data/`; metrics: `results/O1c_synthetic_data/SSIM.csv`
- O2 pipeline: `results/O2a_sift/`; keypoint/match examples: `results/O2b_sift/`; metrics: `results/O2c_sift/metrics.csv`
- O3 pipeline: `results/O3a_disparity/dep_pipeline.jpg`; disparity results/examples: `results/O3b_disparity/`; metrics: `results/O3c_disparity/metrics.csv`
- O4 pipeline: `results/O4a_transformer/`; disparity/raw/confidence/error results: `results/O4b_transformer/`; metrics and fold summary: `results/O4c_transformer/`

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

O4 仍保留两条运行路径：
- `codes/o4.py`：默认提交路径，只使用作业允许的 pip 依赖。
- `codes/o4_dinov2.py`：DINOv2 实验/备份路径，直接加载本地 DINOv2 模型和 checkpoint，不训练。

O4 的正式 `disp0.pfm` 由 Transformer/ViT token 视差直接生成，不再融合 O3/SGM 细节视差。每个 O4 场景目录同时写出 `im0.png`、`im1.png` 源图副本，以及 `disp0_transformer_raw.pfm` 和 `disp0_transformer_raw.png`，用于核验原始 Transformer/ViT 预测结果；`disp0.png` 只是展示预览。

例如，运行默认 O4 baseline：

```bash
python codes/o4.py --profile remote
```

运行 DINOv2 直接模型版本：

```bash
python codes/o4_dinov2.py \
  --profile remote \
  --scene-name chess3 \
  --o4-dinov2-model facebook/dinov2-base \
  --o4-dinov2-repo workspace/external/dinov2 \
  --o4-dinov2-checkpoint /limx_embop/tos/users/Nemo/self-work/models/dinov2_vitb14_reg4_pretrain.pth \
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

## O4 远端脚本

仓库现在提供一套 O4 远端运行脚本，按“上传代码 -> 远端运行 -> 下载结果”三步使用。

### 1. 上传代码到远端

```bash
bash scripts/o4_remote_upload.sh
```

默认上传到：`root@14.103.233.39:/root/code/new_folder/openclaw-operate`

### 2. 在远端运行 O4

完整运行：

```bash
bash scripts/o4_remote_run.sh
```

运行单个场景：

```bash
bash scripts/o4_remote_run.sh -- --scene-name artroom1
```

只做 dry-run：

```bash
bash scripts/o4_remote_run.sh -- --dry-run --max-scenes 0
```

### 3. 下载远端结果到本地

```bash
bash scripts/o4_remote_download.sh
```

会把远端 `results/O4a_transformer`、`results/O4b_transformer`、`results/O4c_transformer` 以及对应日志打包下载并解压到本地 `results/`。

### 可覆盖环境变量

如果需要改远端地址、端口、远端目录或 Python，可在命令前覆盖：

```bash
REMOTE_HOST=root@example.com \
REMOTE_PORT=22 \
REMOTE_PROJECT_DIR=/root/code/openclaw-operate \
REMOTE_PYTHON=/root/miniconda3/envs/ebu6307-whitelist/bin/python \
bash scripts/o4_remote_run.sh
```


## 记录

像素级极值 + 对比度阈值 + Hessian 边缘剔除  换为 3 维泰勒展开迭代定位 + 反向验证剔除 效果提升明显
关键点位置更准，导致可视化上的点更贴结构中心。
主方向更稳，因为方向直方图围绕的中心和尺度更准。
描述子更稳，因为采样窗口和梯度统计区域更准。
