#!/usr/bin/env bash

set -euo pipefail

CONDA_BIN="/root/miniconda3/bin/conda"
REMOTE_BASE="/root/code/new_folder"
PROJECT_NAME="openclaw-operate"
PROJECT_DIR="${REMOTE_BASE}/${PROJECT_NAME}"
ENV_NAME="ebu6307-stereo"

if [[ ! -x "${CONDA_BIN}" ]]; then
  echo "Conda not found at ${CONDA_BIN}" >&2
  exit 1
fi

mkdir -p "${REMOTE_BASE}"
mkdir -p "${PROJECT_DIR}"
mkdir -p "${PROJECT_DIR}/codes/src/ebu6307_stereo"
mkdir -p "${PROJECT_DIR}/codes/tests"
mkdir -p "${PROJECT_DIR}/docs"
mkdir -p "${PROJECT_DIR}/scripts"
mkdir -p "${PROJECT_DIR}/results/O1a_synthetic_data"
mkdir -p "${PROJECT_DIR}/results/O1b_synthetic_data"
mkdir -p "${PROJECT_DIR}/results/O1c_synthetic_data"
mkdir -p "${PROJECT_DIR}/results/O2a_sift"
mkdir -p "${PROJECT_DIR}/results/O2b_sift"
mkdir -p "${PROJECT_DIR}/results/O2c_sift"
mkdir -p "${PROJECT_DIR}/results/O3a_disparity"
mkdir -p "${PROJECT_DIR}/results/O3b_disparity"
mkdir -p "${PROJECT_DIR}/results/O3c_disparity"
mkdir -p "${PROJECT_DIR}/results/O4a_transformer"
mkdir -p "${PROJECT_DIR}/results/O4b_transformer"
mkdir -p "${PROJECT_DIR}/results/O4c_transformer"
mkdir -p "${PROJECT_DIR}/results/logs"
mkdir -p "${PROJECT_DIR}/results/tmp"
mkdir -p "${PROJECT_DIR}/workspace/data"
mkdir -p "${PROJECT_DIR}/workspace/checkpoints"
mkdir -p "${PROJECT_DIR}/workspace/cache"
mkdir -p "${PROJECT_DIR}/workspace/external"

export HF_ENDPOINT="https://hf-mirror.com"

if ! "${CONDA_BIN}" env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  "${CONDA_BIN}" env create -f "${PROJECT_DIR}/environment.yml"
else
  echo "Conda env ${ENV_NAME} already exists; skipping creation."
fi

if ! "${CONDA_BIN}" run -n "${ENV_NAME}" python -c "import torch" >/dev/null 2>&1; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    echo "Attempting CUDA-enabled PyTorch install for ${ENV_NAME}."
    "${CONDA_BIN}" run -n "${ENV_NAME}" python -m pip install \
      "torch>=2.4,<2.7" \
      --index-url https://download.pytorch.org/whl/cu121 \
      || echo "CUDA PyTorch install failed; O4 will fall back to numpy until torch is installed."
  else
    echo "No NVIDIA runtime detected; attempting CPU PyTorch install for ${ENV_NAME}."
    "${CONDA_BIN}" run -n "${ENV_NAME}" python -m pip install \
      "torch>=2.4,<2.7" \
      || echo "CPU PyTorch install failed; O4 will fall back to numpy until torch is installed."
  fi
else
  echo "PyTorch already available in ${ENV_NAME}; skipping torch install."
fi

if ! "${CONDA_BIN}" run -n "${ENV_NAME}" python -c "import transformers, huggingface_hub" >/dev/null 2>&1; then
  echo "Installing Hugging Face tooling for the O4 DINOv2 transformers path."
  "${CONDA_BIN}" run -n "${ENV_NAME}" python -m pip install \
    "transformers>=4.45,<5" \
    "safetensors>=0.4,<1" \
    "huggingface_hub>=0.34,<1" \
    || echo "Hugging Face tooling install failed; the O4 DINOv2 path will remain unavailable until installed."
else
  echo "Hugging Face tooling already available in ${ENV_NAME}; skipping install."
fi

"${CONDA_BIN}" run -n "${ENV_NAME}" python -m pip install -e "${PROJECT_DIR}"

cat <<EOF
Remote bootstrap complete.

Project directory: ${PROJECT_DIR}
Conda env: ${ENV_NAME}

Activate with:
  source /root/miniconda3/bin/activate ${ENV_NAME}

Notes:
  - Keep GPU memory usage under 16 GB for O4 experiments.
  - The O4 path now prefers torch+CUDA automatically when available.
  - HF_ENDPOINT is set to ${HF_ENDPOINT} for mirrored Hugging Face access.
  - Use execution_mode=dinov2_cost_volume with dinov2_model_name=facebook/dinov2-base and dinov2_checkpoint_path=/limx_embop/tos/users/Nemo/self-work/models/dinov2_vitb14_reg4_pretrain.pth for the active O4 descriptor path.
  - Do not run delete commands under /limx_embop/tos/.
  - Dataset placement or syncing is intentionally left manual.
EOF
