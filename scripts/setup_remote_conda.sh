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

if ! "${CONDA_BIN}" env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  "${CONDA_BIN}" env create -f "${PROJECT_DIR}/environment.yml"
else
  echo "Conda env ${ENV_NAME} already exists; skipping creation."
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
  - Do not run delete commands under /limx_embop/tos/.
  - Dataset placement or syncing is intentionally left manual.
EOF
