#!/usr/bin/env bash

set -euo pipefail

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "This file is meant to be sourced by other scripts." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

REMOTE_HOST="${REMOTE_HOST:-root@14.103.233.39}"
REMOTE_PORT="${REMOTE_PORT:-40043}"
REMOTE_PROJECT_DIR="${REMOTE_PROJECT_DIR:-/root/code/new_folder/openclaw-operate}"
REMOTE_PYTHON="${REMOTE_PYTHON:-/root/miniconda3/envs/ebu6307-whitelist/bin/python}"
REMOTE_CONFIG_PATH="${REMOTE_CONFIG_PATH:-configs/dataset_paths.example.yaml}"
REMOTE_PROFILE="${REMOTE_PROFILE:-remote}"
O4_REMOTE_LOG="${O4_REMOTE_LOG:-o4_remote.log}"
LOCAL_RESULTS_ROOT="${LOCAL_RESULTS_ROOT:-${REPO_ROOT}/results}"
LOCAL_TMP_ROOT="${LOCAL_TMP_ROOT:-${LOCAL_RESULTS_ROOT}/tmp}"

timestamp() {
  date +"%Y%m%d-%H%M%S"
}

ssh_remote() {
  /usr/bin/ssh -p "${REMOTE_PORT}" "${REMOTE_HOST}" "$@"
}

scp_to_remote() {
  scp -P "${REMOTE_PORT}" "$@"
}

scp_from_remote() {
  scp -P "${REMOTE_PORT}" "$@"
}

ensure_local_dirs() {
  mkdir -p "${LOCAL_RESULTS_ROOT}"
  mkdir -p "${LOCAL_TMP_ROOT}"
}

restore_o4_gitkeeps() {
  mkdir -p "${REPO_ROOT}/results/O4a_transformer"
  mkdir -p "${REPO_ROOT}/results/O4b_transformer"
  mkdir -p "${REPO_ROOT}/results/O4c_transformer"
  printf '\n' > "${REPO_ROOT}/results/O4a_transformer/.gitkeep"
  printf '\n' > "${REPO_ROOT}/results/O4b_transformer/.gitkeep"
  printf '\n' > "${REPO_ROOT}/results/O4c_transformer/.gitkeep"
}

print_common_overrides() {
  cat <<EOF
Environment overrides:
  REMOTE_HOST=${REMOTE_HOST}
  REMOTE_PORT=${REMOTE_PORT}
  REMOTE_PROJECT_DIR=${REMOTE_PROJECT_DIR}
  REMOTE_PYTHON=${REMOTE_PYTHON}
  REMOTE_CONFIG_PATH=${REMOTE_CONFIG_PATH}
  REMOTE_PROFILE=${REMOTE_PROFILE}
  O4_REMOTE_LOG=${O4_REMOTE_LOG}
EOF
}