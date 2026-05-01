#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/o4_remote_common.sh"

show_help() {
  cat <<EOF
Usage: bash scripts/o4_remote_download.sh [--log-name NAME]

Packages remote O4 results, downloads them locally, and extracts them into results/.

Wrapper options:
  --log-name NAME  Include results/logs/NAME in the downloaded archive.
  --help           Show this message.

EOF
  print_common_overrides
}

log_name="${O4_REMOTE_LOG}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --help)
      show_help
      exit 0
      ;;
    --log-name)
      if [[ $# -lt 2 ]]; then
        echo "--log-name requires a value" >&2
        exit 2
      fi
      log_name="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1" >&2
      show_help >&2
      exit 2
      ;;
  esac
done

ensure_local_dirs

archive_stem="o4_results_${log_name%.log}_$(timestamp)"
remote_archive="/tmp/${archive_stem}.tgz"
local_archive="${LOCAL_TMP_ROOT}/${archive_stem}.tgz"

ssh_remote bash -s -- "${REMOTE_PROJECT_DIR}" "${remote_archive}" "${log_name}" <<'REMOTE'
set -euo pipefail

project_dir="$1"
archive_path="$2"
log_name="$3"

cd "${project_dir}" || exit 1
items=()
for path in \
  results/O4a_transformer \
  results/O4b_transformer \
  results/O4c_transformer \
  results/O4a_transformer_dinov2 \
  results/O4b_transformer_dinov2 \
  results/O4c_transformer_dinov2 \
  "results/logs/${log_name}"
do
  if [[ -e "${path}" ]]; then
    items+=("${path}")
  fi
done
if ((${#items[@]} == 0)); then
  echo "No remote result paths found to download." >&2
  exit 1
fi
tar -czf "${archive_path}" "${items[@]}"
REMOTE

scp_from_remote "${REMOTE_HOST}:${remote_archive}" "${local_archive}"

dirs_to_replace=()
add_dir_to_replace() {
  local dir="$1"
  local existing
  if ((${#dirs_to_replace[@]} > 0)); then
    for existing in "${dirs_to_replace[@]}"; do
      if [[ "${existing}" == "${dir}" ]]; then
        return
      fi
    done
  fi
  dirs_to_replace+=("${dir}")
}

while IFS= read -r archive_entry; do
  archive_entry="${archive_entry%/}"
  case "${archive_entry}" in
    results/O4a_transformer|results/O4a_transformer/*) add_dir_to_replace "results/O4a_transformer" ;;
    results/O4b_transformer|results/O4b_transformer/*) add_dir_to_replace "results/O4b_transformer" ;;
    results/O4c_transformer|results/O4c_transformer/*) add_dir_to_replace "results/O4c_transformer" ;;
    results/O4a_transformer_dinov2|results/O4a_transformer_dinov2/*) add_dir_to_replace "results/O4a_transformer_dinov2" ;;
    results/O4b_transformer_dinov2|results/O4b_transformer_dinov2/*) add_dir_to_replace "results/O4b_transformer_dinov2" ;;
    results/O4c_transformer_dinov2|results/O4c_transformer_dinov2/*) add_dir_to_replace "results/O4c_transformer_dinov2" ;;
  esac
done < <(tar -tzf "${local_archive}")

if ((${#dirs_to_replace[@]} > 0)); then
  for result_dir in "${dirs_to_replace[@]}"; do
    rm -rf "${REPO_ROOT}/${result_dir}"
  done
fi

tar -xzf "${local_archive}" -C "${REPO_ROOT}"
restore_o4_gitkeeps
ssh_remote rm -f "${remote_archive}"

echo "Downloaded and extracted O4 results to ${REPO_ROOT}/results"
echo "Local archive kept at ${local_archive}"