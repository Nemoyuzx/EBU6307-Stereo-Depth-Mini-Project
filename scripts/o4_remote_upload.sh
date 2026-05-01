#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/o4_remote_common.sh"

show_help() {
  cat <<EOF
Usage: bash scripts/o4_remote_upload.sh

Packages the local source tree and uploads it to the configured remote O4 workspace.

What gets uploaded:
  - environment.yml
  - pyproject.toml
  - README.md
  - codes/
  - configs/
  - docs/
  - scripts/

What is intentionally excluded:
  - .git/
  - results/
  - results_backup_*/
  - workspace/data/
  - workspace/cache/
  - workspace/checkpoints/
  - workspace/external/

EOF
  print_common_overrides
}

if [[ "${1:-}" == "--help" ]]; then
  show_help
  exit 0
fi

ensure_local_dirs

local_archive="${LOCAL_TMP_ROOT}/o4_upload_$(timestamp).tgz"
remote_archive="/tmp/$(basename "${local_archive}")"

(
  cd "${REPO_ROOT}"
  COPYFILE_DISABLE=1 COPY_EXTENDED_ATTRIBUTES_DISABLE=1 tar -czf "${local_archive}" \
    --no-mac-metadata \
    --no-xattrs \
    --no-fflags \
    --no-acls \
    --exclude='.git' \
    --exclude='results' \
    --exclude='results_backup_*' \
    --exclude='workspace/data' \
    --exclude='workspace/cache' \
    --exclude='workspace/checkpoints' \
    --exclude='workspace/external' \
    environment.yml \
    pyproject.toml \
    README.md \
    codes \
    configs \
    docs \
    scripts
)

scp_to_remote "${local_archive}" "${REMOTE_HOST}:${remote_archive}"

ssh_remote bash -s -- "${REMOTE_PROJECT_DIR}" "${remote_archive}" <<'REMOTE'
set -euo pipefail

project_dir="$1"
archive_path="$2"

mkdir -p "${project_dir}"
mkdir -p "${project_dir}/results/logs"
mkdir -p "${project_dir}/results/tmp"
mkdir -p "${project_dir}/workspace/data"
mkdir -p "${project_dir}/workspace/cache"

tar -xzf "${archive_path}" -C "${project_dir}"
rm -f "${archive_path}"
REMOTE

echo "Uploaded source archive to ${REMOTE_HOST}:${REMOTE_PROJECT_DIR}"
echo "Local archive kept at ${local_archive}"