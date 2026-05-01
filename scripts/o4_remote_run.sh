#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/o4_remote_common.sh"

show_help() {
  cat <<EOF
Usage: bash scripts/o4_remote_run.sh [--no-clean] [--log-name NAME] [-- O4_ARGS...]

Runs codes/o4.py on the configured remote machine.

Wrapper options:
  --no-clean       Keep existing remote O4 result folders before running.
  --log-name NAME  Write remote log to results/logs/NAME.
  --help           Show this message.

Everything after -- is passed directly to codes/o4.py.

Examples:
  bash scripts/o4_remote_run.sh
  bash scripts/o4_remote_run.sh -- --scene-name artroom1
  bash scripts/o4_remote_run.sh --log-name o4_artroom1.log -- --scene-name artroom1
  bash scripts/o4_remote_run.sh -- --dry-run --max-scenes 0
  bash scripts/o4_remote_run.sh --no-clean -- --validate-results

EOF
  print_common_overrides
}

clean_results=1
log_name="${O4_REMOTE_LOG}"
remote_args=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --help)
      show_help
      exit 0
      ;;
    --no-clean)
      clean_results=0
      shift
      ;;
    --log-name)
      if [[ $# -lt 2 ]]; then
        echo "--log-name requires a value" >&2
        exit 2
      fi
      log_name="$2"
      shift 2
      ;;
    --)
      shift
      remote_args+=("$@")
      break
      ;;
    *)
      remote_args+=("$1")
      shift
      ;;
  esac
done

if ((${#remote_args[@]} > 0)); then
  for arg in "${remote_args[@]}"; do
    if [[ "${arg}" == "--dry-run" || "${arg}" == "--validate-results" ]]; then
      clean_results=0
    fi
  done
fi

run_remote_o4() {
  ssh_remote bash -s -- \
    "${REMOTE_PROJECT_DIR}" \
    "${REMOTE_PYTHON}" \
    "${REMOTE_CONFIG_PATH}" \
    "${REMOTE_PROFILE}" \
    "${log_name}" \
    "${clean_results}" \
    "$@" <<'REMOTE'
set -euo pipefail

project_dir="$1"
python_bin="$2"
config_path="$3"
profile_name="$4"
log_name="$5"
clean_flag="$6"
shift 6

cd "${project_dir}" || exit 1
mkdir -p results/logs

if [[ "${clean_flag}" == "1" ]]; then
  rm -rf \
    results/O4a_transformer \
    results/O4b_transformer \
    results/O4c_transformer \
    results/O4a_transformer_dinov2 \
    results/O4b_transformer_dinov2 \
    results/O4c_transformer_dinov2
fi

PYTHONPATH=codes "${python_bin}" codes/o4.py --config "${config_path}" --profile "${profile_name}" "$@" \
  > "results/logs/${log_name}" 2>&1
rc=$?
tail -n 200 "results/logs/${log_name}" || true
exit "${rc}"
REMOTE
}

if ((${#remote_args[@]} > 0)); then
  run_remote_o4 "${remote_args[@]}"
else
  run_remote_o4
fi