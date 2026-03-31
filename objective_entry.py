from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "dataset_paths.example.yaml"
DEFAULT_PROFILE = os.environ.get("EBU6307_PROFILE", "local")
PROJECT_VENV_PYTHON = REPO_ROOT / ".venv" / "bin" / "python"


def build_parser(objective: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            f"Run objective {objective.upper()} through the packaged ebu6307_stereo CLI "
            "with a direct top-level terminal entry script."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help=f"Path to YAML config. Default: {DEFAULT_CONFIG}")
    parser.add_argument("--profile", default=DEFAULT_PROFILE, help=f"Config profile name. Default: {DEFAULT_PROFILE}")
    parser.add_argument("--max-scenes", type=int, default=None, help="Maximum number of scenes to process. Use 0 for discovery-only.")
    parser.add_argument("--scene-name", default=None, help="Run only one scene by exact directory name.")
    parser.add_argument("--dry-run", action="store_true", help="Print discovery/config info without writing outputs.")
    parser.add_argument("--validate-results", action="store_true", help="Validate existing outputs without rewriting them.")
    parser.add_argument("extra", nargs=argparse.REMAINDER, help="Extra args forwarded to the underlying CLI after '--'.")
    return parser


def run_objective(objective: str, argv: list[str] | None = None) -> int:
    args = build_parser(objective).parse_args(argv)

    python_executable = str(PROJECT_VENV_PYTHON) if PROJECT_VENV_PYTHON.exists() else sys.executable
    command = [
        python_executable,
        "-m",
        "ebu6307_stereo",
        "--config",
        str(args.config),
        "--profile",
        args.profile,
        "--objective",
        objective,
    ]
    if args.max_scenes is not None:
        command.extend(["--max-scenes", str(args.max_scenes)])
    if args.scene_name:
        command.extend(["--scene-name", args.scene_name])
    if args.dry_run:
        command.append("--dry-run")
    if args.validate_results:
        command.append("--validate-results")

    forwarded = list(args.extra)
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]
    command.extend(forwarded)

    env = os.environ.copy()
    src_path = REPO_ROOT / "codes" / "src"
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(src_path) if not existing_pythonpath else f"{src_path}{os.pathsep}{existing_pythonpath}"

    print("Running:", " ".join(command))
    completed = subprocess.run(command, cwd=REPO_ROOT, env=env)
    return completed.returncode
