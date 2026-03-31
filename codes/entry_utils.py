from __future__ import annotations

import argparse
from pathlib import Path
import sys

from cli import main as cli_main


def repo_root_from(module_file: str) -> Path:
    return Path(module_file).resolve().parents[1]


def build_objective_parser(objective: str, module_file: str) -> argparse.ArgumentParser:
    repo_root = repo_root_from(module_file)
    default_config = repo_root / 'configs' / 'dataset_paths.example.yaml'
    parser = argparse.ArgumentParser(
        description=f'Run objective {objective.upper()} from codes/ as the direct entry script.'
    )
    parser.add_argument('--config', type=Path, default=default_config, help=f'Path to YAML config. Default: {default_config}')
    parser.add_argument('--profile', default='local', help='Config profile name. Default: local')
    parser.add_argument('--max-scenes', type=int, default=None, help='Maximum number of scenes to process. Use 0 for discovery-only.')
    parser.add_argument('--scene-name', default=None, help='Run only one scene by exact directory name.')
    parser.add_argument('--dry-run', action='store_true', help='Print discovery/config info without writing outputs.')
    parser.add_argument('--validate-results', action='store_true', help='Validate existing outputs without rewriting them.')
    parser.add_argument('--o4-execution-mode', choices=('baseline', 'dinov2_cost_volume'), default=None)
    parser.add_argument('--o4-dinov2-model', default=None)
    parser.add_argument('--o4-dinov2-checkpoint', type=Path, default=None)
    parser.add_argument('--o4-dinov2-repo', type=Path, default=None)
    parser.add_argument('--o4-regression-mode', choices=('quadratic', 'soft_argmax'), default=None)
    return parser


def run_objective_entry(objective: str, module_file: str, argv: list[str] | None = None) -> int:
    parser = build_objective_parser(objective, module_file)
    args = parser.parse_args(argv)

    repo_root = repo_root_from(module_file)
    codes_dir = Path(module_file).resolve().parent
    original_cwd = Path.cwd()

    cli_argv = [
        'cli.py',
        '--config', str(args.config),
        '--profile', args.profile,
        '--objective', objective,
    ]
    if args.max_scenes is not None:
        cli_argv.extend(['--max-scenes', str(args.max_scenes)])
    if args.scene_name:
        cli_argv.extend(['--scene-name', args.scene_name])
    if args.dry_run:
        cli_argv.append('--dry-run')
    if args.validate_results:
        cli_argv.append('--validate-results')
    if args.o4_execution_mode is not None:
        cli_argv.extend(['--o4-execution-mode', args.o4_execution_mode])
    if args.o4_dinov2_model is not None:
        cli_argv.extend(['--o4-dinov2-model', args.o4_dinov2_model])
    if args.o4_dinov2_checkpoint is not None:
        cli_argv.extend(['--o4-dinov2-checkpoint', str(args.o4_dinov2_checkpoint)])
    if args.o4_dinov2_repo is not None:
        cli_argv.extend(['--o4-dinov2-repo', str(args.o4_dinov2_repo)])
    if args.o4_regression_mode is not None:
        cli_argv.extend(['--o4-regression-mode', args.o4_regression_mode])

    if str(codes_dir) not in sys.path:
        sys.path.insert(0, str(codes_dir))

    previous_argv = sys.argv[:]
    try:
        sys.argv = cli_argv
        # Make config relative paths stable whether invoked from repo root or codes/.
        import os
        os.chdir(repo_root)
        return cli_main()
    finally:
        sys.argv = previous_argv
        os.chdir(original_cwd)
