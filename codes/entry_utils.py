from __future__ import annotations

import argparse
from pathlib import Path
import sys
import os

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
        '--config', str(args.config),  # 指定 YAML 配置文件路径，里面定义数据集和输出目录等路径配置。
        '--profile', args.profile,  # 选择配置文件中的 profile，决定当前运行采用哪组路径与环境设置。
        '--objective', objective,  # 告诉统一 CLI 当前要执行的是哪个目标模块，例如 O1/O2/O3/O4。
    ]
    if args.max_scenes is not None:
        cli_argv.extend(['--max-scenes', str(args.max_scenes)])  # 限制本次最多处理多少个 scene，传 0 时只做发现不实际处理
    if args.scene_name:
        cli_argv.extend(['--scene-name', args.scene_name])  # 只运行名字完全匹配的单个 scene
    if args.dry_run:
        cli_argv.append('--dry-run')  # 只打印发现结果和配置，不写任何输出文件
    if args.validate_results:
        cli_argv.append('--validate-results')  # 仅检查已有结果是否齐全有效，不重新计算
    if args.o4_execution_mode is not None:
        cli_argv.extend(['--o4-execution-mode', args.o4_execution_mode])  # 选择 O4 的执行模式，例如 baseline 或 dinov2_cost_volume
    if args.o4_dinov2_model is not None:
        cli_argv.extend(['--o4-dinov2-model', args.o4_dinov2_model])  # 指定 O4 使用的 DINOv2 模型名称
    if args.o4_dinov2_checkpoint is not None:
        cli_argv.extend(['--o4-dinov2-checkpoint', str(args.o4_dinov2_checkpoint)])  # 指定 O4 使用的 DINOv2 权重文件路径
    if args.o4_dinov2_repo is not None:
        cli_argv.extend(['--o4-dinov2-repo', str(args.o4_dinov2_repo)])  # 指定 O4 加载 DINOv2 代码或仓库的本地路径
    if args.o4_regression_mode is not None:
        cli_argv.extend(['--o4-regression-mode', args.o4_regression_mode])  # 选择 O4 回归视差时采用的输出方式

    if str(codes_dir) not in sys.path:
        sys.path.insert(0, str(codes_dir))

    previous_argv = sys.argv[:]
    try:
        sys.argv = cli_argv
        # Make config relative paths stable whether invoked from repo root or codes/.
        os.chdir(repo_root)
        return cli_main()
    finally:
        sys.argv = previous_argv
        os.chdir(original_cwd)
