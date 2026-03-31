"""EBU6307 stereo depth mini-project bootstrap package."""

__all__ = ["main"]


def main() -> int:
    """延迟导入 CLI 主入口，避免仅导入包时就触发较重的子模块初始化。"""
    # 这里保留函数内导入：避免 `import ebu6307_stereo` 时立刻级联导入 CLI 及其依赖。
    from .cli import main as cli_main

    return cli_main()
