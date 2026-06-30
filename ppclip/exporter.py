"""ppclip exporter — 剪映自动导出封装。

条件：Windows + Jianying v5.9 以下（uiautomation 方案）。
"""

from __future__ import annotations

import os
import sys
from typing import Optional


def run_export(
    draft_name: str,
    output_path: str,
    *,
    resolution: str = "1080",
    fps: int = 30,
    skill_root: Optional[str] = None,
    verbose: bool = True,
) -> bool:
    if sys.platform != "win32":
        if verbose:
            print("  [skip] 自动导出仅支持 Windows，请在剪映中手动打开草稿并导出")
        return False

    _setup_skill_path(skill_root)

    try:
        from auto_exporter import auto_export
    except ImportError:
        if verbose:
            print("  [skip] auto_exporter 不可用，请在剪映中手动导出")
        return False

    try:
        if verbose:
            print(f"ppclip export  \"{draft_name}\" -> {output_path}")
        auto_export(draft_name, output_path, resolution=resolution, framerate=str(fps))
        if verbose:
            print(f"  导出完成: {output_path}")
        return True
    except Exception as e:
        if verbose:
            print(f"  导出失败: {e}，请在剪映中手动导出")
        return False


def _setup_skill_path(skill_root: Optional[str]) -> None:
    if not skill_root:
        return
    scripts_dir = os.path.join(skill_root, "scripts")
    if not os.path.isdir(scripts_dir):
        return
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    os.environ.setdefault("JY_SKILL_ROOT", skill_root)
