"""ppclip vision — 素材视觉分析入口，统一委托到 analyzer 双通道管线"""

from __future__ import annotations

from pathlib import Path

from .config import ApiConfig, PathsConfig, Tier


def run_vision(
    project_dir: Path,
    tier: Tier,
    api: ApiConfig,
    paths: PathsConfig,
    *,
    verbose: bool = True,
    force: bool = False,
) -> int:
    from .analyzer import run_analysis
    return run_analysis(project_dir, tier, api, paths, verbose=verbose, force=force)
