"""ppclip — AI 辅助剪辑，素材 + 想法描述 → 剪映草稿"""

from __future__ import annotations

import json
import os
import shutil
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .config import (
    ApiConfig,
    BuildConfig,
    Config,
    PathsConfig,
    Tier,
    TIERS,
    load_config,
)
from .builder_jianying import build as build_jianying
from .builder_remotion import build as build_remotion
from .enrich import run_enrich
from .exporter import run_export
from .indexer import run_index
from .match import run_match
from .script import generate_script
from .vision import run_vision


@dataclass
class EnvStatus:
    python_ok: bool = True
    ffmpeg_ok: bool = False
    ffmpeg_path: str = ""
    ffmpeg_source: str = ""
    llm_ok: bool = False
    llm_source: str = ""
    vision_ok: bool = False
    vision_source: str = ""


@dataclass
class RunResult:
    project_dir: Path
    materials: Path | None = None
    script: Path | None = None
    timeline: Path | None = None
    enhanced_timeline: Path | None = None
    draft: Path | None = None
    remotion_project: Path | None = None
    env: EnvStatus = field(default_factory=EnvStatus)
    errors: list[str] = field(default_factory=list)


def run(
    *,
    material_dir: str,
    idea: str,
    tier: str,
    project_name: str | None = None,
    config_path: str | None = None,
) -> RunResult:
    if tier not in TIERS:
        raise ValueError(f"无效档位 '{tier}'，可选: {list(TIERS.keys())}")

    t: Tier = TIERS[tier]
    config = load_config(Path(config_path) if config_path else None)
    api = config.api
    paths = config.paths
    build = config.build
    errors: list[str] = []

    # 环境检测
    env = _detect_env(api, paths)

    # 项目目录
    name = project_name or "ppclip"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base = paths.output_base or str(Path(__file__).parent / "output")
    project_dir = Path(output_base) / f"{name}_{ts}"
    for sub in ["thumbs", "draft", "logs"]:
        (project_dir / sub).mkdir(parents=True, exist_ok=True)

    # project.json
    project_json = {
        "name": name,
        "tier": tier,
        "created": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(project_dir.resolve()),
        "current_step": None,
        "config": {
            "canvas_width": build.canvas_width,
            "canvas_height": build.canvas_height,
            "fps": build.fps,
        },
    }
    (project_dir / "project.json").write_text(
        json.dumps(project_json, indent=2, ensure_ascii=False), encoding="utf-8")

    result = RunResult(project_dir=project_dir, env=env)

    # 1. index
    material_path = Path(material_dir).resolve()
    if not material_path.exists():
        errors.append(f"素材目录不存在: {material_path}")
        result.errors = errors
        return result

    try:
        mats = run_index(material_path, project_dir, t, paths)
        result.materials = mats
    except Exception as e:
        errors.append(f"index 失败: {e}")
        result.errors = errors
        return result

    # 2. vision (if enabled)
    if t.features.use_vision and t.index.max_vision_calls > 0:
        try:
            run_vision(project_dir, t, api, paths)
        except Exception as e:
            errors.append(f"vision 失败: {e}")

    # 3. script
    try:
        script_result = generate_script(idea, project_dir, t, api)
        if script_result:
            result.script = project_dir / "script.json"
    except Exception as e:
        errors.append(f"script 失败: {e}")
        result.errors = errors
        return result

    # 4. match
    try:
        run_match(project_dir, t, api)
        result.timeline = project_dir / "timeline.json"
    except Exception as e:
        errors.append(f"match 失败: {e}")
        result.errors = errors
        return result

    # 5. enrich
    skill_data_dir = None
    if paths.jianying_skill_root:
        sd = Path(paths.jianying_skill_root) / "data"
        if sd.exists():
            skill_data_dir = sd

    try:
        run_enrich(project_dir, t, build, config.enrich, skill_data_dir=skill_data_dir)
        result.enhanced_timeline = project_dir / "enhanced_timeline.json"
    except Exception as e:
        errors.append(f"enrich 失败: {e}")

    # 6. build (Jianying)
    try:
        draft_dir = build_jianying(
            project_dir, build,
            skill_root=paths.jianying_skill_root or None,
        )
        if draft_dir:
            result.draft = draft_dir
    except Exception as e:
        errors.append(f"build 失败: {e}")

    # 7. export (optional — only if draft exists and not dev)
    if result.draft and t.name != "dev":
        try:
            draft_name = result.draft.name
            export_path = str(project_dir / f"{draft_name}.mp4")
            run_export(
                draft_name, export_path,
                resolution=str(build.canvas_height),
                fps=build.fps,
                skill_root=paths.jianying_skill_root or None,
            )
        except Exception as e:
            errors.append(f"export 失败: {e}")

    # *. Remotion (optional — parallel path for tutorial/visualization)
    if t.features.use_remotion and build.enable_remotion:
        try:
            remotion_dir = build_remotion(
                project_dir, build,
                skill_root=paths.remotion_skill_root or None,
            )
            if remotion_dir:
                result.remotion_project = remotion_dir
        except Exception as e:
            errors.append(f"remotion 失败: {e}")

    result.errors = errors
    return result


def _detect_env(api: ApiConfig, paths: PathsConfig) -> EnvStatus:
    env = EnvStatus()
    env.python_ok = sys.version_info >= (3, 9)

    ffmpeg_path, ffmpeg_source = _find_ffmpeg(paths)
    env.ffmpeg_ok = ffmpeg_path is not None
    env.ffmpeg_path = str(ffmpeg_path) if ffmpeg_path else ""
    env.ffmpeg_source = ffmpeg_source

    env.llm_ok = bool(api.text_key_1 or api.text_key_2 or api.text_key_3)
    env.vision_ok = bool(api.image_key_1 or api.image_key_2)

    if api.image_key_1:
        env.vision_source = "vision-1"
    elif api.image_key_2:
        env.vision_source = "vision-2"
    else:
        env.vision_source = "无"

    if api.text_key_1:
        env.llm_source = "text-1(可用)"
    if api.text_key_2:
        env.llm_source = f"{env.llm_source}, text-2(可用)" if env.llm_source else "text-2(可用)"
    if api.text_key_3:
        env.llm_source = f"{env.llm_source}, text-3(可用)" if env.llm_source else "text-3(可用)"
    if not env.llm_source:
        env.llm_source = "无"

    return env


def _find_ffmpeg(paths: PathsConfig) -> tuple[Path | None, str]:
    if paths.ffmpeg_path and Path(paths.ffmpeg_path).exists():
        return Path(paths.ffmpeg_path), "config"

    jy_candidates = [
        Path(os.environ.get("LOCALAPPDATA", "")) / "JianyingPro",
        Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "JianyingPro",
        Path("D:/剪映/"),
    ]
    for base in jy_candidates:
        if not base.exists():
            continue
        for root, _dirs, files in os.walk(base):
            if "ffmpeg.exe" in files:
                p = Path(root) / "ffmpeg.exe"
                return p, f"剪映({p})"

    found = shutil.which("ffmpeg")
    if found:
        return Path(found), "PATH"

    return None, ""
