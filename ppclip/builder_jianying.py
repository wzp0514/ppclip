"""ppclip builder_jianying — 基于 jianying-editor-skill 的剪映草稿生成器。

读取 enhanced_timeline.json → 调用 JyProject API → 生成剪映草稿。
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional

from .config import BuildConfig


def _setup_skill_path(skill_root: str) -> None:
    scripts_dir = os.path.join(skill_root, "scripts")
    if not os.path.isdir(scripts_dir):
        raise FileNotFoundError(f"jianying_editor skill scripts not found: {scripts_dir}")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    os.environ.setdefault("JY_SKILL_ROOT", skill_root)


def build(
    project_dir: Path,
    build_cfg: BuildConfig,
    *,
    skill_root: Optional[str] = None,
    verbose: bool = True,
) -> Optional[Path]:
    enhanced_path = project_dir / "enhanced_timeline.json"
    if not enhanced_path.exists():
        if verbose:
            print("  enhanced_timeline.json 不存在")
        return None

    with open(enhanced_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if skill_root:
        _setup_skill_path(skill_root)

    try:
        from jy_wrapper import JyProject
    except ImportError:
        if verbose:
            print("  [降级] jianying-editor-skill 不可用，使用简化草稿生成")
        return _build_fallback(project_dir, data, build_cfg, verbose)

    shots = data.get("shots", [])
    if not shots:
        return None

    if verbose:
        print("ppclip build (jianying)")

    project_name = f"ppclip_{project_dir.name}"
    try:
        proj = JyProject(project_name, width=build_cfg.canvas_width, height=build_cfg.canvas_height, overwrite=True)
    except Exception as e:
        if verbose:
            print(f"  [降级] JyProject 初始化失败: {e}")
        return _build_fallback(project_dir, data, build_cfg, verbose)

    # ── Intro ──
    intro = data.get("intro", {})
    if intro.get("enabled"):
        try:
            text = intro.get("text", "") or project_dir.name
            dur = intro.get("duration", 2.0)
            proj.add_text_simple(text, start_time="0s", duration=f"{dur}s", anim_in="淡入")
        except Exception as e:
            if verbose:
                print(f"  [warn] intro 失败: {e}")

    # ── BGM ──
    bgm = data.get("bgm", {})
    if bgm.get("cloud_music_id"):
        try:
            seg = proj.add_cloud_music(
                bgm["cloud_music_id"],
                start_time="0s",
                name=bgm.get("name", ""),
                duration_s=None,
                track_name="BGM",
            )
            if seg:
                seg.volume = bgm.get("volume", 0.4)
        except Exception as e:
            if verbose:
                print(f"  [warn] BGM 添加失败: {e}")

    # ── Shots ──
    tts_enabled = data.get("tts", {}).get("enabled", False)
    tts_speaker = data.get("tts", {}).get("speaker", build_cfg.default_speaker)

    timeline_cursor = 0.0  # Track timeline position for effect placement
    for shot in shots:
        clip_id = shot.get("clip_id", "")
        if not clip_id:
            continue

        # Resolve material file path
        file_path = _find_clip_path(project_dir, clip_id)
        if not file_path:
            if verbose:
                print(f"  [skip] 镜 {shot['id']}: 素材未找到 {clip_id}")
            continue

        src_start = shot.get("src_start", 0)
        src_end = shot.get("src_end", 0)
        if src_end <= src_start:
            continue
        clip_dur = src_end - src_start

        # Add video clip
        try:
            seg = proj.add_clip(
                file_path,
                source_start=f"{src_start}s",
                duration=f"{clip_dur}s",
                track_name="主视频",
            )
        except Exception as e:
            if verbose:
                print(f"  [warn] 镜 {shot['id']} 添加素材失败: {e}")
            continue

        # Transition
        trans = shot.get("transition", "")
        trans_dur = shot.get("transition_duration", 0.3)
        if trans and trans not in ("cut", "fade_in") and seg:
            try:
                proj.add_transition_simple(trans, video_segment=seg, duration=f"{trans_dur}s")
            except Exception as e:
                if verbose:
                    print(f"  [warn] 镜 {shot['id']} 转场失败: {e}")

        # Effect
        effect = shot.get("effect")
        if effect and seg:
            try:
                proj.add_effect_simple(
                    effect,
                    start_time=f"{timeline_cursor}s",
                    duration=f"{clip_dur}s",
                    track_name="EffectTrack",
                )
            except Exception as e:
                if verbose:
                    print(f"  [warn] 镜 {shot['id']} 特效失败: {e}")

        timeline_cursor += clip_dur

        # Subtitle / TTS
        subtitle = shot.get("subtitle", "")
        if subtitle:
            try:
                if tts_enabled:
                    proj.add_narrated_subtitles(subtitle, speaker=tts_speaker)
                else:
                    text_anim = shot.get("text_anim", {})
                    anim_in = text_anim.get("in_anim", "")
                    proj.add_text_simple(
                        subtitle,
                        duration=f"{clip_dur}s",
                        track_name="字幕",
                        anim_in=anim_in,
                    )
            except Exception as e:
                if verbose:
                    print(f"  [warn] 镜 {shot['id']} 字幕失败: {e}")
                # Fallback: plain text
                try:
                    proj.add_text_simple(subtitle, duration=f"{clip_dur}s", track_name="字幕")
                except Exception as fe:
                    if verbose:
                        print(f"  [warn] 镜 {shot['id']} 字幕降级也失败: {fe}")

        # Keyframes
        kf = shot.get("keyframes", {})
        if seg and kf.get("scale_start") and kf.get("scale_end"):
            scale_s = float(kf["scale_start"])
            scale_e = float(kf["scale_end"])
            if scale_s != scale_e:
                try:
                    from pyJianYingDraft import KeyframeProperty as KP, Keyframe

                    t_start = 0
                    t_end = int(clip_dur * 1_000_000)
                    easing = Keyframe.EASE_IN_OUT if kf.get("easing") == "ease_in_out" else {}
                    seg.add_keyframe(KP.uniform_scale, t_start, scale_s, **easing)
                    seg.add_keyframe(KP.uniform_scale, t_end, scale_e, **easing)
                except Exception as ke:
                    if verbose:
                        print(f"  [warn] 镜 {shot['id']} 关键帧失败: {ke}")

    # ── Outro ──
    outro = data.get("outro", {})
    if outro.get("enabled"):
        try:
            dur = outro.get("duration", 3.0)
            text = outro.get("text", "感谢观看")
            proj.add_text_simple(text, duration=f"{dur}s", track_name="字幕", anim_in="淡入")
        except Exception:
            pass

    # Save
    try:
        proj.save()
    except Exception as e:
        if verbose:
            print(f"  [warn] save 失败: {e}")

    draft_dir = Path(proj.root) / proj.name
    if verbose:
        print(f"  草稿已生成: {draft_dir}")

    return draft_dir


def _find_clip_path(project_dir: Path, clip_id: str) -> Optional[str]:
    materials_path = project_dir / "materials.json"
    if not materials_path.exists():
        return None
    try:
        with open(materials_path, "r", encoding="utf-8") as f:
            mats = json.load(f)
        for mf in mats.get("files", []):
            for clip in mf.get("clips", []):
                if clip["id"] == clip_id:
                    path = mf["path"]
                    if os.path.exists(path):
                        return path
    except Exception:
        pass
    return None


def _build_fallback(project_dir: Path, data: dict, build_cfg: BuildConfig, verbose: bool) -> Optional[Path]:
    """降级模式: pyJianYingDraft 草稿生成 → 文本剪辑指导（三级保底）。"""
    shots = data.get("shots", [])
    if not shots:
        return None

    try:
        from pyJianYingDraft import draft as dr
    except ImportError:
        if verbose:
            print("  pyJianYingDraft 不可用，生成文本剪辑指导")
        return _write_text_guide(project_dir, data, verbose)

    draft_name = f"ppclip_{project_dir.name}"
    try:
        from utils.formatters import get_default_drafts_root
        drafts_root = os.environ.get("JIANYING_DRAFT_DIR") or get_default_drafts_root()
    except Exception:
        drafts_root = os.environ.get("JIANYING_DRAFT_DIR", str(project_dir / "draft"))
    draft_dir = Path(drafts_root) / draft_name
    draft_dir.mkdir(parents=True, exist_ok=True)

    script = dr.ScriptFile(build_cfg.canvas_width, build_cfg.canvas_height, build_cfg.fps, True)
    script.add_track(dr.TrackType.video, "主视频")
    script.add_track(dr.TrackType.text, "字幕")

    cursor = 0.0
    for shot in shots:
        clip_id = shot.get("clip_id", "")
        file_path = _find_clip_path(project_dir, clip_id)
        if not file_path:
            continue

        src_start = shot.get("src_start", 0)
        src_end = shot.get("src_end", 0)
        if src_end <= src_start:
            continue

        dur_us = int((src_end - src_start) * 1_000_000)
        tgt_start = int(cursor * 1_000_000)

        vseg = dr.VideoSegment(
            file_path,
            dr.trange(tgt_start, dur_us),
            source_timerange=dr.trange(int(src_start * 1_000_000), dur_us),
        )
        script.add_segment(vseg, "主视频")

        subtitle = shot.get("subtitle", "")
        if subtitle:
            tseg = dr.TextSegment(
                subtitle,
                dr.trange(tgt_start, dur_us),
                style=dr.TextStyle(size=build_cfg.subtitle_font_size, bold=build_cfg.subtitle_bold),
            )
            script.add_segment(tseg, "字幕")

        cursor += src_end - src_start

    content_path = draft_dir / "draft_content.json"
    script.dump(str(content_path))

    if verbose:
        print(f"  [降级] 简化草稿已生成: {draft_dir}")

    return draft_dir


def _write_text_guide(project_dir: Path, data: dict, verbose: bool) -> Optional[Path]:
    """三级保底：生成文本剪辑指导 .md。"""
    shots = data.get("shots", [])
    bgm = data.get("bgm", {})
    tts = data.get("tts", {})

    lines = [
        f"# 剪辑指导 — {data.get('project', project_dir.name)}",
        "",
        f"> BGM: {bgm.get('name', '(无)')}（ID: {bgm.get('cloud_music_id', '-')}）",
        f"> TTS: {'启用' if tts.get('enabled') else '禁用'}（{tts.get('speaker', '-')}）",
        "",
        "| 镜号 | 时长 | 素材片段 | 源起止 | 字幕 | 转场 | 特效 | 角色 |",
        "|------|------|---------|--------|------|------|------|------|",
    ]

    for s in shots:
        cid = s.get("clip_id", "-")
        dur = s.get("duration", 0)
        ss = s.get("src_start", 0)
        se = s.get("src_end", 0)
        subtitle = s.get("subtitle", "")
        trans = s.get("transition", "")
        effect = s.get("effect", "") or "-"
        role = s.get("clip_role", "") or "-"

        lines.append(
            f"| {s.get('id', '?')} | {dur:.0f}s | {cid} | "
            f"{ss:.1f}-{se:.1f} | {subtitle} | {trans} | {effect} | {role} |"
        )

    lines.append("")
    lines.append("## 手动操作步骤")
    lines.append("1. 打开剪映专业版")
    lines.append("2. 新建草稿，设置分辨率和帧率")
    lines.append("3. 按上表导入素材、调整起止位置")
    lines.append("4. 添加字幕文本")
    lines.append(f"5. 如有 BGM ID，在剪映音频→音乐→搜索「{bgm.get('name', '')}」")

    guide_path = project_dir / "剪辑指导.md"
    guide_path.write_text("\n".join(lines), encoding="utf-8")
    if verbose:
        print(f"  [保底] 文本剪辑指导已生成: {guide_path}")
    return None  # 无草稿目录，但指导文件可用
