"""ppclip indexer — 素材扫描 → 元数据 → 场景检测 → 缩略图 → materials.json"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .audio import analyze_audio, has_meaningful_audio
from .config import PathsConfig, Tier
from .scenedetect import (
    MEDIA_EXTENSIONS,
    ClipSegment,
    _get_duration_ffmpeg,
    detect_scenes,
    extract_keyframes,
    scan_media_files,
)


@dataclass
class FileMetadata:
    path: str
    duration: float = 0.0
    width: int = 0
    height: int = 0
    fps: float = 0.0
    codec: str = ""


@dataclass
class ClipInfo:
    id: str
    start: float
    end: float
    duration: float
    media_type: str = "video"  # "video" | "audio"
    thumbnail: str = ""
    vision: Optional[dict] = None
    audio_events: Optional[list[dict]] = None


@dataclass
class IndexedFile:
    file: str
    path: str
    duration: float
    width: int
    height: int
    fps: float
    file_md5: str = ""
    media_type: str = "video"  # "video" | "audio"
    clips: list[ClipInfo] = field(default_factory=list)


def _read_metadata_pymediainfo(filepath: Path) -> FileMetadata:
    try:
        from pymediainfo import MediaInfo
    except ImportError:
        raise RuntimeError("pymediainfo not installed")

    info = MediaInfo.parse(str(filepath))
    meta = FileMetadata(path=str(filepath))
    for track in info.tracks:
        if track.track_type == "Video":
            meta.width = track.width or 0
            meta.height = track.height or 0
            meta.fps = float(track.frame_rate) if track.frame_rate else 0.0
            meta.codec = track.codec_id or ""
        if track.track_type == "General":
            meta.duration = float(track.duration) / 1000.0 if track.duration else 0.0
    return meta


def _read_metadata_ffprobe(filepath: Path) -> FileMetadata:
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_format", "-show_streams", str(filepath),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    data = json.loads(result.stdout)
    meta = FileMetadata(path=str(filepath))
    for stream in data.get("streams", []):
        if stream.get("codec_type") == "video":
            meta.width = stream.get("width", 0)
            meta.height = stream.get("height", 0)
            fps_str = stream.get("r_frame_rate", "0/1")
            try:
                num, den = fps_str.split("/")
                meta.fps = float(num) / float(den) if float(den) != 0 else 0.0
            except (ValueError, ZeroDivisionError):
                meta.fps = 0.0
    fmt = data.get("format", {})
    meta.duration = float(fmt.get("duration", 0))
    return meta


def _read_metadata_basic(filepath: Path) -> FileMetadata:
    meta = FileMetadata(path=str(filepath))
    meta.duration = _get_duration_ffmpeg(filepath)
    return meta


def read_metadata(filepath: Path, verbose: bool = True) -> FileMetadata:
    # Tier 1: ffprobe (fast, instant for any file size)
    try:
        meta = _read_metadata_ffprobe(filepath)
        if meta.duration > 0 and meta.width > 0:
            return meta
    except Exception:
        pass
    # Tier 2: pymediainfo (slow for large files, but gets all metadata)
    try:
        meta = _read_metadata_pymediainfo(filepath)
        if meta.duration > 0:
            return meta
    except Exception:
        pass
    # Tier 3: basic (ffmpeg duration only)
    if verbose:
        print(f"    [WARN] ffprobe/pymediainfo unavailable, duration estimate only")
    return _read_metadata_basic(filepath)


def _filter_segments(segments: list[ClipSegment], min_duration: float = 2.0) -> list[ClipSegment]:
    return [s for s in segments if s.duration >= min_duration]


def _file_md5(filepath: Path) -> str:
    """Compute MD5 hash of a file. Used for cache invalidation."""
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def run_index(
    material_dir: Path,
    project_dir: Path,
    tier: Tier,
    paths: PathsConfig,
    *,
    verbose: bool = True,
    force: bool = False,
) -> Path:
    idx = tier.index
    threshold = idx.scene_threshold
    min_clip = idx.min_clip_duration
    frame_interval = idx.frame_interval
    quality = idx.vision_quality
    thumb_w = idx.thumbnail_width
    skip_scene = idx.skip_scene_detect
    skip_thumbs = idx.skip_thumbnails
    max_size_mb = idx.max_file_size_mb
    ffmpeg = paths.ffmpeg_path or "ffmpeg"

    if verbose:
        print(f"ppclip index  {material_dir}")
        print(f"  场景检测: threshold={threshold}, min_clip={min_clip}s")
        print(f"  关键帧: interval={frame_interval}s, quality={quality}")
        print()

    video_files, audio_files = scan_media_files(material_dir)

    # Read prior materials.json for cache lookup (MD5→existing IndexedFile)
    use_cache = tier.features.use_cache and not force
    prior_md5_map: dict[str, dict] = {}  # file_path → {md5, index_entry}
    cache_hit_count = 0
    cache_miss_count = 0
    cache_invalidated = False
    if use_cache:
        prior_path = project_dir / "materials.json"
        if prior_path.exists():
            try:
                prior = json.loads(prior_path.read_text(encoding="utf-8"))
                # Check cache_info threshold mismatch → invalidate all
                prior_cache = prior.get("cache_info", {})
                if prior_cache:
                    if (prior_cache.get("scene_threshold") != threshold or
                        prior_cache.get("min_clip_duration") != min_clip or
                        prior_cache.get("frame_interval") != frame_interval):
                        if verbose:
                            print(f"  [CACHE] 阈值已变 → 全部失效重建")
                        cache_invalidated = True
                    else:
                        for pf in prior.get("files", []):
                            if pf.get("file_md5") and pf.get("clips"):
                                prior_md5_map[pf["path"]] = {"md5": pf["file_md5"], "entry": pf}
                else:
                    # Legacy cache without cache_info
                    for pf in prior.get("files", []):
                        if pf.get("file_md5") and pf.get("clips"):
                            prior_md5_map[pf["path"]] = {"md5": pf["file_md5"], "entry": pf}
            except (json.JSONDecodeError, OSError):
                pass

    thumbs_dir = project_dir / "thumbs"
    thumbs_dir.mkdir(parents=True, exist_ok=True)

    indexed: list[IndexedFile] = []
    total_clips = 0

    # Process video files
    for fpath in video_files:
        size_mb = fpath.stat().st_size / (1024 * 1024)
        if max_size_mb > 0 and size_mb > max_size_mb:
            if verbose:
                print(f"  [SKIP] {fpath.name} ({size_mb:.0f}MB > {max_size_mb}MB limit)")
            continue

        fmd5 = _file_md5(fpath)
        fpath_str = str(fpath)

        # MD5 cache hit: reuse existing clips
        if fpath_str in prior_md5_map and prior_md5_map[fpath_str]["md5"] == fmd5:
            cached = prior_md5_map[fpath_str]["entry"]
            cache_hit_count += 1
            if verbose:
                print(f"  [CACHE] {fpath.name} (MD5 unchanged, {len(cached['clips'])} clips)")
            from .scenedetect import ClipSegment as CS
            clips = []
            for cc in cached["clips"]:
                clips.append(ClipInfo(
                    id=cc["id"], start=cc["start"], end=cc["end"],
                    duration=cc["duration"],
                    media_type=cc.get("media_type", "video"),
                    thumbnail=cc.get("thumbnail", ""),
                    audio_events=cc.get("audio_events"),
                ))
            indexed.append(IndexedFile(
                file=fpath.name, path=fpath_str,
                duration=cached["duration"], width=cached.get("width", 0),
                height=cached.get("height", 0), fps=cached.get("fps", 0),
                file_md5=fmd5, clips=clips,
            ))
            total_clips += len(clips)
            continue

        cache_miss_count += 1
        if verbose:
            print(f"  [VIDEO] {fpath.name}")

        meta = read_metadata(fpath, verbose=verbose)
        if verbose and meta.width:
            print(f"    元数据: {meta.width}x{meta.height}, {meta.duration:.1f}s, {meta.fps:.1f}fps")

        # Scene detection
        if skip_scene:
            segments = [ClipSegment(start=0.0, end=meta.duration, duration=meta.duration)]
            if verbose:
                print(f"    场景检测: skip（整文件=1片段）")
        else:
            segments, method = detect_scenes(fpath, threshold=threshold, ffmpeg_path=ffmpeg, verbose=verbose)
            segments = _filter_segments(segments, min_clip)
            if not segments:
                if verbose:
                    print(f"    场景检测: 0 个切点，自动切为每{frame_interval}s一段")
                from .scenedetect import detect_scenes_fixed
                segments = detect_scenes_fixed(fpath, frame_interval)

        # Audio analysis
        audio_events_raw: list[dict] = []
        if idx.audio_enabled:
            try:
                silence_db = idx.silence_threshold_db
                silence_min = idx.silence_min_duration
                events = analyze_audio(fpath, ffmpeg_path=ffmpeg,
                                       silence_db=silence_db, silence_min=silence_min,
                                       verbose=verbose)
                audio_events_raw = [
                    {"start": e.start, "end": e.end, "type": e.type, "db": e.db}
                    for e in events
                ]
                if verbose and audio_events_raw:
                    print(f"    音频: {len(audio_events_raw)} 静音段")
            except Exception:
                if verbose:
                    print(f"    音频: 分析失败，跳过")

        # Keyframes
        if skip_thumbs:
            thumbs = {}
            if verbose:
                print(f"    缩略图: skip")
        else:
            thumbs = extract_keyframes(
                fpath, segments, thumbs_dir,
                frame_interval=frame_interval, thumbnail_width=thumb_w,
                quality=quality, ffmpeg_path=ffmpeg,
            )

        clips: list[ClipInfo] = []
        for seg in segments:
            basename = _sanitize_basename(fpath.stem)
            clip_id = f"{basename}_{len(clips) + 1:03d}"
            thumb_rel = ""
            if clip_id in thumbs and thumbs[clip_id]:
                thumb_abs = thumbs[clip_id][0]
                try:
                    thumb_rel = str(thumb_abs.relative_to(project_dir)).replace("\\", "/")
                except ValueError:
                    thumb_rel = str(thumb_abs).replace("\\", "/")
            clips.append(ClipInfo(
                id=clip_id, start=seg.start, end=seg.end, duration=seg.duration,
                thumbnail=thumb_rel,
                audio_events=audio_events_raw if audio_events_raw else None,
            ))

        indexed.append(IndexedFile(
            file=fpath.name, path=str(fpath),
            duration=meta.duration, width=meta.width, height=meta.height, fps=meta.fps,
            file_md5=fmd5, clips=clips,
        ))
        total_clips += len(clips)

    # Process audio files (whole file = 1 clip, no scene detection)
    for fpath in audio_files:
        fmd5 = _file_md5(fpath)
        fpath_str = str(fpath)
        # MD5 cache for audio
        if fpath_str in prior_md5_map and prior_md5_map[fpath_str]["md5"] == fmd5:
            cached = prior_md5_map[fpath_str]["entry"]
            cache_hit_count += 1
            if verbose:
                print(f"  [CACHE] {fpath.name} (MD5 unchanged)")
            clips = [ClipInfo(
                id=cached["clips"][0]["id"], start=0.0,
                end=cached["duration"], duration=cached["duration"],
                media_type="audio",
            )]
            indexed.append(IndexedFile(
                file=fpath.name, path=fpath_str,
                duration=cached["duration"], width=0, height=0, fps=0,
                file_md5=fmd5, media_type="audio", clips=clips,
            ))
            total_clips += 1
            continue
        cache_miss_count += 1
        if verbose:
            print(f"  [AUDIO] {fpath.name}")
        meta = read_metadata(fpath, verbose=verbose)
        if verbose:
            print(f"    时长: {meta.duration:.1f}s")
        clips = [ClipInfo(
            id=_sanitize_basename(fpath.stem) + "_001",
            start=0.0, end=meta.duration, duration=meta.duration,
            media_type="audio",
        )]
        indexed.append(IndexedFile(
            file=fpath.name, path=str(fpath),
            duration=meta.duration, width=0, height=0, fps=0,
            file_md5=fmd5, media_type="audio", clips=clips,
        ))
        total_clips += 1

    cache_info = {
        "scene_threshold": threshold,
        "min_clip_duration": min_clip,
        "frame_interval": frame_interval,
    }

    if not indexed:
        print("  未找到媒体文件")
        return _write_materials(project_dir, [], False, "", None, cache_info=cache_info)

    vision_provider = ""  # will be set after vision step
    materials_path = _write_materials(project_dir, indexed, False, vision_provider, None, cache_info=cache_info)

    if verbose:
        print()
        print(f"素材索引完成：")
        print(f"  视频: {len(video_files)} 个 → {total_clips} 片段")
        print(f"  音频: {len(audio_files)} 个（{', '.join(a.name for a in audio_files) if audio_files else '无'}）")
        if use_cache:
            total_files = cache_hit_count + cache_miss_count
            print(f"  缓存: {cache_hit_count}/{total_files} 命中，{cache_miss_count} 重建" + (" (阈值变化失效)" if cache_invalidated else ""))
        print(f"  成片目录: 已自动排除")
        print(f"  Vision: 待分析")
        print(f"  输出: {materials_path}")

    return materials_path


def _write_materials(
    project_dir: Path,
    indexed: list[IndexedFile],
    vision_available: bool,
    vision_provider: str,
    vision_failures: Optional[int],
    cache_info: Optional[dict] = None,
) -> Path:
    output = {
        "version": 1,
        "created": datetime.now(timezone.utc).isoformat(),
        "vision_available": vision_available,
        "vision_provider": vision_provider,
        "files": [
            {
                "file": f.file,
                "path": f.path,
                "duration": f.duration,
                "width": f.width,
                "height": f.height,
                "fps": f.fps,
                "file_md5": f.file_md5,
                "media_type": f.media_type,
                "clips": [
                    {
                        "id": c.id,
                        "start": c.start,
                        "end": c.end,
                        "duration": c.duration,
                        "media_type": c.media_type,
                        "thumbnail": c.thumbnail,
                        "vision": c.vision,
                        "audio_events": c.audio_events,
                    }
                    for c in f.clips
                ],
            }
            for f in indexed
        ],
    }
    if cache_info:
        output["cache_info"] = cache_info
    path = project_dir / "materials.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    return path


def _sanitize_basename(name: str) -> str:
    result = "".join(c if c.isalnum() or c in "._-" else "_" for c in name)
    return result[:20]
