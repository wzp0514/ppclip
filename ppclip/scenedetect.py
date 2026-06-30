"""ppclip scene detection — 4 级降级: FFmpeg scdet → PySceneDetect → 固定间隔 → 整文件"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Fix Windows encoding: both stdout and subprocess need UTF-8
if os.name == "nt":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

MEDIA_EXTENSIONS = {
    ".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv", ".wmv",
    ".m4v", ".3gp", ".ts", ".mts", ".m2ts",
}

AUDIO_EXTENSIONS = {".wav", ".mp3", ".aac", ".ogg", ".flac", ".m4a", ".wma"}

EXCLUDE_NAME_KEYWORDS = ["成片", "output", "export", "成品", "导出"]


def _run_ffmpeg(ffmpeg_path: str, args: list[str], timeout: int = 300) -> subprocess.CompletedProcess:
    """Centralized FFmpeg subprocess runner. Handles Windows encoding properly."""
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    return subprocess.run(
        [ffmpeg_path] + args,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        env=env,
    )


def _is_excluded(path: Path) -> bool:
    name = path.name.lower()
    # Check file/dir name
    if any(kw in name for kw in EXCLUDE_NAME_KEYWORDS):
        return True
    # Check parent dirs
    for parent in path.parents:
        if any(kw in parent.name.lower() for kw in EXCLUDE_NAME_KEYWORDS):
            return True
    return False


@dataclass
class ClipSegment:
    start: float
    end: float
    duration: float


def scan_media_files(directory: Path) -> tuple[list[Path], list[Path]]:
    """Returns (video_files, audio_files). Excludes files/dirs matching keywords."""
    videos: list[Path] = []
    audios: list[Path] = []
    for root, _dirs, filenames in os.walk(directory):
        if _is_excluded(Path(root)):
            continue
        for f in filenames:
            p = Path(root) / f
            if _is_excluded(p):
                continue
            ext = p.suffix.lower()
            if ext in MEDIA_EXTENSIONS:
                videos.append(p)
            elif ext in AUDIO_EXTENSIONS:
                audios.append(p)
    return sorted(videos), sorted(audios)


def _parse_scdet_output(stderr: str, file_duration: float) -> list[ClipSegment]:
    """Parse FFmpeg scdet filter stderr output for scene change timestamps."""
    cuts: list[float] = [0.0]
    for line in stderr.splitlines():
        if "lavfi.scene_score" in line or "scene_score" in line:
            pass  # info line
        # Extract timestamp from lines like: "frame:123 pts:456 pts_time:4.5"
        if "pts_time:" in line:
            try:
                parts = line.split("pts_time:")
                if len(parts) > 1:
                    t = float(parts[1].split()[0])
                    if t > 0.1:  # skip cuts too close to start
                        cuts.append(t)
            except (ValueError, IndexError):
                continue
    cuts.append(file_duration)
    cuts = sorted(set(cuts))

    segments: list[ClipSegment] = []
    for i in range(len(cuts) - 1):
        duration = cuts[i + 1] - cuts[i]
        if duration > 0.5:  # skip sub-second fragments
            segments.append(ClipSegment(start=cuts[i], end=cuts[i + 1], duration=duration))
    return segments


def detect_scenes_ffmpeg(
    filepath: Path,
    threshold: float = 0.3,
    ffmpeg_path: str = "ffmpeg",
) -> list[ClipSegment]:
    result = _run_ffmpeg(ffmpeg_path, [
        "-i", str(filepath),
        "-vf", f"scdet=threshold={threshold}",
        "-f", "null", "-",
    ], timeout=300)
    duration = _get_duration_ffmpeg(filepath, ffmpeg_path)
    return _parse_scdet_output(result.stderr, duration)


def detect_scenes_pyscenedetect(
    filepath: Path,
    threshold: float = 0.3,
) -> list[ClipSegment]:
    try:
        from scenedetect import open_video, SceneManager, ContentDetector
    except ImportError:
        raise RuntimeError("PySceneDetect not installed")

    video = open_video(str(filepath))
    manager = SceneManager()
    manager.add_detector(ContentDetector(threshold=threshold * 100))
    manager.detect_scenes(video)
    scenes = manager.get_scene_list()

    if not scenes:
        return []

    return [
        ClipSegment(start=s[0].get_seconds(), end=s[1].get_seconds(),
                     duration=s[1].get_seconds() - s[0].get_seconds())
        for s in scenes
    ]


def detect_scenes_fixed(
    filepath: Path,
    interval: float = 10.0,
) -> list[ClipSegment]:
    duration = _get_duration_ffmpeg(filepath)
    if duration <= 0:
        return [ClipSegment(start=0.0, end=10.0, duration=10.0)]
    segments: list[ClipSegment] = []
    t = 0.0
    while t < duration:
        end = min(t + interval, duration)
        segments.append(ClipSegment(start=t, end=end, duration=end - t))
        t = end
    return segments


def _get_duration_ffmpeg(filepath: Path, ffmpeg_path: str = "ffmpeg") -> float:
    try:
        result = _run_ffmpeg(ffmpeg_path, ["-i", str(filepath), "-f", "null", "-"], timeout=60)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        size_mb = filepath.stat().st_size / (1024 * 1024)
        return max(size_mb * 0.5, 10.0)
    for line in result.stderr.splitlines():
        if "Duration" in line:
            try:
                ts = line.split("Duration:")[1].split(",")[0].strip()
                parts = ts.split(":")
                return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
            except (ValueError, IndexError):
                pass
    return 0.0


def extract_keyframes(
    filepath: Path,
    segments: list[ClipSegment],
    output_dir: Path,
    *,
    frame_interval: float = 15.0,
    thumbnail_width: int = 480,
    quality: str = "medium",
    ffmpeg_path: str = "ffmpeg",
) -> dict[str, list[Path]]:
    """Extract keyframe thumbnails. Returns {clip_id: [thumb_paths]}.
    If ffmpeg is unavailable, returns empty dict (no thumbnails)."""
    # Check ffmpeg availability
    try:
        _run_ffmpeg(ffmpeg_path, ["-version"], timeout=5)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return {}

    thumbs: dict[str, list[Path]] = {}
    basename = _sanitize_basename(filepath.stem)

    for idx, seg in enumerate(segments):
        clip_id = f"{basename}_{idx + 1:03d}"
        timestamps = _pick_timestamps(seg, quality, frame_interval)
        thumb_paths: list[Path] = []
        for ti, ts in enumerate(timestamps):
            suffix = f"_{ti + 1}.jpg" if len(timestamps) > 1 else ".jpg"
            out = output_dir / f"{clip_id}{suffix}"
            try:
                _run_ffmpeg(ffmpeg_path, [
                    "-ss", str(ts), "-i", str(filepath),
                    "-vframes", "1", "-vf", f"scale={thumbnail_width}:-1",
                    "-y", str(out),
                ], timeout=30)
            except Exception:
                pass
            if out.exists():
                thumb_paths.append(out)
        thumbs[clip_id] = thumb_paths

    return thumbs


def _pick_timestamps(seg: ClipSegment, quality: str, frame_interval: float) -> list[float]:
    dur = seg.duration
    if quality == "low" or dur < frame_interval * 2:
        return [seg.start + dur * 0.33]  # avoid mid-point bias
    elif quality == "medium":
        return [seg.start + dur * 0.25, seg.start + dur * 0.5, seg.start + dur * 0.75]
    else:  # high
        ts = []
        t = seg.start + frame_interval / 2
        while t < seg.end:
            ts.append(t)
            t += frame_interval
        return ts if ts else [seg.start + dur * 0.5]


def _sanitize_basename(name: str) -> str:
    result = "".join(c if c.isalnum() or c in "._-" else "_" for c in name)
    return result[:20]


def dedup_frames(frames: list[Path], threshold: int = 8) -> list[Path]:
    """dHash 感知去重 — 移除视觉上近乎相同的帧（来自 Vidlizer dedup.py，MIT）。"""
    if threshold <= 0 or len(frames) <= 1:
        return frames
    try:
        import fitz
    except ImportError:
        return frames

    def _dhash(path: Path, size: int = 8) -> int:
        doc = fitz.open(str(path))
        page = doc[0]
        w = max(page.rect.width, 1)
        h = max(page.rect.height, 1)
        mat = fitz.Matrix((size + 1) / w, size / h)
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csGRAY)
        doc.close()
        s = pix.samples
        bits = [
            s[row * (size + 1) + col] > s[row * (size + 1) + col + 1]
            for row in range(size)
            for col in range(size)
        ]
        result = 0
        for b in bits:
            result = (result << 1) | b
        return result

    def _hamming(a: int, b: int) -> int:
        return bin(a ^ b).count("1")

    kept: list[Path] = []
    hashes: list[int] = []
    for f in frames:
        try:
            h = _dhash(f)
        except Exception:
            kept.append(f)
            continue
        if not hashes or all(_hamming(h, prev) >= threshold for prev in hashes):
            kept.append(f)
            hashes.append(h)
    return kept


def detect_scenes(
    filepath: Path,
    threshold: float = 0.3,
    fixed_interval: float = 10.0,
    ffmpeg_path: Optional[str] = None,
    verbose: bool = True,
) -> tuple[list[ClipSegment], str]:
    """Main entry: try FFmpeg scdet → PySceneDetect → fixed interval.
    Returns (segments, method_used)."""
    ff = ffmpeg_path or "ffmpeg"

    # Tier 1: FFmpeg scdet
    try:
        segs = detect_scenes_ffmpeg(filepath, threshold, ff)
        if len(segs) >= 1:
            if verbose:
                print(f"    场景检测: FFmpeg scdet → {len(segs)} 个片段")
            return segs, "ffmpeg_scdet"
    except Exception:
        pass

    # Tier 2: PySceneDetect
    try:
        segs = detect_scenes_pyscenedetect(filepath, threshold)
        if len(segs) >= 1:
            if verbose:
                print(f"    场景检测: PySceneDetect → {len(segs)} 个片段")
            return segs, "pyscenedetect"
    except Exception:
        pass

    # Tier 3: Fixed interval
    try:
        segs = detect_scenes_fixed(filepath, fixed_interval)
        if len(segs) >= 1:
            if verbose:
                print(f"    场景检测: 固定间隔({fixed_interval}s) → {len(segs)} 个片段")
            return segs, "fixed"
    except Exception:
        pass

    # Ultimate fallback: whole file as one segment
    duration = _get_duration_ffmpeg(filepath, ff)
    if duration <= 0:
        duration = max(filepath.stat().st_size / (1024 * 1024) * 0.5, 30.0)
    segs = [ClipSegment(start=0.0, end=duration, duration=duration)]
    if verbose:
        print(f"    场景检测: 整文件单片段 → {duration:.0f}s")
    return segs, "whole_file"
