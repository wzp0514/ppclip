"""ppclip audio — FFmpeg 音频分析: 静音检测 + 音量统计"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class AudioEvent:
    start: float
    end: float
    type: str  # "silence" | "loud_peak"
    db: float


def _run_ffmpeg(ffmpeg_path: str, args: list[str], timeout: int = 300) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env.pop("PYTHONIOENCODING", None)
    return subprocess.run(
        [ffmpeg_path, *args],
        capture_output=True, text=True,
        encoding="utf-8", errors="replace",
        timeout=timeout, env=env,
    )


def analyze_audio(
    filepath: Path,
    ffmpeg_path: str = "ffmpeg",
    silence_db: float = -40,
    silence_min: float = 0.5,
    *,
    verbose: bool = False,
) -> list[AudioEvent]:
    """FFmpeg silencedetect + volumedetect. Returns sorted silence events."""
    events: list[AudioEvent] = []

    # Silence detection
    try:
        r = _run_ffmpeg(ffmpeg_path, [
            "-i", str(filepath),
            "-af", f"silencedetect=n={silence_db}dB:d={silence_min}",
            "-f", "null", "-",
        ])
        silence_start = None
        for line in r.stderr.split("\n"):
            if "silence_start:" in line:
                silence_start = float(line.split("silence_start:")[1].strip())
            elif "silence_end:" in line and silence_start is not None:
                end = float(line.split("silence_end:")[1].split("|")[0].strip())
                events.append(AudioEvent(
                    start=round(silence_start, 2), end=round(end, 2),
                    type="silence", db=silence_db,
                ))
                silence_start = None
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        if verbose:
            print("    [WARN] 静音检测失败，跳过")

    # Volume statistics
    try:
        r2 = _run_ffmpeg(ffmpeg_path, ["-i", str(filepath), "-af", "volumedetect", "-f", "null", "-"])
        for line in r2.stderr.split("\n"):
            if "mean_volume:" in line and verbose:
                print(f"    平均音量: {line.split('mean_volume:')[1].strip()}")
            if "max_volume:" in line and verbose:
                print(f"    最大音量: {line.split('max_volume:')[1].strip()}")
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        pass

    events.sort(key=lambda e: e.start)
    return events


def has_meaningful_audio(events: list[AudioEvent], total_duration: float) -> bool:
    """Returns True if the clip has non-trivial audio (not 100% silence)."""
    if not events:
        return True  # Unknown → assume has audio
    silence_dur = sum(e.end - e.start for e in events)
    return (silence_dur / total_duration) < 0.9 if total_duration > 0 else True
