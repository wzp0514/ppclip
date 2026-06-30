"""ppclip enrich — 生产配置增强：BGM选曲/TTS配置/特效分配/转场选择/关键帧方案。

读取 timeline.json + materials.json → 输出 enhanced_timeline.json。
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from .config import BuildConfig, EnrichConfig, EnrichTierConfig, Tier


@dataclass
class BgmConfig:
    cloud_music_id: str = ""
    name: str = ""
    volume: float = 0.4
    fade_in: float = 1.0
    fade_out: float = 2.0


@dataclass
class TtsConfig:
    enabled: bool = True
    speaker: str = "zh_male_huoli"
    backend: str = "edge"


@dataclass
class TextAnim:
    in_anim: str = ""
    in_duration: str = "0.6s"


@dataclass
class KeyframeConfig:
    scale_start: float = 1.0
    scale_end: float = 1.0
    easing: str = ""


@dataclass
class IntroOutro:
    enabled: bool = True
    text: str = ""
    style: str = ""
    style_id: str = ""
    duration: float = 2.0
    fade_out: float = 1.0


@dataclass
class EnhancedShot:
    id: int = 0
    duration: float = 0.0
    clip_id: str = ""
    src_start: float = 0.0
    src_end: float = 0.0
    subtitle: str = ""
    transition: str = "crossfade"
    transition_duration: float = 0.3
    effect: Optional[str] = None
    text_anim: TextAnim = field(default_factory=TextAnim)
    keyframes: KeyframeConfig = field(default_factory=KeyframeConfig)
    match_confidence: float = 0.0
    match_reason: str = ""
    match_validation: dict = field(default_factory=dict)


@dataclass
class EnhancedTimeline:
    version: int = 1
    project: str = ""
    total_duration: float = 0.0
    video_type: str = "clip_splicing"
    bgm: BgmConfig = field(default_factory=BgmConfig)
    tts: TtsConfig = field(default_factory=TtsConfig)
    shots: list[EnhancedShot] = field(default_factory=list)
    intro: IntroOutro = field(default_factory=IntroOutro)
    outro: IntroOutro = field(default_factory=IntroOutro)


# 情绪 → BGM 关键词映射
MOOD_BGM_KEYWORDS: dict[str, list[str]] = {
    "激烈": ["Epic", "Cinematic", "动感", "Action", "Rock", "战"],
    "紧张": ["Suspense", "Tension", "紧张", "Dark", "Thriller"],
    "搞笑": ["Funny", "Comedy", "可爱", "萌宠", "轻快"],
    "平静": ["Lofi", "舒缓", "Calm", "Peaceful", "旅行", "VLOG"],
    "悲伤": ["Sad", "Emotional", "Piano", "悲伤"],
    "庄严": ["Corporate", "Motivational", "企业", "进取"],
    "建设": ["Corporate", "Uplifting", "进取", "创新"],
}


def _speaker_for_mood(mood: str, default: str) -> str:
    mood_speaker = {
        "激烈": "zh_male_huoli",
        "紧张": "zh_male_huoli",
        "搞笑": "zh_male_xionger_stream_gpu",
        "平静": "zh_female_xiaopengyou",
        "悲伤": "zh_female_inspirational",
        "庄严": "zh_female_inspirational",
        "建设": "zh_male_huoli",
    }
    return mood_speaker.get(mood, default)


def _effect_for_role(clip_role: str, mood: str, mode: str) -> Optional[str]:
    """根据 clip_role、情绪和 enrich 档位分配特效。"""
    if mode == "none":
        return None

    mood_effects: dict[str, str] = {
        "激烈": "抖动",
        "紧张": "模糊",
        "搞笑": "放大",
        "平静": "柔光",
        "悲伤": "柔光",
        "庄严": "闪光",
        "建设": "柔光",
    }
    base = mood_effects.get(mood)

    if mode == "minimal":
        if clip_role == "highlight_source":
            return base
        return None

    if mode == "normal":
        if clip_role in ("highlight_source", "intro_outro"):
            return base
        return None

    # rich: all non-transition shots get effects
    if clip_role == "transition":
        return None
    return base


def _anim_for_mood(mood: str) -> str:
    """根据情绪选择字幕入场动画。"""
    mood_anims = {
        "激烈": "弹入",
        "紧张": "淡入",
        "搞笑": "弹跳",
        "平静": "淡入",
        "悲伤": "淡入",
        "庄严": "优雅浮现",
        "建设": "滑入",
    }
    return mood_anims.get(mood, "淡入")


def _transition_for_shot(prev_mood: Optional[str], curr_mood: str, default: str) -> str:
    if prev_mood is None:
        return "fade_in"
    if prev_mood == curr_mood:
        return "crossfade"
    return "cut"


def _ken_burns_for_role(clip_role: str, mode: str) -> KeyframeConfig:
    if mode == "none":
        return KeyframeConfig()
    if mode == "minimal" and clip_role in ("intro_outro", "highlight_source"):
        return KeyframeConfig(scale_start=1.0, scale_end=1.05, easing="ease_in_out")
    if mode == "normal":
        if clip_role == "intro_outro":
            return KeyframeConfig(scale_start=1.0, scale_end=1.08, easing="ease_in_out")
        if clip_role == "highlight_source":
            return KeyframeConfig(scale_start=1.0, scale_end=1.03, easing="ease_in_out")
    return KeyframeConfig()


def _load_bgm_library(skill_data_dir: Path) -> list[dict]:
    csv_path = skill_data_dir / "cloud_music_library.csv"
    if not csv_path.exists():
        return []
    items: list[dict] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(
            (line for line in f if not line.startswith("#")),
        )
        for row in reader:
            items.append(row)
    return items


def _select_bgm(moods: list[str], bgm_library: list[dict], search_limit: int, volume: float) -> BgmConfig:
    if not bgm_library or search_limit <= 0:
        return BgmConfig()

    keywords: set[str] = set()
    for m in moods:
        for kw in MOOD_BGM_KEYWORDS.get(m, []):
            keywords.add(kw.lower())

    scored: list[tuple[int, dict]] = []
    for item in bgm_library:
        cats = (item.get("categories", "") or "").lower()
        title = (item.get("title", "") or "").lower()
        text = f"{cats} {title}"
        score = sum(1 for kw in keywords if kw in text)
        if score > 0:
            scored.append((score, item))

    scored.sort(key=lambda x: -x[0])
    if not scored:
        return BgmConfig()

    best = scored[0][1]
    return BgmConfig(
        cloud_music_id=best.get("music_id", ""),
        name=best.get("title", ""),
        volume=volume,
    )


def _dominant_moods(shots: list[dict]) -> list[str]:
    counts: dict[str, int] = {}
    for s in shots:
        m = s.get("mood", "")
        if m and m != "-":
            counts[m] = counts.get(m, 0) + 1
    return sorted(counts, key=counts.get, reverse=True)  # type: ignore[arg-type, return-value]


def run_enrich(
    project_dir: Path,
    tier: Tier,
    build: BuildConfig,
    enrich_cfg: EnrichConfig,
    *,
    skill_data_dir: Optional[Path] = None,
    verbose: bool = True,
) -> Optional[EnhancedTimeline]:
    timeline_path = project_dir / "timeline.json"
    if not timeline_path.exists():
        if verbose:
            print("  timeline.json 不存在，跳过 enrich")
        return None

    with open(timeline_path, "r", encoding="utf-8") as f:
        timeline = json.load(f)

    shots = timeline.get("shots", [])
    if not shots:
        return None

    etier = tier.enrich
    moods = _dominant_moods(shots)
    dominant_mood = moods[0] if moods else "平静"

    # BGM
    bgm = BgmConfig()
    if enrich_cfg.bgm_enabled and etier.bgm_search_limit > 0 and skill_data_dir:
        bgm_lib = _load_bgm_library(skill_data_dir)
        bgm = _select_bgm(moods, bgm_lib, etier.bgm_search_limit, enrich_cfg.default_bgm_volume)

    # TTS
    tts = TtsConfig(
        enabled=tier.features.use_tts,
        speaker=_speaker_for_mood(dominant_mood, build.default_speaker),
    )

    # Shots
    enhanced_shots: list[EnhancedShot] = []
    prev_mood: Optional[str] = None
    for s in shots:
        sid = s.get("id", 0)
        if not sid:
            continue
        mood = s.get("mood", "")
        role = s.get("clip_role", "")
        dur = s.get("duration", 3.0)

        trans = _transition_for_shot(prev_mood, mood, build.default_transition)
        xdur = build.crossfade_duration if trans == "crossfade" else 0.0

        effect = None
        if enrich_cfg.effect_enabled:
            effect = _effect_for_role(role, mood, etier.effect_mode)

        text_anim = TextAnim()
        if etier.effect_mode not in ("none",):
            text_anim = TextAnim(in_anim=_anim_for_mood(mood), in_duration="0.6s")

        kf = KeyframeConfig()
        if enrich_cfg.ken_burns_enabled:
            kf = _ken_burns_for_role(role, etier.ken_burns_mode)

        validation = s.get("match_validation", {})

        es = EnhancedShot(
            id=sid,
            duration=dur,
            clip_id=s.get("clip_id", ""),
            src_start=s.get("src_start", 0.0),
            src_end=s.get("src_end", 0.0),
            subtitle=s.get("subtitle", ""),
            transition=trans,
            transition_duration=round(xdur, 2),
            effect=effect,
            text_anim=text_anim,
            keyframes=kf,
            match_confidence=s.get("match_confidence", 0.0),
            match_reason=s.get("match_reason", ""),
            match_validation=validation,
        )
        enhanced_shots.append(es)
        prev_mood = mood

    total = sum(es.duration for es in enhanced_shots)

    # Intro/Outro
    intro = IntroOutro(
        enabled=enrich_cfg.intro_enabled,
        text=timeline.get("project", ""),
        style="花字",
        style_id="7351316503771368713",  # 红色花字1（剪映内置花字样式ID，从 cloud_text_styles.csv 索引）
        duration=2.0,
    )
    outro = IntroOutro(
        enabled=enrich_cfg.outro_enabled,
        text="感谢观看",
        duration=3.0,
        fade_out=1.0,
    )
    if etier.effect_mode == "none":
        intro.enabled = False
        outro.enabled = False

    result = EnhancedTimeline(
        project=timeline.get("project", ""),
        total_duration=total,
        video_type=timeline.get("video_type", "clip_splicing"),
        bgm=bgm,
        tts=tts,
        shots=enhanced_shots,
        intro=intro,
        outro=outro,
    )

    output_path = project_dir / "enhanced_timeline.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(asdict(result), f, indent=2, ensure_ascii=False)

    if verbose:
        print(f"ppclip enrich")
        print(f"  情绪: {', '.join(moods)} | BGM: {bgm.name or '(无)'} | TTS: {tts.speaker}")
        print(f"  intro: {'on' if intro.enabled else 'off'} | outro: {'on' if outro.enabled else 'off'}")
        print(f"  effects: {etier.effect_mode} | ken_burns: {etier.ken_burns_mode}")
        print(f"  enhanced_timeline.json 已输出")

    return result

