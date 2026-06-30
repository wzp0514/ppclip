"""ppclip config — 环境变量 > config.json，两级优先级合并，全量 dataclass"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

CONFIG_HOME = Path.home() / ".ppclip"
CONFIG_FILE = CONFIG_HOME / "config.json"


@dataclass
class ApiConfig:
    api_key: str = ""
    text_key_1: str = ""
    text_url_1: str = "https://api.deepseek.com"
    text_model_1: str = "deepseek-chat"
    text_key_2: str = ""
    text_url_2: str = "https://open.bigmodel.cn/api/paas/v4"
    text_model_2: str = "glm-4-flash"
    text_key_3: str = ""
    text_url_3: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    text_model_3: str = "qwen-turbo"
    image_key_1: str = ""
    image_model_1: str = "glm-4.6v-flash"
    image_key_2: str = ""
    image_model_2: str = "qwen-vl-plus"
    tts_key: str = ""
    tts_voice_id: str = ""


@dataclass
class PathsConfig:
    output_base: str = ""
    ffmpeg_path: str = ""
    jianying_draft_dir: str = ""
    jianying_user_data: str | None = None
    material_cache_dir: str | None = None
    hf_endpoint: str = "https://huggingface.co"
    jianying_skill_root: str = ""
    remotion_skill_root: str = ""


@dataclass
class BuildConfig:
    canvas_width: int = 1920
    canvas_height: int = 1080
    fps: int = 30
    default_transition: str = "crossfade"
    crossfade_duration: float = 0.3
    subtitle_font_size: int = 15
    subtitle_bold: bool = True
    subtitle_has_shadow: bool = True
    subtitle_overlap: float = 0.3
    default_speaker: str = "zh_male_huoli"
    tts_enabled: bool = True
    enable_remotion: bool = False


@dataclass
class FeaturesConfig:
    use_vision: bool = True
    use_llm_match: bool = True
    use_scene_detect: bool = True
    use_ds: bool = False
    use_audio: bool = False
    use_vidlizer_flow: bool = False
    use_cache: bool = True
    use_tts: bool = False
    use_bgm: bool = False
    use_effects: bool = False
    use_remotion: bool = False


@dataclass
class IndexConfig:
    scene_threshold: float = 0.3
    min_clip_duration: float = 2.0
    max_clips_per_file: int = 30
    frame_interval: float = 15.0
    max_vision_calls: int = 100
    vision_quality: str = "medium"
    vision_delay_ms: int = 2000
    thumbnail_width: int = 320
    thumbnail_quality: int = 3
    skip_scene_detect: bool = False
    skip_thumbnails: bool = False
    max_file_size_mb: int = 0
    audio_enabled: bool = False
    silence_threshold_db: int = -40
    silence_min_duration: float = 0.5
    use_vidlizer_flow: bool = False
    vidlizer_model: str = ""
    local_classifier_model: str = "MobileCLIP-S2"
    local_classifier_device: str = "cpu"
    ocr_lang: str = "ch"


@dataclass
class ScriptConfig:
    min_shot_duration: float = 1.0
    max_shot_duration: float = 30.0
    min_total_duration: float = 15.0
    max_total_duration: float = 120.0
    max_shots: int = 20
    max_tokens: int = 4096
    temperature: float = 0.7


@dataclass
class MatchConfig:
    confidence_threshold: float = 0.7
    batch_size: int = 5
    max_candidates_per_shot: int = 50
    prefer_same_source: bool = False
    temperature: float = 0.3
    max_tokens: int = 1024
    skip_llm: bool = False
    match_llm_candidate_limit: int = 20


@dataclass
class EnrichTierConfig:
    bgm_search_limit: int = 3
    effect_mode: str = "minimal"  # none | minimal | normal | rich
    ken_burns_mode: str = "none"  # none | minimal | normal


@dataclass
class Tier:
    name: str
    description: str
    est_time: str
    features: FeaturesConfig
    index: IndexConfig
    script: ScriptConfig
    match: MatchConfig
    enrich: EnrichTierConfig = field(default_factory=EnrichTierConfig)


@dataclass
class Config:
    api: ApiConfig
    paths: PathsConfig
    build: BuildConfig
    enrich: "EnrichConfig" = None

    def __post_init__(self):
        if self.enrich is None:
            self.enrich = EnrichConfig()


@dataclass
class EnrichConfig:
    bgm_enabled: bool = True
    default_bgm_volume: float = 0.4
    effect_enabled: bool = True
    ken_burns_enabled: bool = True
    intro_enabled: bool = True
    outro_enabled: bool = True


# ─── 4 档位 (每档全量自包含，不 merge 不继承) ───

DEV = Tier(
    name="dev",
    description="开发自测，极简快速过流程",
    est_time="~30s",
    features=FeaturesConfig(use_vision=False, use_llm_match=False, use_scene_detect=False, use_ds=False, use_audio=False, use_vidlizer_flow=False, use_cache=True, use_tts=False, use_bgm=False, use_effects=False, use_remotion=False),
    index=IndexConfig(max_vision_calls=0, max_clips_per_file=5, skip_scene_detect=True, skip_thumbnails=True, thumbnail_width=80, max_file_size_mb=100),
    script=ScriptConfig(temperature=0.3, max_tokens=2000, max_shots=8, max_total_duration=30),
    match=MatchConfig(max_candidates_per_shot=10, temperature=0.1, skip_llm=True, match_llm_candidate_limit=10),
    enrich=EnrichTierConfig(bgm_search_limit=0, effect_mode="none", ken_burns_mode="none"),
)

TEST = Tier(
    name="test",
    description="测试验证，检查匹配效果",
    est_time="~120s",
    features=FeaturesConfig(use_vision=False, use_llm_match=True, use_scene_detect=False, use_ds=False, use_audio=False, use_vidlizer_flow=False, use_cache=True, use_tts=False, use_bgm=False, use_effects=False, use_remotion=False),
    index=IndexConfig(max_vision_calls=0, max_clips_per_file=10, skip_scene_detect=True, thumbnail_width=240, max_file_size_mb=500),
    script=ScriptConfig(temperature=0.5, max_tokens=4000),
    match=MatchConfig(max_candidates_per_shot=30, temperature=0.2, skip_llm=False),
    enrich=EnrichTierConfig(bgm_search_limit=3, effect_mode="minimal", ken_burns_mode="none"),
)

PROD = Tier(
    name="prod",
    description="生产输出，推荐日常使用",
    est_time="~200s",
    features=FeaturesConfig(use_vision=True, use_llm_match=True, use_scene_detect=True, use_ds=True, use_audio=False, use_vidlizer_flow=False, use_cache=True, use_tts=True, use_bgm=True, use_effects=True, use_remotion=False),
    index=IndexConfig(max_vision_calls=30, thumbnail_width=320, vision_quality="medium", vision_delay_ms=2000, scene_threshold=0.3, skip_scene_detect=False, skip_thumbnails=False),
    script=ScriptConfig(temperature=0.7, max_tokens=5000),
    match=MatchConfig(max_candidates_per_shot=50, temperature=0.3, skip_llm=False),
    enrich=EnrichTierConfig(bgm_search_limit=5, effect_mode="normal", ken_burns_mode="minimal"),
)

FULL = Tier(
    name="full",
    description="全量输出，长片/复杂素材",
    est_time="~400s",
    features=FeaturesConfig(use_vision=True, use_llm_match=True, use_scene_detect=True, use_ds=True, use_audio=True, use_vidlizer_flow=True, use_cache=True, use_tts=True, use_bgm=True, use_effects=True, use_remotion=True),
    index=IndexConfig(max_vision_calls=80, thumbnail_width=480, vision_quality="high", vision_delay_ms=1500, scene_threshold=0.2, max_clips_per_file=30, skip_scene_detect=False, skip_thumbnails=False),
    script=ScriptConfig(temperature=0.8, max_tokens=5000),
    match=MatchConfig(max_candidates_per_shot=100, temperature=0.5, skip_llm=False),
    enrich=EnrichTierConfig(bgm_search_limit=10, effect_mode="rich", ken_burns_mode="normal"),
)

TIERS: dict[str, Tier] = {
    "dev": DEV,
    "test": TEST,
    "prod": PROD,
    "full": FULL,
}

# ─── 环境变量映射 ───

ENV_KEY_MAP = {
    "PPCLIP_API_KEY": "api_key",
    "DEEPSEEK_API_KEY": "text_key_1",
    "ZHIPU_API_KEY": "text_key_2",
    "QWEN_API_KEY": "text_key_3",
    "MINIMAX_API_KEY": "tts_key",
    "MINIMAX_VOICE_ID": "tts_voice_id",
}


def _apply_env(api: ApiConfig) -> ApiConfig:
    for env_var, field in ENV_KEY_MAP.items():
        val = os.environ.get(env_var, "").strip()
        if val:
            setattr(api, field, val)
    return api


# ─── 配置加载 ───

def _bundled_skill_root(name: str) -> str:
    p = Path(__file__).parent / "skills" / name
    if p.exists():
        return str(p)
    return ""


def load_config(config_path: Optional[Path] = None) -> Config:
    path = Path(config_path) if config_path else CONFIG_FILE

    api = ApiConfig()
    paths = PathsConfig(
        output_base=str(Path(__file__).parent / "output"),
        hf_endpoint="https://huggingface.co",
        jianying_skill_root=_bundled_skill_root("jianying-editor"),
        remotion_skill_root=_bundled_skill_root("remotion-video"),
    )
    build = BuildConfig()
    enrich = EnrichConfig()

    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            _populate(api, raw.get("api", {}))
            _populate(paths, raw.get("paths", {}))
            _populate(build, raw.get("build", {}))
            _populate(enrich, raw.get("enrich", {}))
        except (json.JSONDecodeError, OSError) as e:
            import sys
            print(f"[warn] config.json 解析失败，使用默认配置: {e}", file=sys.stderr)

    api = _apply_env(api)
    if api.api_key:
        for field in ("text_key_1", "text_key_2", "text_key_3", "image_key_1", "image_key_2"):
            if not getattr(api, field):
                setattr(api, field, api.api_key)
    return Config(api=api, paths=paths, build=build, enrich=enrich)


def _populate(obj, data: dict) -> None:
    for k, v in data.items():
        if hasattr(obj, k) and v is not None:
            # Coerce string numbers for int/float fields (config.json values are always strings)
            current = getattr(obj, k)
            if isinstance(current, bool):
                if isinstance(v, str):
                    v = v.lower() in ("true", "1", "yes")
            elif isinstance(current, int) and isinstance(v, str):
                try:
                    v = int(v)
                except ValueError:
                    continue
            elif isinstance(current, float) and isinstance(v, str):
                try:
                    v = float(v)
                except ValueError:
                    continue
            setattr(obj, k, v)


def ensure_config_dir() -> Path:
    CONFIG_HOME.mkdir(parents=True, exist_ok=True)
    return CONFIG_HOME

