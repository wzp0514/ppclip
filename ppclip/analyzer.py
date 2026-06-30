"""ppclip 素材分析器 — 场景分析 + 情绪识别 → 8维生产级标签

双通道管线:
  通道一「场景分析」: 多帧 → 视觉LLM → 全维度场景分析
                (叙事阶段/场景描述/主体对象/动作描述/画面文字
                /持续状态/异常标注/内容类型/具体风格/制作质量/标识)
  通道二「情绪识别」: 单帧 → 视觉LLM → 情绪标签 + 可用性
  合并 → 统一素材分析结果

对齐生产系统真实标签维度:
  内容: 内容类型, 具体风格, 叙事阶段
  视觉: 制作质量
  语义: 主体对象, 动作描述, 画面文字, 情绪标签
"""

from __future__ import annotations

import base64
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .classifier import classify_frame, ClassifierResult, _get_classifier
from .config import ApiConfig, PathsConfig, Tier
from .models import get_vision_client, LLMChainResult
from .ocr import ocr_extract, _get_ocr


# ═══════════════════════════════════════════════════════════════
# Dataclasses
# ═══════════════════════════════════════════════════════════════

@dataclass
class VidlizerScene:
    """Vidlizer 单场景分析输出"""
    phase: str = ""
    scene: str = ""
    subjects: list[str] = field(default_factory=list)
    action: str = ""
    text_visible: str = ""
    context: str = ""
    observations: str = ""
    content_type: str = ""       # real-world footage / video game / animation / CGI
    specific_style: str = ""     # vlog / tutorial / gameplay / anime / etc.
    production_quality: str = "" # professional studio / amateur handheld / webcam / TV
    logos: list[str] = field(default_factory=list)


@dataclass
class ClipAnalysis:
    """统一素材分析结果 — 对齐ByteDance生产级8维标签体系"""
    # 内容维度
    content_type: str = ""
    specific_style: str = ""
    scene_phase: str = ""
    # 视觉维度
    production_quality: str = ""
    # 语义维度
    objects: list[str] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    text_visible: str = ""
    mood: str = ""
    # 辅助
    scene_desc: str = ""
    logos: list[str] = field(default_factory=list)
    observations: str = ""
    usable: bool = True
    sources: list[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════
# Prompts
# ═══════════════════════════════════════════════════════════════

VIDLIZER_SCENE_PROMPT = """Role: Expert Video Scene Analyst.

Analyze these frames from a SINGLE video scene and return a JSON object describing it exhaustively.

Rules:
- Describe only what is visually evident. Note uncertainty where present.
- Capture ALL visible text: UI labels, captions, subtitles, titles, overlays, on-screen code.
- Track any ongoing state: timer, score, progress bar, speaker identity, topic, brand.
- Flag errors, anomalies, quality problems, or anything unexpected.
- Infer the logical phase of this scene within a narrative structure.
- Classify content type, visual style, and production quality from the choices below.

Return ONLY this JSON (no prose, no code fences):
{
  "phase": "Introduction | Action | Demo | Dialogue | Transition | Conclusion | B-roll | Other",
  "scene": "Detailed description of setting and what is visible (1-2 sentences)",
  "subjects": ["subject1 with visual detail", "subject2 with visual detail"],
  "action": "What is happening — interaction, movement, event, narration",
  "text_visible": "All readable text on screen. Empty string if none.",
  "context": "Persistent state or background — brand, topic, score, timer, speaker identity",
  "observations": "Quality issues, errors, anomalies, emotional cues, key facts",
  "content_type": "real-world footage | video game | animation | cartoon | CGI | VTuber | other",
  "specific_style": "vlog | tutorial | news | gameplay | anime | 3D animation | mobile | TV | film | other",
  "production_quality": "professional studio | amateur handheld | webcam recording | TV broadcast | screen recording | other",
  "logos": ["logo1 with description", "logo2 with description"]
}"""


MOOD_PROMPT = """分析这张视频截图，仅返回JSON（不要其他文字）：
{
  "mood": "从[平静, 紧张, 激烈, 搞笑, 悲伤, 庄严, 建设, 温馨, 恐怖, 悬疑, 兴奋, 无聊]中选1-2个，用/分隔",
  "usable": true
}
规则：
- mood: 允许复合情绪（如\"激烈/紧张\"），选最突出的1-2个
- usable: 模糊/剧烈晃动/大面积UI遮挡/过曝/转场黑屏/过暗 → false"""


# ═══════════════════════════════════════════════════════════════
# Vidlizer 场景分析通道
# ═══════════════════════════════════════════════════════════════

def _analyze_scene_vidlizer(
    thumb_paths: list[Path],
    client: LLMChainResult,
    retries: int = 2,
) -> Optional[VidlizerScene]:
    """用 Vidlizer 风格的 prompt 对单个场景的关键帧做分析"""
    if not thumb_paths:
        return None

    content: list[dict] = [{"type": "text", "text": VIDLIZER_SCENE_PROMPT}]
    for p in thumb_paths:
        b64 = _image_to_base64(p)
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        })

    for attempt in range(retries + 1):
        try:
            resp = client.client.chat.completions.create(
                model=client.model,
                messages=[{"role": "user", "content": content}],
                max_tokens=800,
                temperature=0.2,
            )
            raw = (resp.choices[0].message.content or "").strip()
            data = _parse_json_lenient(raw)
            if data:
                return VidlizerScene(
                    phase=data.get("phase", ""),
                    scene=data.get("scene", ""),
                    subjects=data.get("subjects", []),
                    action=data.get("action", ""),
                    text_visible=data.get("text_visible", ""),
                    context=data.get("context", ""),
                    observations=data.get("observations", ""),
                    content_type=data.get("content_type", ""),
                    specific_style=data.get("specific_style", ""),
                    production_quality=data.get("production_quality", ""),
                    logos=data.get("logos", []),
                )
        except Exception:
            if attempt < retries:
                time.sleep(2 ** attempt)
    return None


# ═══════════════════════════════════════════════════════════════
# Mood 通道 (复用现有 ppclip vision prompt 结构)
# ═══════════════════════════════════════════════════════════════

def _analyze_mood(
    thumb_path: Path,
    client: LLMChainResult,
    retries: int = 2,
) -> tuple[str, bool]:
    """单帧情绪分析，返回 (mood, usable)"""
    b64 = _image_to_base64(thumb_path)
    for attempt in range(retries + 1):
        try:
            resp = client.client.chat.completions.create(
                model=client.model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                        {"type": "text", "text": MOOD_PROMPT},
                    ],
                }],
                max_tokens=80,
                temperature=0.3,
            )
            raw = (resp.choices[0].message.content or "").strip()
            data = _parse_json_lenient(raw)
            if data:
                return data.get("mood", ""), data.get("usable", True)
        except Exception:
            if attempt < retries:
                time.sleep(2 ** attempt)
    return "", True


# ═══════════════════════════════════════════════════════════════
# 文件名兜底
# ═══════════════════════════════════════════════════════════════

_FILENAME_CT_MAP: dict[str, str] = {
    "红警": "video game", "游戏": "video game", "game": "video game",
    "三角洲": "video game", "lol": "video game", "英雄联盟": "video game",
    "vlog": "real-world footage", "日常": "real-world footage",
    "教程": "real-world footage", "tutorial": "real-world footage",
    "动漫": "animation", "动画": "animation", "anime": "animation",
    "3d": "CGI", "cgi": "CGI", "渲染": "CGI",
    "加速": "video game", "倍速": "video game",
    "红绿灯": "real-world footage", "交通": "real-world footage",
}

_FILENAME_STYLE_MAP: dict[str, str] = {
    "红警": "gameplay", "游戏": "gameplay", "game": "gameplay",
    "三角洲": "gameplay", "lol": "gameplay",
    "加速": "gameplay", "倍速": "gameplay",
    "vlog": "vlog", "日常": "vlog",
    "教程": "tutorial", "tutorial": "tutorial",
    "动漫": "anime", "动画": "anime", "anime": "anime",
}


def _filename_fallback(file_name: str) -> tuple[str, str]:
    """Extract content_type and specific_style from filename keywords."""
    name_lower = file_name.lower()
    ct = ""
    style = ""
    for kw, v in _FILENAME_CT_MAP.items():
        if kw.lower() in name_lower:
            ct = v
            break
    for kw, v in _FILENAME_STYLE_MAP.items():
        if kw.lower() in name_lower:
            style = v
            break
    return ct, style


# ═══════════════════════════════════════════════════════════════
# 融合引擎
# ═══════════════════════════════════════════════════════════════

def _merge_results(
    vs: Optional[VidlizerScene],
    mood: str,
    usable: bool,
    local_vs: Optional[ClassifierResult] = None,
    ocr_text: str = "",
    file_name: str = "",
) -> ClipAnalysis:
    """多通道融合: Vidlizer API + Mood API + LocalClassifier + LocalOCR → 统一 ClipAnalysis

    主通道(Vidlizer API)优先，本地方案补位缺失字段。
    所有通道失效时，文件名关键词兜底。
    """
    sources: list[str] = []
    if vs:
        sources.append("vidlizer")
    if mood:
        sources.append("mood")

    # ── 分类维度: API优先 → 本地补位 ──
    content_type = vs.content_type if vs else ""
    specific_style = vs.specific_style if vs else ""
    production_quality = vs.production_quality if vs else ""
    if local_vs:
        if not content_type and local_vs.content_type:
            content_type = local_vs.content_type
            if "vidlizer" not in sources:
                sources.append("local_cls")
        if not specific_style and local_vs.specific_style:
            specific_style = local_vs.specific_style
        if not production_quality and local_vs.production_quality:
            production_quality = local_vs.production_quality

    # ── 主体对象/动作/场景 → 仅 Vidlizer 提供 (本地不覆盖) ──
    objects = list(vs.subjects) if vs else []
    actions = [vs.action] if (vs and vs.action) else []
    scene_phase = vs.phase if vs else ""
    scene_desc = vs.scene if vs else ""

    # ── 画面文字: API + OCR 合并去重 ──
    text_visible = vs.text_visible if vs else ""
    if ocr_text:
        api_lower = text_visible.lower()
        ocr_parts = [t for t in ocr_text.split() if t.lower() not in api_lower]
        if ocr_parts:
            supplement = " ".join(ocr_parts)
            text_visible = f"{text_visible} | {supplement}" if text_visible else supplement
        if "local_ocr" not in sources:
            sources.append("local_ocr")

    # ── 文件名兜底: content_type 空时从文件名提取 ──
    fn_ct = ""
    fn_style = ""
    if not content_type and file_name:
        fn_ct, fn_style = _filename_fallback(file_name)
        if fn_ct:
            content_type = fn_ct
            sources.append("filename")
        if fn_style and not specific_style:
            specific_style = fn_style
            if "filename" not in sources:
                sources.append("filename")

    # ── Observations + quality 标记 ──
    obs_parts = []
    if vs and vs.observations:
        obs_parts.append(vs.observations)
    if production_quality in ("amateur handheld", "webcam recording"):
        obs_parts.append(f"production: {production_quality}")
    observations = " | ".join(obs_parts)

    # Logos
    logos = list(vs.logos) if vs else []

    # Usable
    if not usable:
        final_usable = False
    elif production_quality in ("amateur handheld", "webcam recording"):
        final_usable = True
    else:
        final_usable = True

    return ClipAnalysis(
        content_type=content_type,
        specific_style=specific_style,
        scene_phase=scene_phase,
        production_quality=production_quality,
        objects=objects,
        actions=actions,
        text_visible=text_visible,
        mood=mood,
        scene_desc=scene_desc,
        logos=logos,
        observations=observations,
        usable=final_usable,
        sources=sources,
    )


# ═══════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════

def run_analysis(
    project_dir: Path,
    tier: Tier,
    api: ApiConfig,
    paths: PathsConfig,
    *,
    verbose: bool = True,
    force: bool = False,
) -> int:
    """主分析入口：读取 materials.json → 三层分析 → 写回

    返回成功分析的 clip 数量。
    """
    materials_path = project_dir / "materials.json"
    if not materials_path.exists():
        print("[ppclip] materials.json not found")
        return 0

    with open(materials_path, "r", encoding="utf-8") as f:
        materials = json.load(f)

    idx = tier.index
    feat = tier.features

    use_vidlizer = feat.use_vidlizer_flow
    use_cache = feat.use_cache and not force
    cls_model = idx.local_classifier_model
    cls_device = idx.local_classifier_device
    ocr_lang = idx.ocr_lang
    hf_endpoint = paths.hf_endpoint
    max_calls = idx.max_vision_calls
    delay_s = idx.vision_delay_ms / 1000.0

    # 获取 vision client（Vidlizer 和 Mood 共用）
    vision_client = None
    if use_vidlizer or feat.use_vision:
        vision_client = get_vision_client(api, verbose=verbose)

    # 本地模型 auto-detect（API 失败/未启用时静默补位）
    local_cls_available = _get_classifier(cls_model, cls_device, hf_endpoint=hf_endpoint) is not None
    local_ocr_available = _get_ocr(ocr_lang) is not None

    if verbose:
        print(f"ppclip analyze")
        channels = []
        if use_vidlizer and vision_client:
            channels.append(f"Vidlizer ({vision_client.provider_name}/{vision_client.model})")
        if vision_client:
            channels.append("Mood")
        if local_cls_available:
            channels.append(f"LocalCls [fallback] ({cls_model})")
        if local_ocr_available:
            channels.append(f"LocalOCR [supplement] ({ocr_lang})")
        if not channels:
            channels.append("legacy vision (fallback)")
        print(f"  通道: {', '.join(channels)}")
        print(f"  max_calls={max_calls}, delay={delay_s}s")
        print()

    # 构建 file_md5 索引用于缓存判断
    file_md5_map: dict[str, str] = {}  # file_name → md5
    for mf in materials.get("files", []):
        if mf.get("file_md5"):
            file_md5_map[mf["file"]] = mf["file_md5"]

    # 收集所有带缩略图的 clip
    clip_items: list[tuple[str, list[Path], dict, str]] = []  # (file_name, [thumb_paths], clip_dict, file_md5)
    cache_hits = 0
    for mf in materials.get("files", []):
        if mf.get("media_type") == "audio":
            continue
        fmd5 = mf.get("file_md5", "")
        for clip in mf.get("clips", []):
            # MD5 cache: skip if analysis already exists and file MD5 unchanged
            if use_cache and clip.get("analysis") and fmd5:
                cache_hits += 1
                continue
            thumb_rel = clip.get("thumbnail", "")
            if not thumb_rel:
                continue
            thumb_path = project_dir / thumb_rel
            if not thumb_path.exists():
                continue
            clip_items.append((mf["file"], [thumb_path], clip, fmd5))

    if cache_hits and verbose:
        print(f"  [CACHE] {cache_hits} clips (analysis exists, skip)")

    if not clip_items:
        if verbose:
            if cache_hits:
                print("[ppclip] 所有 clips 已分析，无需重新分析")
            else:
                print("[ppclip] 无可分析的缩略图")
        return cache_hits

    per_file_quota = max(1, max_calls // max(len(clip_items), 1))
    if verbose:
        print(f"  {len(clip_items)} clips, ~{per_file_quota} calls max per clip")

    call_count = 0
    success = 0

    needs_api = use_vidlizer or (vision_client is not None)

    for file_name, thumb_paths, clip, _fmd5 in clip_items:
        if needs_api and call_count >= max_calls:
            break

        if verbose:
            cid = clip.get("id", "?")
            if needs_api:
                print(f"  [{call_count + 1}/{max_calls}] {cid}...", end=" ", flush=True)
            else:
                print(f"  [local] {cid}...", end=" ", flush=True)

        vs_result: Optional[VidlizerScene] = None
        mood = ""
        usable = True

        # L2a: Vidlizer 场景分析（含分类维度：content_type/specific_style/production_quality/logos）
        if use_vidlizer and vision_client:
            vs_result = _analyze_scene_vidlizer(thumb_paths, vision_client)
            call_count += 1
            time.sleep(delay_s)

        # L2b: Mood 情绪
        if vision_client and thumb_paths:
            mood, usable = _analyze_mood(thumb_paths[0], vision_client)
            if not use_vidlizer:
                call_count += 1
                time.sleep(delay_s)

        # L2c: LocalClassifier — API 失败/未启用时静默补位
        local_vs: Optional[ClassifierResult] = None
        if local_cls_available and not vs_result:
            local_vs = classify_frame(thumb_paths[0], model_name=cls_model, device=cls_device, hf_endpoint=hf_endpoint)

        # L2d: LocalOCR — 始终补充文字提取（独立通道）
        ocr_text = ""
        if local_ocr_available:
            ocr_text = ocr_extract(thumb_paths[0], lang=ocr_lang)

        # L3: 融合
        analysis = _merge_results(vs_result, mood, usable, local_vs, ocr_text, file_name=file_name)

        # 写回 clip（中文键名 — 对齐生产系统标签维度）
        clip["analysis"] = {
            "内容类型": analysis.content_type,
            "具体风格": analysis.specific_style,
            "叙事阶段": analysis.scene_phase,
            "制作质量": analysis.production_quality,
            "主体对象": analysis.objects,
            "动作描述": analysis.actions,
            "画面文字": analysis.text_visible,
            "情绪标签": analysis.mood,
            "场景描述": analysis.scene_desc,
            "标识": analysis.logos,
            "异常标注": analysis.observations,
            "可用": analysis.usable,
            "来源通道": analysis.sources,
        }
        success += 1

        if verbose:
            dims = []
            if analysis.content_type:
                dims.append(analysis.content_type)
            if analysis.specific_style:
                dims.append(analysis.specific_style)
            if analysis.mood:
                dims.append(analysis.mood)
            print(f"OK ({', '.join(dims) if dims else '?'}) [{'+'.join(analysis.sources)}]")

    # 更新 materials.json
    materials["analysis_version"] = 2
    materials["analysis_provider"] = (
        f"{vision_client.provider_name}/{vision_client.model}" if vision_client else "none"
    )
    materials["analysis_success"] = success
    materials["analysis_total"] = min(len(clip_items), max_calls)
    materials["analysis_channels"] = [
        c for c, enabled in [
            ("vidlizer", use_vidlizer),
            ("mood", bool(vision_client)),
            ("local_cls", local_cls_available),
            ("local_ocr", local_ocr_available),
        ] if enabled
    ]

    with open(materials_path, "w", encoding="utf-8") as f:
        json.dump(materials, f, indent=2, ensure_ascii=False)

    if verbose:
        print()
        print(f"Analysis done: {success}/{min(len(clip_items), max_calls)} success")
        print(f"  materials.json updated (analysis_version=2)")

    return success


# ═══════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════

def _image_to_base64(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _parse_json_lenient(raw: str) -> Optional[dict]:
    """宽松JSON解析：纯JSON → code fence → regex"""
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    m = re.search(r'```(?:json)?\s*([\s\S]*?)```', raw)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    m = re.search(r'\{[\s\S]*\}', raw)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return None
