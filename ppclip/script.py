"""ppclip script — LLM 降级链生成分镜框架，输出 script.md + script.json"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .config import ApiConfig, Tier
from .models import get_llm_client, LLMChainResult

SYSTEM_PROMPT = """你是皮皮出片（ppclip）的短视频剪辑策划助手。
你的任务是根据用户的创意想法和素材概览，生成创意分镜框架。素材分配由后续步骤完成。

## 核心约束（违反即重做）

1. **你的任务是写创意分镜框架，不是分配素材**。素材分配由后续步骤完成。
2. **画面需求列写得像"找素材指令"**——具体、可检索。
   ✅ "红警坦克集群冲锋画面，炮火密集"
   ❌ "展示一段精彩内容"
3. **情绪标签从素材档案中已有的标签中选择**。如果素材情绪分布与用户想法不匹配（如用户要"温馨"但素材无此标签），优先用最接近的已有标签（如"平静"）并说明理由。不要自创标签。
  可选标签：平静 / 紧张 / 激烈 / 搞笑 / 悲伤 / 庄严 / 建设
4. **无素材清单时**，画面需求列填"【需录制】+ 描述"，情绪列填"-"。
5. 口播字幕：5-15 字/句，口语化、接地气，杜绝书面语和连接词（"随后/因此/然而"）。
6. 总时长：40-70 秒（有素材时），30-60 秒（无素材时）。

## 创作原则

- 用户想法是第一优先级，不要自行添加不相干创意
- 搞笑类视频：铺垫（正常）→ 转折（意外）→ 揭秘（反转）
- 开头 3 秒必须有钩子（冲突/悬念/利益点/反常识事实）
- 结尾引导互动（点赞/关注/评论）
- 节奏紧凑，每 10 秒有信息变化

## 输出格式（仅输出此 JSON，不要其他文字）

```json
{
  "total_duration": 45.0,
  "shots": [
    {
      "id": 1,
      "duration": 3.0,
      "need": "红警开局基地部署",
      "mood": "平静",
      "source_hint": "红警全程",
      "clip_role": "feature",
      "subtitle": "兄弟们今天来一把",
      "transition": "crossfade"
    }
  ]
}
```

字段说明：
- id: 从 1 开始的镜号
- duration: 单镜 1-30s
- need: 画面需求，具体可检索。无素材时填"【需录制】描述"
- mood: 从可选标签中选择，无素材时填"-"
- source_hint: 建议从哪个素材文件取（填文件名，如"红警全程"），不确定填""
- clip_role: 该镜头在叙事中的角色，必填以下之一：
  - highlight_source: 高光素材，提供核心内容（游戏精彩片段/关键画面）
  - transition: 过渡桥段，连接两个段落（加速/空镜/转场）
  - feature: 特色内容，配合解说展示特定信息
  - intro_outro: 片头或片尾
- subtitle: 5-15字口语化文案
- transition: crossfade（默认）或 cut
- total_duration: 所有镜头时长之和
"""


@dataclass
class ShotScript:
    id: int
    duration: float
    need: str
    mood: str
    subtitle: str
    transition: str = "crossfade"
    source_hint: str = ""      # 建议文件名（供match优先搜索）
    clip_role: str = ""        # highlight_source | transition | feature | intro_outro
    clip_id: str = ""
    src_start: float = 0.0
    src_end: float = 0.0


@dataclass
class ScriptResult:
    project: str
    idea: str
    total_duration: float
    has_materials: bool
    shots: list[ShotScript] = field(default_factory=list)
    llm_provider: str = ""
    llm_model: str = ""


def _build_materials_summary(materials_path: Path) -> Optional[str]:
    """Build a compact summary of materials for the LLM prompt."""
    if not materials_path.exists():
        return None
    with open(materials_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    files = data.get("files", [])
    if not files:
        return None

    lines = ["## 可用素材概览（含内容分类，供 source_hint 参考）"]
    for f in files:
        name = f["file"]
        dur = f.get("duration", 0)
        clips = f.get("clips", [])
        # Collect mood distribution and content types from analysis
        moods: dict[str, int] = {}
        ct_counts: dict[str, int] = {}
        style_counts: dict[str, int] = {}
        samples: list[str] = []
        for c in clips[:10]:
            a = c.get("analysis") or c.get("vision") or {}
            m = a.get("情绪标签") or a.get("mood", "")
            if m:
                for part in m.replace("/", " ").split():
                    if part.strip():
                        moods[part.strip()] = moods.get(part.strip(), 0) + 1
            ct = a.get("内容类型", "")
            if ct:
                ct_counts[ct] = ct_counts.get(ct, 0) + 1
            st = a.get("具体风格", "")
            if st:
                style_counts[st] = style_counts.get(st, 0) + 1
            summary = a.get("场景描述") or a.get("summary", "")
            if summary and len(samples) < 3:
                samples.append(summary)
        mood_str = " | ".join(f"{m} {cnt}/{len(clips)}" for m, cnt in sorted(moods.items(), key=lambda x: -x[1]))
        ct_str = ", ".join(f"{k}({v})" for k, v in sorted(ct_counts.items(), key=lambda x: -x[1]))
        style_str = ", ".join(f"{k}({v})" for k, v in sorted(style_counts.items(), key=lambda x: -x[1]))
        sample_str = " / ".join(samples[:3])
        lines.append(f"文件: {name} ({dur:.0f}s, {len(clips)}片段)")
        if ct_str:
            lines.append(f"  内容类型: {ct_str}")
        if style_str:
            lines.append(f"  风格: {style_str}")
        if mood_str:
            lines.append(f"  情绪分布: {mood_str}")
        if sample_str:
            lines.append(f"  代表性画面: {sample_str}")

    return "\n".join(lines)


def generate_script(
    idea: str,
    project_dir: Path,
    tier: Tier,
    api: ApiConfig,
    *,
    verbose: bool = True,
) -> Optional[ScriptResult]:
    scfg = tier.script
    temperature = scfg.temperature
    max_tokens = scfg.max_tokens
    max_shots = scfg.max_shots
    max_total = scfg.max_total_duration

    # Check materials
    materials_path = project_dir / "materials.json"
    has_materials = materials_path.exists()
    materials_summary = _build_materials_summary(materials_path) if has_materials else None

    # Build user prompt
    if has_materials and materials_summary:
        user_prompt = f"""## 我的想法
{idea}

{materials_summary}

## 创作要求
- 根据素材的文件名、内容类型、情绪分布和代表性画面设计分镜
- source_hint: 根据素材概览中的文件内容类型，为你认为最适合该镜头的素材文件填上文件名（如"红警全程"），不确定可填空字符串
- clip_role: 根据素材用途分配合适角色（highlight_source/transition/feature/intro_outro）
- 情绪标签从素材档案中已有的标签中选择
- 按短视频节奏编排：钩子开头 → 发展 → 高潮 → 反转/结尾

## 输出要求
输出上述 JSON 格式，仅输出 JSON，不要其他文字。"""
    else:
        user_prompt = f"""## 我的想法
{idea}

## 可用素材
（未提供素材）

## 要求
输出"理想素材需求清单"，让用户去录制/收集对应画面。

## 输出要求
输出上述 JSON 格式。画面需求字段填"【需录制】+ 描述"，情绪填"-"。仅输出 JSON，不要其他文字。"""

    if verbose:
        print(f"ppclip script  \"{idea[:40]}{'...' if len(idea) > 40 else ''}\"")
        print(f"  素材: {'有' if has_materials else '无'}")

    # LLM call with fallback chain
    llm = get_llm_client(api, tier.features.use_ds, temperature=temperature, verbose=verbose)
    if llm is None:
        if verbose:
            print("  ⚠ 所有 LLM 均不可达，生成空模板")
        result = _empty_template(idea, has_materials, project_dir.name if project_dir.name else "")
        _write_script_md(result, project_dir)
        _write_script_json(result, project_dir)
        if verbose:
            print(f"  script.md + script.json 已输出")
        return result

    if verbose:
        print(f"  生成中... ({llm.provider_name}/{llm.model})")

    try:
        response = llm.client.chat.completions.create(
            model=llm.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        raw = response.choices[0].message.content or ""
    except Exception as e:
        if verbose:
            print(f"  ✗ LLM 调用失败: {e}")
        result = _empty_template(idea, has_materials, project_dir.name)
        _write_both(result, project_dir, verbose)
        return result

    # Parse JSON from LLM response
    script_data = _parse_json_response(raw)
    if script_data is None:
        if verbose:
            print("  ⚠ JSON 解析失败，使用空模板")
        result = _empty_template(idea, has_materials, project_dir.name)
        _write_both(result, project_dir, verbose)
        return result

    shots = _build_shots(script_data)
    if not shots:
        if verbose:
            print("  ⚠ 未解析到有效镜头，使用空模板")
        result = _empty_template(idea, has_materials, project_dir.name)
        _write_both(result, project_dir, verbose)
        return result

    total = sum(s.duration for s in shots)
    if verbose:
        print(f"  ✓ 生成 {len(shots)} 镜，总时长 {total:.0f}s")

    result = ScriptResult(
        project=project_dir.name,
        idea=idea,
        total_duration=total,
        has_materials=has_materials,
        shots=shots,
        llm_provider=llm.provider_name,
        llm_model=llm.model,
    )

    # Write outputs
    _write_both(result, project_dir, verbose=False)
    if verbose:
        mood_counts: dict[str, int] = {}
        for s in result.shots:
            m = s.mood or "-"
            mood_counts[m] = mood_counts.get(m, 0) + 1
        mood_summary = " | ".join(f"{m}×{c}" for m, c in sorted(mood_counts.items(), key=lambda x: -x[1]))
        print(f"  script.md + script.json 已输出")
        print()
        print(f"  ╔══════════════════════════════════════════╗")
        print(f"  ║ 检查分镜结果:  {project_dir.name}")
        print(f"  ║ {len(result.shots)} 镜 / {result.total_duration:.0f}s")
        print(f"  ║ 情绪: {mood_summary}")
        print(f"  ║ 编辑 script.json 后可重新 match")
        print(f"  ║ 编辑 script.json 后可重新 match")
        print(f"  ╚══════════════════════════════════════════╝")

    return result


def _parse_json_response(raw: str) -> Optional[dict]:
    """Try to extract JSON from LLM response with 3-level fallback."""
    # Level 1: Direct JSON parse
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    # Level 2: Extract JSON block from markdown
    m = re.search(r'```(?:json)?\s*([\s\S]*?)```', raw)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # Level 3: Try to find { ... } in the text
    m = re.search(r'\{[\s\S]*"shots"[\s\S]*\}', raw)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return None


def _build_shots(data: dict) -> list[ShotScript]:
    shots: list[ShotScript] = []
    for s in data.get("shots", []):
        shots.append(ShotScript(
            id=int(s.get("id", len(shots) + 1)),
            duration=float(s.get("duration", 3.0)),
            need=str(s.get("need", "")),
            mood=str(s.get("mood", "平静")),
            source_hint=str(s.get("source_hint", "")),
            clip_role=str(s.get("clip_role", "")),
            subtitle=str(s.get("subtitle", "")),
            transition=str(s.get("transition", "crossfade")),
        ))
    return shots


def _write_both(result: ScriptResult, project_dir: Path, verbose: bool) -> None:
    _write_script_md(result, project_dir)
    _write_script_json(result, project_dir)
    if verbose:
        print(f"  script.md + script.json 已输出")


def _empty_template(idea: str, has_materials: bool, project: str = "") -> ScriptResult:
    shots = [
        ShotScript(id=1, duration=3, need="【需录制】开场钩子" if not has_materials else "开场画面", mood="平静", subtitle="开场文案占位"),
        ShotScript(id=2, duration=5, need="【需录制】主体内容" if not has_materials else "主体画面", mood="激烈", subtitle="主体文案占位"),
        ShotScript(id=3, duration=4, need="【需录制】结尾反转" if not has_materials else "结尾画面", mood="搞笑", subtitle="结尾文案占位"),
    ]
    return ScriptResult(project=project, idea=idea, total_duration=12.0, has_materials=has_materials, shots=shots)


def _write_script_md(result: ScriptResult, project_dir: Path) -> None:
    lines = [
        f"# {project_dir.name} - 分镜脚本",
        "",
        f"> 创意: {result.idea}",
        f"> 总时长: {result.total_duration:.0f}s | 镜头: {len(result.shots)}",
        f"> 生成: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        f"> LLM: {result.llm_provider}/{result.llm_model}" if result.llm_provider else "",
        "",
        "| 镜号 | 时长 | 画面需求 | 情绪 | 素材 | 角色 | 字幕 | 转场 |",
        "|------|------|---------|------|------|------|------|------|",
    ]
    for s in result.shots:
        src = s.source_hint or "-"
        role = s.clip_role or "-"
        lines.append(f"| {s.id} | {s.duration:.0f}s | {s.need} | {s.mood} | {src} | {role} | {s.subtitle} | {s.transition} |")
    (project_dir / "script.md").write_text("\n".join(lines), encoding="utf-8")


def _write_script_json(result: ScriptResult, project_dir: Path) -> None:
    data = {
        "version": 1,
        "project": project_dir.name,
        "idea": result.idea,
        "total_duration": result.total_duration,
        "has_materials": result.has_materials,
        "llm_provider": result.llm_provider,
        "llm_model": result.llm_model,
        "shots": [
            {
                "id": s.id,
                "duration": s.duration,
                "need": s.need,
                "mood": s.mood,
                "source_hint": s.source_hint or "",
                "clip_role": s.clip_role or "",
                "subtitle": s.subtitle,
                "transition": s.transition,
                "clip_id": s.clip_id or "",
                "src_start": s.src_start,
                "src_end": s.src_end,
            }
            for s in result.shots
        ],
    }
    (project_dir / "script.json").write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
