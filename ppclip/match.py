"""ppclip match — LLM 降级链逐镜素材匹配 + 交叉验证 → match_report.md + timeline.json"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .config import ApiConfig, Tier
from .models import get_llm_client, LLMChainResult

MATCH_SYSTEM_PROMPT = """你是视频素材匹配助手。根据镜头的画面需求和情绪标签，从素材档案中选择最匹配的片段。

## 选择规则
- 画面内容与"画面需求"语义匹配度最高优先
- 情绪标签一致优先
- usable=true 的片段优先
- 源起止时间在片段时长范围内
- 如果没有合适的片段（所有候选都不像），返回 needs_review: true

## 输出格式（仅输出此 JSON，不要其他文字）
```json
{
  "shot_id": 1,
  "selected": {"clip_id": "xxx_001", "src_start": 0.0, "src_end": 3.0, "confidence": 0.85, "reason": "匹配理由"},
  "alternatives": [{"clip_id": "xxx_002", "src_start": 0.0, "src_end": 3.0, "confidence": 0.60}],
  "needs_review": false
}
```
"""


@dataclass
class MatchEntry:
    shot_id: int
    clip_id: str
    src_start: float
    src_end: float
    confidence: float
    reason: str
    needs_review: bool = False
    alternatives: list[dict] = field(default_factory=list)
    # Cross-validation
    time_range_valid: bool = True
    file_exists: bool = True
    mood_match: str = "unknown"
    duration_sufficient: bool = True
    objective_score: float = 0.0


def _candidate_summary(clip: dict, index: int) -> str:
    v = clip.get("vision")
    if v:
        return (
            f"[{index}] {clip['id']} | {clip['duration']:.1f}s | "
            f"画面: {v.get('summary', '?')} | 情绪: {v.get('mood', '?')} | "
            f"可用: {v.get('usable', True)}"
        )
    return f"[{index}] {clip['id']} | {clip['duration']:.1f}s | 无视觉标注"


def run_match(
    project_dir: Path,
    tier: Tier,
    api: ApiConfig,
    *,
    verbose: bool = True,
) -> Optional[list[MatchEntry]]:
    mcfg = tier.match
    threshold = mcfg.confidence_threshold
    max_candidates = mcfg.max_candidates_per_shot
    temperature = mcfg.temperature
    skip_llm = mcfg.skip_llm
    llm_candidate_limit = mcfg.match_llm_candidate_limit

    # Load data
    script_path = project_dir / "script.json"
    materials_path = project_dir / "materials.json"
    if not script_path.exists():
        print(f"✗ script.json 不存在，请先运行 ppclip script")
        return None

    with open(script_path, "r", encoding="utf-8") as f:
        script = json.load(f)
    shots = script.get("shots", [])
    if not shots:
        print("✗ 脚本无镜头")
        return None

    has_materials = materials_path.exists()
    if not has_materials:
        print("✗ materials.json 不存在，请先运行 ppclip index")
        return None

    with open(materials_path, "r", encoding="utf-8") as f:
        materials = json.load(f)

    all_clips: list[dict] = []
    audio_clips: list[dict] = []
    for f in materials.get("files", []):
        file_type = f.get("media_type", "video")
        for c in f.get("clips", []):
            c["_file_path"] = f["path"]
            c["_media_type"] = c.get("media_type", file_type)
            if c["_media_type"] == "audio":
                audio_clips.append(c)
            else:
                all_clips.append(c)

    if not all_clips:
        print("✗ 素材列表为空")
        return None

    if verbose:
        print(f"ppclip match")
        print(f"  镜头: {len(shots)} | 候选片段: {len(all_clips)}")

    # LLM (skip if tier disables per-shot LLM)
    llm = None if skip_llm else get_llm_client(api, tier.features.use_ds, temperature=temperature, verbose=verbose)
    if skip_llm and verbose:
        print("[ppclip] match skip_llm=true: 使用文件名启发式匹配，不调LLM")

    entries: list[MatchEntry] = []
    used_clip_ids: set[str] = set()  # Track used clips for diversity
    checkpoint_file = project_dir / ".checkpoint.json"
    completed_shots = _load_checkpoint(checkpoint_file, len(shots))

    for i, shot in enumerate(shots):
        sid = shot.get("id")
        if sid is None:
            if verbose:
                print(f"  [warn] 镜 {i + 1}: 缺少 id 字段，跳过")
            continue
        if sid in completed_shots:
            if verbose:
                print(f"  镜 {sid} [跳过，已完成]")
            continue

        if verbose:
            need_short = shot.get('need', '?')[:40]
            print(f"  镜 {sid}/{len(shots)}: {need_short}...")

        # Build candidates (with source_hint priority + diversity)
        source_hint = shot.get("source_hint", "")
        candidates = _filter_candidates(shot, all_clips, max_candidates, used_clip_ids, source_hint=source_hint)
        if verbose:
            top_score = 0.0
            if candidates:
                # Show top candidates summary
                top_names = set()
                for c in candidates[:5]:
                    cid = c.get("id", "")
                    top_names.add(cid.rsplit("_", 1)[0] if "_" in cid else cid[:20])
                print(f"    候选: {len(candidates)}片段 (来自 {len(top_names)} 文件)")

        if not candidates:
            entries.append(MatchEntry(
                shot_id=sid, clip_id="", src_start=0, src_end=0,
                confidence=0, reason="无匹配候选", needs_review=True,
            ))
            _save_checkpoint(checkpoint_file, sid, len(shots))
            continue

        # LLM match
        entry = _match_shot_llm(shot, candidates, llm, temperature, llm_candidate_limit) if llm else _match_shot_fallback(shot, candidates)

        # Cross-validation
        _cross_validate(entry, all_clips, shot)

        if entry.confidence < threshold or not entry.time_range_valid or not entry.file_exists:
            entry.needs_review = True

        if verbose:
            cid = entry.clip_id or "(无)"
            src = f"{entry.src_start:.1f}-{entry.src_end:.1f}s" if entry.clip_id else ""
            obj = entry.objective_score
            status = "[NEEDS_REVIEW]" if entry.needs_review else "[OK]"
            print(f"    -> {cid} {src} conf={entry.confidence:.2f} obj={obj:.1f} {status}")

        entries.append(entry)
        if entry.clip_id:
            used_clip_ids.add(entry.clip_id)  # Mark used for diversity
        _save_checkpoint(checkpoint_file, sid, len(shots))

    # Write outputs
    _write_match_report(entries, shots, project_dir, llm)
    _write_timeline(entries, shots, project_dir, llm)

    needs = sum(1 for e in entries if e.needs_review)
    if verbose:
        print()
        print(f"匹配完成: {len(entries)}镜, needs_review: {needs}/{len(entries)}")
        print(f"  match_report.md + timeline.json 已输出")
        # Per-shot source summary
        for e in entries:
            cid = e.clip_id or "-"
            src_file = cid.rsplit("_", 1)[0] if "_" in cid else "-"
            review = " ⚠REVIEW" if e.needs_review else ""
            print(f"  镜{e.shot_id} → {cid} ({src_file}) conf={e.confidence:.2f}{review}")
        print()
        print(f"  ╔══════════════════════════════════════════╗")
        print(f"  ║ 检查匹配结果:  {project_dir.name}")
        print(f"  ║ {len(entries)}镜 / needs_review: {needs}")
        print(f"  ║ 编辑 script.json 可重新 match")
        print(f"  ╚══════════════════════════════════════════╝")

    return entries


def _filter_candidates(shot: dict, clips: list[dict], max_candidates: int,
                       used_clip_ids: set | None = None,
                       source_hint: str = "") -> list[dict]:
    """多维度预过滤: 8维标签 + 情绪 + 关键词 + 时长 + 多样性 + source_hint 评分排序。"""
    need = (shot.get("need", "") or "").lower()
    mood = shot.get("mood", "")
    dur = shot.get("duration", 0.0)
    keywords = [w for w in need.split() if len(w) >= 2]
    used = used_clip_ids or set()
    used_files = {c.split("_")[0] for c in used if "_" in c}
    hint_lower = source_hint.lower()

    # 从 shot.need 推断内容类型偏好
    _need_ct = _infer_content_type(need)

    # source_hint 硬过滤: 指定文件时仅保留该文件候选
    if hint_lower:
        filtered = []
        for c in clips:
            fp = c.get("_file_path", "") or c.get("id", "")
            fn = Path(fp).stem.lower() if fp else ""
            if hint_lower in fn:
                filtered.append(c)
        if filtered:
            clips = filtered  # 仅在指定文件内搜索
        # 如果没有匹配文件，回退全局搜索（source_hint 可能指向不存在的文件）

    scored: list[tuple[float, dict]] = []
    for c in clips:
        score = 0.0
        cid = c.get("id", "")
        cfile = cid.split("_")[0] if "_" in cid else cid

        # ── v2: 结构化分析标签评分 ──
        a = c.get("analysis")
        if a:
            # 内容类型 硬匹配
            ct = a.get("内容类型", "")
            if _need_ct and ct:
                if _need_ct == ct:
                    score += 4.0     # 精确匹配
                elif _need_ct in ct or ct in _need_ct:
                    score += 2.0     # 部分匹配

            # 具体风格 匹配
            style = a.get("具体风格", "")
            if style:
                style_lower = style.lower()
                if any(kw in style_lower for kw in keywords):
                    score += 2.0

            # 制作质量: 专业制作 > 业余拍摄
            pq = a.get("制作质量", "")
            if pq and "professional" in pq:
                score += 1.5
            elif pq and "amateur" in pq:
                score += 0.3

            # 主体对象 关键词匹配（每个命中 +1.2，上限 6）
            obj_hits = 0
            for obj in a.get("主体对象", []):
                if any(kw in obj.lower() for kw in keywords):
                    obj_hits += 1
            score += min(obj_hits * 1.2, 6.0)

            # 动作描述 关键词匹配（每个命中 +1.5，上限 3）
            act_hits = 0
            for act in a.get("动作描述", []):
                if any(kw in act.lower() for kw in keywords):
                    act_hits += 1
            score += min(act_hits * 1.5, 3.0)

            # 画面文字 关键词匹配
            tv = a.get("画面文字", "").lower()
            if tv:
                tv_hits = sum(1 for kw in keywords if kw in tv)
                score += min(tv_hits * 1.5, 3.0)

            # 叙事阶段 匹配
            phase = a.get("叙事阶段", "").lower()
            if phase:
                if any(kw in phase for kw in keywords):
                    score += 1.0

            # 情绪标签
            am = a.get("情绪标签", "")
            if am:
                am_parts = set(am.replace("/", " ").split())
                if mood in am_parts or am == mood:
                    score += 3.0
                elif am_parts and any(m in mood or mood in m for m in am_parts):
                    score += 1.5

            # 可用性
            if a.get("可用", True):
                score += 1.0

            # 构建 search_text 用于通用关键词匹配
            search_text = " ".join(a.get("主体对象", []) + a.get("动作描述", [])).lower()
            search_text += " " + a.get("场景描述", "").lower()
            search_text += " " + a.get("异常标注", "").lower()
        else:
            search_text = ""

        # ── v1: Vision-based scoring (向后兼容) ──
        v = c.get("vision")
        if v:
            vm = v.get("mood", "")
            vm_parts = set(vm.replace("/", " ").split())
            if not a:  # only if analysis didn't already score mood
                if mood in vm_parts or vm == mood:
                    score += 3.0
                elif vm_parts and any(m in mood or mood in m for m in vm_parts):
                    score += 1.5
            search_text += " " + (v.get("summary", "") or "").lower() + " " + \
                          " ".join(v.get("elements", []) or []).lower()
            if v.get("usable", True) and not a:
                score += 1.0

        # ── Filename keyword matching (always available) ──
        file_path = c.get("_file_path", "") or cid
        file_name = Path(file_path).stem.lower() if file_path else cid.lower()
        search_text += " " + file_name
        for word in keywords:
            if word in search_text:
                score += 1.0

        # ── Audio event scoring ──
        audio_events = c.get("audio_events") or []
        if audio_events:
            non_silence = [e for e in audio_events if e.get("type") != "silence"]
            if non_silence:
                score += 0.5

        # ── Duration proximity ──
        cdur = c.get("duration", 0.0)
        if cdur > 0 and dur > 0:
            ratio = min(cdur, dur) / max(cdur, dur)
            score += ratio * 2.0

        # ── Diversity ──
        if cid in used:
            score -= 3.0
        if cfile in used_files and cid not in used:
            score -= 0.5

        scored.append((score, c))

    scored.sort(key=lambda x: -x[0])
    return [c for _, c in scored[:max_candidates]]


def _infer_content_type(need: str) -> str:
    """从 shot.need 文本推断期望的 content_type"""
    need_lower = need.lower()
    if any(kw in need_lower for kw in ["游戏", "gameplay", "电竞", "网游", "手游"]):
        return "video game"
    if any(kw in need_lower for kw in ["动画", "卡通", "动漫", "animation", "cartoon"]):
        return "animation"
    if any(kw in need_lower for kw in ["vlog", "自拍", "日常", "生活"]):
        return "real-world footage"
    if any(kw in need_lower for kw in ["cgi", "特效", "3d", "渲染"]):
        return "CGI"
    return ""


def _match_shot_llm(shot: dict, candidates: list[dict], llm: LLMChainResult,
                     temperature: float = 0.3, llm_candidate_limit: int = 20) -> MatchEntry:
    sid = shot["id"]
    need = shot.get("need", "")
    mood = shot.get("mood", "")
    dur = shot.get("duration", 3.0)

    # Build prompt with top N candidates (context window limit)
    top = candidates[:llm_candidate_limit]
    clip_lines = "\n".join(_candidate_summary(c, i) for i, c in enumerate(top))
    user_prompt = f"""## 当前镜头
- 画面需求: {need}
- 期望情绪: {mood}
- 需要时长: {dur}s

## 素材档案
{clip_lines}

## 要求
从上述素材中选择最匹配的片段。仅输出 JSON。"""

    try:
        response = llm.client.chat.completions.create(
            model=llm.model,
            messages=[
                {"role": "system", "content": MATCH_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=1024,
        )
        raw = response.choices[0].message.content or ""
    except Exception as e:
        import sys
        print(f"    [warn] LLM 调用失败，降级为启发式匹配: {e}", file=sys.stderr)
        return _match_shot_fallback(shot, candidates)

    data = _parse_json_raw(raw)
    if data is None:
        return _match_shot_fallback(shot, candidates)

    sel = data.get("selected") or {}
    return MatchEntry(
        shot_id=sid,
        clip_id=sel.get("clip_id", ""),
        src_start=float(sel.get("src_start", 0)),
        src_end=float(sel.get("src_end", dur)),
        confidence=float(sel.get("confidence", 0.5)),
        reason=str(sel.get("reason", "")),
        needs_review=data.get("needs_review", False),
        alternatives=data.get("alternatives", [])[:2],
    )


def _match_shot_fallback(shot: dict, candidates: list[dict]) -> MatchEntry:
    """No-LLM fallback: pick first matching candidate."""
    if candidates:
        c = candidates[0]
        return MatchEntry(
            shot_id=shot["id"],
            clip_id=c["id"],
            src_start=c.get("start", 0),
            src_end=min(c.get("end", 3.0), c.get("start", 0) + shot.get("duration", 3.0)),
            confidence=0.3,
            reason="无 LLM，自动选第一个候选",
            needs_review=True,
        )
    return MatchEntry(
        shot_id=shot["id"],
        clip_id="", src_start=0, src_end=0,
        confidence=0, reason="无可用素材", needs_review=True,
    )


def _cross_validate(entry: MatchEntry, all_clips: list[dict], shot: dict) -> None:
    clip_map = {c["id"]: c for c in all_clips if "id" in c}
    if not entry.clip_id or entry.clip_id not in clip_map:
        entry.file_exists = False
        entry.time_range_valid = False
        return

    c = clip_map[entry.clip_id]
    # Time range check
    if not (0 <= entry.src_start < entry.src_end <= c.get("duration", 0)):
        entry.time_range_valid = False
    # File exists check
    fp = c.get("_file_path", "")
    if fp and not Path(fp).exists():
        entry.file_exists = False
    # Mood match — supports compound moods like "激烈/紧张"
    v = c.get("vision")
    if v:
        cm = v.get("mood", "")
        sm = shot.get("mood", "")
        cm_parts = set(cm.replace("/", " ").split())
        sm_parts = set(sm.replace("/", " ").split())
        if cm == sm or (cm_parts and cm_parts == sm_parts):
            entry.mood_match = "exact"
        elif cm_parts & sm_parts:
            entry.mood_match = "partial"
        else:
            entry.mood_match = "mismatch"
    # Duration sufficient
    if c.get("duration", 0) < shot.get("duration", 0):
        entry.duration_sufficient = False
    # Objective score: meaningful basis for evaluation
    score = 0.0
    if entry.time_range_valid:
        score += 0.3
    if entry.file_exists:
        score += 0.2
    if entry.mood_match == "exact":
        score += 0.2
    elif entry.mood_match == "partial":
        score += 0.1
    elif entry.mood_match == "mismatch":
        score += 0.0  # explicit penalty
    # unknown mood means no vision data — neutral, neither bonus nor penalty
    if entry.duration_sufficient:
        score += 0.15
    # Heuristic match (no LLM) gets lower ceiling
    if entry.confidence <= 0.3:  # fallback/filename match
        score = min(score, 0.5)
    entry.objective_score = score


def _parse_json_raw(raw: str) -> Optional[dict]:
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
    m = re.search(r'\{[\s\S]*"selected"[\s\S]*\}', raw)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return None


def _write_match_report(entries: list[MatchEntry], shots: list[dict], project_dir: Path, llm: Optional[LLMChainResult]) -> None:
    lines = [
        f"# {project_dir.name} - 素材匹配报告",
        "",
        f"| 镜号 | 片段ID | 源起止 | 置信度 | 客观分 | 情绪 | Review | 理由 |",
        f"|------|--------|--------|--------|--------|------|--------|------|",
    ]
    for e in entries:
        nr = "⚠" if e.needs_review else "✓"
        lines.append(
            f"| {e.shot_id} | {e.clip_id or '-'} | {e.src_start:.1f}-{e.src_end:.1f} | "
            f"{e.confidence:.2f} | {e.objective_score:.2f} | {e.mood_match} | {nr} | "
            f"{e.reason[:40] if e.reason else '-'} |"
        )
    (project_dir / "match_report.md").write_text("\n".join(lines), encoding="utf-8")


def _write_timeline(entries: list[MatchEntry], shots: list[dict], project_dir: Path, llm: Optional[LLMChainResult]) -> None:
    shot_map = {s["id"]: s for s in shots}
    data = {
        "version": 1,
        "project": project_dir.name,
        "total_duration": sum(
            shot_map[e.shot_id].get("duration", 0) if e.shot_id in shot_map else 0
            for e in entries
        ),
        "llm_provider": llm.provider_name if llm else "fallback",
        "llm_model": llm.model if llm else "",
        "shots": [
            {
                "id": e.shot_id,
                "duration": shot_map[e.shot_id].get("duration", 0) if e.shot_id in shot_map else 0,
                "clip_id": e.clip_id,
                "src_start": e.src_start,
                "src_end": e.src_end,
                "subtitle": shot_map[e.shot_id].get("subtitle", "") if e.shot_id in shot_map else "",
                "mood": shot_map[e.shot_id].get("mood", "") if e.shot_id in shot_map else "",
                "clip_role": shot_map[e.shot_id].get("clip_role", "") if e.shot_id in shot_map else "",
                "transition": shot_map[e.shot_id].get("transition", "crossfade") if e.shot_id in shot_map else "crossfade",
                "match_confidence": e.confidence,
                "match_reason": e.reason,
                "match_validation": {
                    "time_range_valid": e.time_range_valid,
                    "file_exists": e.file_exists,
                    "mood_match": e.mood_match,
                    "duration_sufficient": e.duration_sufficient,
                    "objective_score": e.objective_score,
                },
            }
            for e in entries
        ],
    }
    (project_dir / "timeline.json").write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _load_checkpoint(path: Path, shot_count: int) -> set[int]:
    if not path.exists():
        return set()
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        completed = set(data.get("completed_shots", []))
        # Stale detection: if completed shots > current script shots, checkpoint is stale
        if completed and max(completed) > shot_count:
            return set()
        return completed
    except (json.JSONDecodeError, OSError, ValueError):
        return set()


def _save_checkpoint(path: Path, shot_id: int, shot_count: int = 999) -> None:
    completed = _load_checkpoint(path, shot_count)
    completed.add(shot_id)
    data = {
        "step": "match",
        "total_shots": shot_count,
        "completed_shots": sorted(completed),
        "failed_shots": {},
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
