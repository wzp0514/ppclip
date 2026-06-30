"""ppclip local classifier — MobileCLIP 零样本分类，纯CPU

对缩略图做三路零样本分类：content_type / specific_style / production_quality。
API 不可达时作为降级补位通道。

标签体系对齐 ppclip 的 13 维中文 analysis schema，输出英文值由调用方直接写入。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

MODEL_DIR = Path(__file__).parent / "models" / "mobileclip"

# ── 三路零样本标签 ──

_CONTENT_TYPE_LABELS = [
    "video game", "real-world footage", "animation", "cartoon", "CGI", "VTuber",
]

_SPECIFIC_STYLE_LABELS = [
    "gameplay", "vlog", "tutorial", "anime", "3D animation",
    "news", "film", "TV", "mobile recording",
]

_PRODUCTION_QUALITY_LABELS = [
    "professional studio", "amateur handheld",
    "screen recording", "webcam recording", "TV broadcast",
]

# ── 全局单例 ──

_CLASSIFIER_INSTANCE = None


@dataclass
class ClassifierResult:
    content_type: str = ""
    content_type_conf: float = 0.0
    specific_style: str = ""
    specific_style_conf: float = 0.0
    production_quality: str = ""
    production_quality_conf: float = 0.0

    @classmethod
    def empty(cls):
        return cls()


def _get_classifier(model_name: str = "MobileCLIP-S2", device: str = "cpu",
                    hf_endpoint: str = ""):
    global _CLASSIFIER_INSTANCE
    if _CLASSIFIER_INSTANCE is not None:
        return _CLASSIFIER_INSTANCE

    # Set HF mirror before any open_clip import (only on first load)
    if hf_endpoint:
        os.environ.setdefault("HF_ENDPOINT", hf_endpoint)

    model_path = MODEL_DIR / "open_clip_model.safetensors"
    if not model_path.exists():
        return None

    try:
        import open_clip
        import torch

        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=str(model_path)
        )
        model = model.to(device)
        model.eval()
        tokenizer = open_clip.get_tokenizer(model_name)

        _CLASSIFIER_INSTANCE = {
            "model": model,
            "tokenizer": tokenizer,
            "preprocess": preprocess,
            "device": device,
        }
        return _CLASSIFIER_INSTANCE
    except ImportError:
        return None
    except Exception:
        return None


def classify_frame(image_path: Path,
                   model_name: str = "MobileCLIP-S2",
                   device: str = "cpu",
                   hf_endpoint: str = "") -> ClassifierResult:
    """Classify a single keyframe — zero-shot on 3 dimensions.

    Returns ClassifierResult.empty() if model unavailable or inference fails.
    """
    if not image_path.exists():
        return ClassifierResult.empty()

    engine = _get_classifier(model_name, device, hf_endpoint=hf_endpoint)
    if engine is None:
        return ClassifierResult.empty()

    try:
        from PIL import Image
        import torch

        image = engine["preprocess"](Image.open(image_path)).unsqueeze(0).to(device)

        def _top1(labels: list[str]) -> tuple[str, float]:
            text = engine["tokenizer"](labels).to(device)
            with torch.no_grad():
                img_feat = engine["model"].encode_image(image)
                txt_feat = engine["model"].encode_text(text)
                img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
                txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
                probs = (100.0 * img_feat @ txt_feat.T).softmax(dim=-1)[0]
            idx = int(probs.argmax())
            return labels[idx], float(probs[idx])

        ct, ct_conf = _top1(_CONTENT_TYPE_LABELS)
        ss, ss_conf = _top1(_SPECIFIC_STYLE_LABELS)
        pq, pq_conf = _top1(_PRODUCTION_QUALITY_LABELS)

        return ClassifierResult(
            content_type=ct, content_type_conf=round(ct_conf, 4),
            specific_style=ss, specific_style_conf=round(ss_conf, 4),
            production_quality=pq, production_quality_conf=round(pq_conf, 4),
        )
    except Exception:
        return ClassifierResult.empty()
