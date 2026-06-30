"""ppclip local OCR — PaddleOCR 文字提取，纯CPU，轻量级"""

from __future__ import annotations

import os
from pathlib import Path

# 模型下载到 ppclip/models/ 下（PaddleX 自动管理 official_models/ 子目录），不污染 C 盘
_OCR_MODEL_DIR = Path(__file__).parent / "models" / "paddleocr"
os.environ.setdefault("PADDLEX_HOME", str(_OCR_MODEL_DIR.parent))

_OCR_INSTANCE = None
_OCR_FAILED = False  # 永久失败标记，避免重复初始化


def _get_ocr(lang: str = "ch"):
    global _OCR_INSTANCE, _OCR_FAILED
    if _OCR_FAILED:
        return None
    if _OCR_INSTANCE is not None:
        return _OCR_INSTANCE
    try:
        from paddleocr import PaddleOCR
        _OCR_INSTANCE = PaddleOCR(use_angle_cls=True, lang=lang, show_log=False)
    except Exception:
        _OCR_FAILED = True
        return None
    return _OCR_INSTANCE


def ocr_extract(image_path: Path, lang: str = "ch") -> str:
    """Extract visible text from an image. Returns empty string on failure."""
    if not image_path.exists():
        return ""
    ocr = _get_ocr(lang)
    if ocr is None:
        return ""
    try:
        result = ocr.ocr(str(image_path), cls=True)
        if not result or not result[0]:
            return ""
        texts = [line[1][0] for line in result[0] if line[1][0].strip()]
        return " ".join(texts)
    except Exception:
        return ""
