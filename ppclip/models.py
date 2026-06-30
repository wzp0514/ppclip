"""ppclip models — LLM 和 Vision 客户端，各自含降级链"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from openai import OpenAI

from .config import ApiConfig


@dataclass
class ProviderStatus:
    name: str
    model: str
    connected: bool = False
    error: Optional[str] = None


@dataclass
class LLMChainResult:
    client: OpenAI
    model: str
    provider_name: str
    chain_log: list[ProviderStatus] = field(default_factory=list)


def _build_llm_chain(api: ApiConfig, use_ds: bool) -> list[dict]:
    chain = []
    if use_ds and api.text_key_1:
        chain.append({"name": "LLM-1", "key": api.text_key_1, "base_url": api.text_url_1, "model": api.text_model_1})
    if api.text_key_2:
        chain.append({"name": "LLM-2", "key": api.text_key_2, "base_url": api.text_url_2, "model": api.text_model_2})
    if api.text_key_3:
        chain.append({"name": "LLM-3", "key": api.text_key_3, "base_url": api.text_url_3, "model": api.text_model_3})
    return chain


def _build_vision_chain(api: ApiConfig) -> list[dict]:
    chain = []
    if api.image_key_1:
        chain.append({"name": "Vision-1", "key": api.image_key_1, "base_url": api.text_url_2, "model": api.image_model_1})
    if api.image_key_2:
        chain.append({"name": "Vision-2", "key": api.image_key_2, "base_url": api.text_url_3, "model": api.image_model_2})
    return chain


def _ping_provider(provider: dict, timeout: float = 10.0) -> ProviderStatus:
    status = ProviderStatus(name=provider["name"], model=provider["model"])
    try:
        client = OpenAI(api_key=provider["key"], base_url=provider["base_url"], timeout=timeout)
        client.chat.completions.create(
            model=provider["model"],
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1,
        )
        status.connected = True
    except Exception as e:
        status.error = _classify_error(e)
    return status


def _classify_error(e: Exception) -> str:
    s = str(e).lower()
    if "401" in s or "403" in s or "unauthorized" in s or "invalid" in s:
        return "API key 无效或已过期"
    if "timeout" in s:
        return "超时"
    if "dns" in s or "resolve" in s or "name or service not known" in s:
        return "DNS 解析失败"
    return str(e)[:100]


def get_llm_client(
    api: ApiConfig,
    use_ds: bool,
    *,
    temperature: float = 0.7,
    verbose: bool = True,
) -> Optional[LLMChainResult]:
    chain = _build_llm_chain(api, use_ds)
    if not chain:
        if verbose:
            print("[ppclip] 未配置任何 LLM API key，LLM 功能不可用")
        return None

    log: list[ProviderStatus] = []
    for provider in chain:
        status = _ping_provider(provider)
        log.append(status)
        if status.connected:
            if verbose:
                label = "[OK]" if status.name == chain[0]["name"] else "[DEGRADE]"
                print(f"  {label} LLM: {status.name} ({status.model})")
            return LLMChainResult(
                client=OpenAI(api_key=provider["key"], base_url=provider["base_url"]),
                model=provider["model"],
                provider_name=status.name,
                chain_log=log,
            )
        else:
            if verbose:
                print(f"  [SKIP] {status.name} unreachable({status.error}), next...")

    if verbose:
        print("[ppclip] 所有 LLM 均不可达，LLM 功能降级为手动模式")
    return None


def get_vision_client(
    api: ApiConfig,
    *,
    verbose: bool = True,
) -> Optional[LLMChainResult]:
    chain = _build_vision_chain(api)
    if not chain:
        if verbose:
            print("[ppclip] 未配置任何 Vision API key，自动切换无视觉模式")
        return None

    log: list[ProviderStatus] = []
    for provider in chain:
        status = _ping_provider(provider)
        log.append(status)
        if status.connected:
            if verbose:
                label = "[OK]" if status.name == chain[0]["name"] else "[DEGRADE]"
                print(f"  {label} Vision: {status.name} ({status.model})")
            return LLMChainResult(
                client=OpenAI(api_key=provider["key"], base_url=provider["base_url"]),
                model=provider["model"],
                provider_name=status.name,
                chain_log=log,
            )
        else:
            if verbose:
                print(f"  [SKIP] {status.name} unreachable({status.error}), next...")

    if verbose:
        print("[ppclip] 所有 Vision 模型均不可达，自动切换无视觉模式")
    return None
