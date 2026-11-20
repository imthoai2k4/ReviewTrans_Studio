"""Thinking controls for OpenAI-compatible endpoints, shared with the editor."""
from urllib.parse import urlparse

from ...config import THINKING_OPTIONS
from .. import ProviderError


def protocol(profile):
    host = urlparse(profile.base_url or "https://api.openai.com/v1").hostname
    if host == "openrouter.ai":
        return "openrouter"
    if host == "api.deepseek.com":
        return "deepseek"
    if host == "api.openai.com":
        return "openai"
    return "compat"


def supported_modes(profile, model):
    route = protocol(profile)
    if route == "openrouter":
        catalog = profile.options.get("reasoning_catalog", {})
        data = catalog.get(model) if profile.options.get("reasoning_catalog_url") == profile.base_url else None
        if data is None:
            return {""}
        modes = {""}
        if data.get("supported"):
            modes.add("enabled")
            if not data.get("required"):
                modes.add("disabled")
            modes.update(set(data.get("efforts", [])) & {"low", "medium", "high"})
        return modes
    if route == "deepseek":
        modes = {"", "disabled", "enabled"}
        if model.startswith("deepseek-v4") or model == "deepseek-flash":
            modes.update(("low", "high"))
        return modes
    if route == "openai":
        if model.startswith(("o3", "o4-mini")) and not model.startswith("o3-pro"):
            return {"", "enabled", "low", "medium", "high"}
        return {""}
    return {"", "disabled", "enabled"}


def selectable_modes(profile, model):
    if protocol(profile) != "openrouter":
        return supported_modes(profile, model)
    # OpenRouter accepts a unified reasoning configuration. Missing model
    # metadata is not evidence that the user cannot request an effort level.
    modes = {value for value, _label in THINKING_OPTIONS}
    if profile.options.get("reasoning_catalog_url") == profile.base_url:
        data = profile.options.get("reasoning_catalog", {}).get(model) or {}
        if data.get("required"):
            modes.discard("disabled")
    return modes


def request_options(profile, model, mode):
    if not mode:
        return {}
    if mode not in selectable_modes(profile, model):
        raise ProviderError(
            f"Chưa xác nhận model {model} hỗ trợ thinking '{dict(THINKING_OPTIONS).get(mode, mode)}'. "
            "Tải danh sách model trong Cấu hình API hoặc chọn Theo mặc định API."
        )
    route = protocol(profile)
    if route == "openrouter":
        reasoning = {"enabled": mode == "enabled"} if mode in ("enabled", "disabled") else {"effort": mode}
        return {"extra_body": {"reasoning": reasoning}}
    if route == "openai":
        return {"reasoning_effort": "medium" if mode == "enabled" else mode}
    result = {"extra_body": {"thinking": {"type": "disabled" if mode == "disabled" else "enabled"}}}
    if mode in ("low", "medium", "high"):
        result["reasoning_effort"] = mode
    return result
