"""Provider LLM: API kiểu OpenAI, API kiểu Gemini, Anthropic."""
from __future__ import annotations

from .. import ProviderError
from .base import LLMTranslator
from .thinking import protocol, request_options


def _temperature(profile) -> float:
    try:
        return float(profile.option("temperature", 0.3))
    except (TypeError, ValueError):
        return 0.3


class OpenAICompatTranslator(LLMTranslator):
    default_model = "gpt-4o-mini"

    def _call(self, system: str, prompt: str, key: str) -> str:
        import openai

        client = openai.OpenAI(
            api_key=key or "sk-no-key",
            base_url=self.profile.base_url or None,
            timeout=self.request_timeout,
            max_retries=0,
        )
        request = dict(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            temperature=_temperature(self.profile),
        )
        thinking = self.thinking_mode()
        request.update(request_options(self.profile, self.model, thinking))
        if thinking and thinking != "disabled" and protocol(self.profile) in ("openai", "deepseek"):
            request.pop("temperature", None)
        response = client.chat.completions.create(**request)
        content = response.choices[0].message.content if response.choices else None
        if not content:
            raise ProviderError("API trả về nội dung rỗng")
        return content

    def list_models(self) -> list[str]:
        import openai

        client = openai.OpenAI(api_key=self.keys.current() or "sk-no-key", base_url=self.profile.base_url or None, timeout=30)
        models = list(client.models.list())
        if protocol(self.profile) == "openrouter":
            catalog = {}
            for model in models:
                data = model.model_dump()
                reasoning = data.get("reasoning") or {}
                parameters = data.get("supported_parameters") or []
                catalog[model.id] = {
                    "supported": bool(reasoning) or "reasoning" in parameters or "reasoning_effort" in parameters,
                    "required": reasoning.get("mandatory", False),
                    "efforts": (
                        ["low", "medium", "high"] if "supported_efforts" in reasoning
                        and reasoning["supported_efforts"] is None else reasoning.get("supported_efforts", [])
                    ),
                }
            self.profile.options["reasoning_catalog"] = catalog
            self.profile.options["reasoning_catalog_url"] = self.profile.base_url
        return sorted(model.id for model in models)


class GeminiCompatTranslator(LLMTranslator):
    default_model = "gemini-2.5-flash"

    def _call(self, system: str, prompt: str, key: str) -> str:
        from google import genai
        from google.genai import types

        if not key:
            raise ProviderError("Thiếu API key Gemini")
        # google-genai 3.x mặc định dùng v1, chưa hỗ trợ system_instruction.
        http_options = types.HttpOptions(
            api_version="v1beta", base_url=self.profile.base_url or None,
            timeout=int(self.request_timeout * 1000),
        )
        client = genai.Client(api_key=key, http_options=http_options)
        response = client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system,
                temperature=_temperature(self.profile),
            ),
        )
        text = response.text
        if not text:
            raise ProviderError("Gemini trả về nội dung rỗng (có thể bị chặn an toàn)")
        return text

    def list_models(self) -> list[str]:
        from google import genai
        from google.genai import types

        http_options = types.HttpOptions(api_version="v1beta", base_url=self.profile.base_url or None)
        client = genai.Client(api_key=self.keys.current(), http_options=http_options)
        names = []
        for model in client.models.list():
            actions = getattr(model, "supported_actions", None) or []
            if actions and "generateContent" not in actions:
                continue
            names.append(str(model.name).removeprefix("models/"))
        return sorted(names)


class AnthropicTranslator(LLMTranslator):
    default_model = "claude-opus-5"
    _fallbacks_supported = True

    def _call(self, system: str, prompt: str, key: str) -> str:
        import anthropic

        if not key:
            raise ProviderError("Thiếu API key Anthropic")
        client = anthropic.Anthropic(
            api_key=key, base_url=self.profile.base_url or None, timeout=self.request_timeout, max_retries=0,
        )
        request = dict(
            model=self.model,
            max_tokens=16000,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        # Endpoint chính thức: bật fallback phía server khi model từ chối (refusal).
        use_fallbacks = (
            not self.profile.base_url
            and AnthropicTranslator._fallbacks_supported
            and str(self.profile.option("fallbacks", "on")) != "off"
        )
        if use_fallbacks:
            try:
                response = client.beta.messages.create(
                    betas=["server-side-fallback-2026-07-01"], fallbacks="default", **request
                )
            except anthropic.BadRequestError as exc:
                if "fallback" not in str(exc).lower():
                    raise
                AnthropicTranslator._fallbacks_supported = False
                response = client.messages.create(**request)
        else:
            response = client.messages.create(**request)
        if response.stop_reason == "refusal":
            raise ProviderError("Claude từ chối yêu cầu (refusal)")
        text = "".join(block.text for block in response.content if block.type == "text")
        if not text:
            raise ProviderError("Claude trả về nội dung rỗng")
        return text

    def list_models(self) -> list[str]:
        import anthropic

        client = anthropic.Anthropic(api_key=self.keys.current(), base_url=self.profile.base_url or None)
        return [model.id for model in client.models.list()]
