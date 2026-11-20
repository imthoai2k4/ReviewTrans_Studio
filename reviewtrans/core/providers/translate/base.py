from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field, replace
from typing import Callable

from ...context import extract_json
from ...langs import english_name
from ...proc import StopRequested, call_interruptibly
from .. import KeyRing, ProviderError
from ...config import MAX_TRANSLATE_BATCH_SIZE, ProviderProfile


@dataclass
class TranslateJob:
    source_lang: str = "auto"
    target_lang: str = "vi"
    context_text: str = ""
    glossary: list[tuple[str, str]] = field(default_factory=list)
    previous: list[tuple[str, str]] = field(default_factory=list)  # (gốc, đã dịch) ngay trước batch
    durations: list[float] = field(default_factory=list)  # thời lượng lời nói đã căn, cùng thứ tự lines
    log: Callable[[str], None] | None = None
    stop_event: threading.Event | None = None

    def check_stop(self) -> None:
        if self.stop_event is not None and self.stop_event.is_set():
            raise StopRequested()

    def emit(self, message: str) -> None:
        if self.log:
            self.log(message)


class Translator:
    uses_context = False
    default_model = ""
    max_batch = 100
    request_timeout: float = 300

    def __init__(self, profile: ProviderProfile, model_override: str = "") -> None:
        self.profile = profile
        self.model = model_override or profile.model or self.default_model
        self.keys = KeyRing(profile.api_keys)
        self.purpose = "subtitle"

    def thinking_mode(self) -> str:
        mode = self.profile.option(f"{self.purpose}_thinking", "")
        return mode if mode in ("enabled", "disabled", "low", "medium", "high") else ""

    def translate(self, lines: list[str], job: TranslateJob) -> list[str]:
        raise NotImplementedError

    def complete(
        self, system: str, prompt: str, job: TranslateJob | None = None, attempts: int | None = None,
    ) -> str:
        raise ProviderError(f"{self.profile.name} không hỗ trợ sinh văn bản tự do (cần LLM).")

    def list_models(self) -> list[str]:
        """Danh sách model mà API cho phép (máy dịch không có model → rỗng)."""
        return []

    def test(self) -> str:
        result = self.translate(["你好，世界"], TranslateJob(source_lang="zh", target_lang="vi"))
        return result[0]


RULES = (
    "You are a professional subtitle translator for movie-recap / review videos that will be dubbed.\n"
    "CONSTRAINTS:\n"
    "- The translation is read aloud by TTS, so it must be concise and match the spoken duration of "
    "the original line as closely as possible. Prefer short, natural wording; never pad or explain.\n"
    "- If the source line is short, the target must be short. Do NOT expand it into a longer "
    "sentence unless absolutely required by context.\n"
    "- Each duration_seconds is the available spoken time, not a request to add words. "
    "Choose concise phrasing that can be spoken in that time without losing the meaning.\n"
    "- Translate every item independently and keep the same number of items with the same ids. "
    "Never merge, split, skip, reorder or add items.\n"
    "- Keep names, titles and forms of address consistent with the context below.\n"
    "- Output natural spoken {target} with correct pronouns for the relationship between speakers.\n"
    "- Do not add notes, romanisation or quotes that are not in the source."
)


class LLMTranslator(Translator):
    uses_context = True
    max_batch = MAX_TRANSLATE_BATCH_SIZE

    def _call(self, system: str, prompt: str, key: str) -> str:
        raise NotImplementedError

    def complete(
        self, system: str, prompt: str, job: TranslateJob | None = None, attempts: int | None = None,
    ) -> str:
        return self._with_retry(
            lambda key: self._call(system, prompt, key),
            attempts=max(3, len(self.keys)) if attempts is None else max(1, attempts), job=job,
        )

    def _with_retry(self, func, attempts: int = 3, job: TranslateJob | None = None):
        last_error: Exception | None = None
        for attempt in range(attempts):
            if job:
                job.check_stop()
            key = self.keys.next()
            try:
                result = call_interruptibly(lambda: func(key), job.stop_event if job else None)
                if job:
                    job.check_stop()
                return result
            except StopRequested:
                raise
            except Exception as exc:  # noqa: BLE001 - SDK khác nhau ném lỗi khác nhau
                if job:
                    job.check_stop()
                last_error = exc
                if "free-models-per-day" in str(exc).lower():
                    raise ProviderError(
                        f"{self.profile.name}: Đã hết hạn mức dịch model miễn phí trong ngày "
                        "(OpenRouter, HTTP 429). Chờ hạn mức được đặt lại hoặc chọn provider/model "
                        "có hạn mức khả dụng trong API và cấu hình video. "
                        "Chạy lại ngay không giải quyết được lỗi này; các câu đã dịch vẫn được lưu."
                    ) from exc
                retry = attempt + 1 < attempts and not (_is_fatal(exc) and len(self.keys) <= 1)
                delay = min(20, 2 ** attempt)
                if job:
                    action = f"thử lại sau {delay}s (lần {attempt + 2}/{attempts})" if retry else "dừng thử"
                    job.emit(f"  lỗi {type(exc).__name__}: {str(exc)[:300]} — {action}")
                if not retry:
                    break
                if job and job.stop_event is not None:
                    if job.stop_event.wait(delay):
                        job.check_stop()
                else:
                    time.sleep(delay)
        raise ProviderError(f"{self.profile.name}: {last_error}") from last_error

    def build_prompts(self, lines: list[str], job: TranslateJob) -> tuple[str, str]:
        target = english_name(job.target_lang)
        system = RULES.format(target=target)
        if job.context_text:
            system += "\n\nPROJECT CONTEXT:\n" + job.context_text
        items = [{"id": i + 1, "text": text} for i, text in enumerate(lines)]
        for item, seconds in zip(items, job.durations):
            if seconds > 0:
                item["duration_seconds"] = round(seconds, 3)
        prompt = f"Translate from {english_name(job.source_lang)} to {target}.\n"
        if job.previous:
            prompt += "Previous lines for continuity (already translated, do NOT output them):\n"
            prompt += "\n".join(f"{src} => {dst}" for src, dst in job.previous) + "\n\n"
        prompt += (
            "Items (JSON):\n"
            + json.dumps(items, ensure_ascii=False)
            + '\n\nReturn ONLY a JSON array like [{"id": 1, "text": "..."}] with exactly '
            f"{len(items)} items and the same ids."
        )
        return system, prompt

    def translate(self, lines: list[str], job: TranslateJob) -> list[str]:
        if not lines:
            return []
        system, prompt = self.build_prompts(lines, job)
        for attempt in range(2):
            raw = self._with_retry(lambda key: self._call(system, prompt, key), attempts=max(3, len(self.keys)), job=job)
            result = parse_items(raw, len(lines))
            if result is not None:
                return result
            job.emit(f"  kết quả không khớp số dòng ({len(lines)}), thử lại…")
        if len(lines) <= 3:
            out = []
            for index, line in enumerate(lines):
                single_job = replace(job, durations=job.durations[index:index + 1])
                single = self._with_retry(
                    lambda key, l=line: self._call(*self.build_prompts([l], single_job), key), job=job
                )
                parsed = parse_items(single, 1)
                out.append(parsed[0] if parsed else single.strip())
            return out
        middle = len(lines) // 2
        job.emit(f"  chia nhỏ batch {len(lines)} → {middle} + {len(lines) - middle}")
        return self.translate(lines[:middle], replace(job, durations=job.durations[:middle])) + self.translate(
            lines[middle:], replace(job, durations=job.durations[middle:]),
        )


def parse_items(raw: str, expected: int) -> list[str] | None:
    try:
        data = extract_json(raw)
    except (ValueError, TypeError):
        data = None
    if isinstance(data, dict):
        for key in ("items", "translations", "result", "data"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
        else:
            if all(str(k).isdigit() for k in data.keys()):
                data = [{"id": int(k), "text": v} for k, v in data.items()]
    if isinstance(data, list):
        mapping: dict[int, str] = {}
        for position, item in enumerate(data, start=1):
            if isinstance(item, dict):
                ident = item.get("id", position)
                text = item.get("text", item.get("t", item.get("translation", "")))
            else:
                ident, text = position, item
            try:
                mapping[int(ident)] = str(text).strip()
            except (TypeError, ValueError):
                continue
        if all(i in mapping for i in range(1, expected + 1)):
            return [mapping[i] for i in range(1, expected + 1)]
        return None
    lines = [line.strip() for line in raw.strip().splitlines() if line.strip()]
    if len(lines) == expected:
        return lines
    return None


def _is_fatal(exc: Exception) -> bool:
    text = str(exc).lower()
    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if status in (400, 401, 403, 404):
        return "rate" not in text and "quota" not in text
    return False
