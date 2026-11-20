from __future__ import annotations

import traceback
import time
from dataclasses import dataclass, field

from ..models import STAGE_LABELS, STATE_DONE, STATE_ERROR, STATE_RUNNING, STATE_STALE, Segment, VideoDoc
from ..proc import StopRequested
from ..providers import ProviderError
from . import RunContext
from .asr import ensure_media_info, run_asr
from .mix import run_mix
from .render import run_render
from .translate import run_translate, update_context
from .tts import run_tts

PIPELINE_ORDER = ["asr", "translate", "context", "tts", "mix", "render"]
STEP_LABELS = {**STAGE_LABELS, "context": "Cập nhật ngữ cảnh"}


def ordered_steps(steps: list[str]) -> list[str]:
    requested = set(steps)
    if "render" in requested:
        requested.add("mix")
    return [step for step in PIPELINE_ORDER if step in requested]


@dataclass
class PipelineCheckpoint:
    """Giữ các bước và câu đã xong khi dừng / tiếp tục cùng một job."""

    completed_steps: set[str] = field(default_factory=set)
    translated_ids: set[int] = field(default_factory=set)
    tts_ids: set[int] = field(default_factory=set)


class VideoPipeline:
    """Chạy các bước cho một video, lưu kết quả sau mỗi bước."""

    def __init__(self, ctx: RunContext, video_id: str) -> None:
        self.ctx = ctx
        self.force_asr = False
        self.video_id = video_id
        self.doc: VideoDoc = ctx.store.load_video(video_id)
        self.segments: list[Segment] = ctx.store.load_segments(video_id)
        self.checkpoint = PipelineCheckpoint()

    # ------------------------------------------------------------ persistence

    def save(self) -> None:
        self.ctx.store.save_video(self.doc)
        self.ctx.store.save_segments(self.video_id, self.segments)
        self.ctx.video_changed(self.video_id)

    def save_segments(self) -> None:
        self.ctx.store.save_segments(self.video_id, self.segments)
        self.ctx.video_changed(self.video_id)

    # ------------------------------------------------------------ run

    def run(
        self, steps: list[str], only_missing: bool = True, force_context: bool = False, *, start_at: str = "",
    ) -> None:
        ctx = self.ctx
        ctx.check_stop()
        steps = ordered_steps(steps)
        if start_at:
            steps = steps[steps.index(start_at):]
        if "asr" in steps and self.segments and not self.force_asr:
            ctx.log(f"{self.doc.name}: giữ {len(self.segments)} câu hiện có, bỏ qua nhận dạng.")
            steps.remove("asr")
        if not steps:
            return
        ensure_media_info(self.doc)
        for step in steps:
            ctx.check_stop()
            if step in self.checkpoint.completed_steps:
                continue
            if step == "context" and not (ctx.project.context_auto_update or force_context):
                continue
            ctx.log(f"── {self.doc.name}: {STEP_LABELS.get(step, step)}")
            ctx.progress(0, STEP_LABELS.get(step, step))
            started = time.monotonic()
            stage = step if step != "context" else None
            if stage:
                self.doc.set_stage(stage, STATE_RUNNING)
                self.save()
            try:
                warning = self._run_step(step, only_missing)
            except StopRequested:
                if stage:
                    self.doc.set_stage(stage, STATE_STALE)
                    self.save()
                raise
            except Exception as exc:  # noqa: BLE001
                if ctx.stop_event.is_set():
                    if stage:
                        self.doc.set_stage(stage, STATE_STALE)
                        self.save()
                    raise StopRequested() from exc
                if (
                    step == "context" and isinstance(exc, ProviderError) and not force_context
                    and any(s in steps for s in ("tts", "mix", "render"))
                ):
                    ctx.log(
                        f"Cảnh báo: cập nhật ngữ cảnh lỗi sau {time.monotonic() - started:.1f}s: {exc}. "
                        "Giữ ngữ cảnh cũ và tiếp tục các bước còn lại."
                    )
                    ctx.progress(100, "Giữ ngữ cảnh cũ; tiếp tục xử lý")
                    self.checkpoint.completed_steps.add(step)
                    continue
                ctx.log(traceback.format_exc(limit=3))
                if stage:
                    self.doc.set_stage(stage, STATE_ERROR, str(exc))
                    self.save()
                raise
            if stage:
                self.doc.set_stage(stage, STATE_STALE if warning else STATE_DONE, warning or "")
            self.save()
            self.checkpoint.completed_steps.add(step)
            ctx.log(f"Hoàn tất {STEP_LABELS.get(step, step)} sau {time.monotonic() - started:.1f}s.")

    def _run_step(self, step: str, only_missing: bool) -> str | None:
        ctx = self.ctx
        if step == "asr":
            segments = run_asr(ctx, self.doc)
            if self.segments and not segments:
                raise ValueError("Nhận dạng không trả về câu nào. Đã giữ lại phụ đề hiện có.")
            self.segments = segments
            self.doc.mark_stale_from("translate")
        elif step == "translate":
            run_translate(
                ctx, self.doc, self.segments, only_missing=only_missing, save=self.save_segments,
                completed_ids=self.checkpoint.translated_ids,
            )
            self.doc.mark_stale_from("tts")
        elif step == "context":
            update_context(ctx, self.doc, self.segments)
        elif step == "tts":
            _done, failed = run_tts(
                ctx, self.doc, self.segments, force=not only_missing, save=self.save_segments,
                completed_ids=self.checkpoint.tts_ids,
            )
            self.doc.mark_stale_from("mix")
            if failed > 1:
                raise RuntimeError(f"{failed} câu tạo lồng tiếng thất bại (các câu khác đã lưu).")
            if failed:
                skipped = next(s for s in self.segments if s.tts_error and s.text.strip())
                warning = f"Bỏ qua lồng tiếng câu #{skipped.id}, giữ phụ đề và tiếp tục các bước đã chọn."
                ctx.log(warning)
                return warning
        elif step == "mix":
            run_mix(ctx, self.doc, self.segments)
            self.doc.mark_stale_from("render")
        elif step == "render":
            mix = ctx.store.cache_dir(self.video_id) / "mix.wav"
            output = run_render(ctx, self.doc, self.segments, mix if mix.exists() else None)
            self.doc.last_output = str(output)
