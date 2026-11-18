from __future__ import annotations

from PyQt6 import QtWidgets

from ...core.pipeline import RunContext
from ...core.pipeline.runner import STEP_LABELS, ordered_steps
from ...core.pipeline.tts import needs_tts, resolve_voice, segment_key
from ...core.providers import ProviderError
from ...core.providers.tts import create_tts
from ...core.resolve import resolve
from ..jobs import run_background


def summarize_run(ctx: RunContext, ids: list[str], steps: list[str], only_missing: bool) -> str:
    steps = ordered_steps(steps)
    if "context" in steps and not ctx.project.context_auto_update:
        steps.remove("context")
    asr = subtitles = translate = tts = unknown_tts = 0
    context = ctx.store.load_context() if "tts" in steps else None
    for video_id in ids:
        ctx.check_stop()
        segments = ctx.store.load_segments(video_id)
        subtitles += bool(segments)
        recognize = not segments and "asr" in steps
        asr += recognize
        will_translate = "translate" in steps and (recognize or any(
            s.source.strip() and not s.locked and (not only_missing or not s.text.strip()) for s in segments
        ))
        translate += will_translate
        if "tts" not in steps:
            continue
        if will_translate:
            tts += 1
            continue
        voiced = [s for s in segments if s.text.strip()]
        if not voiced:
            continue
        if not only_missing:
            tts += 1
            continue
        doc = ctx.store.load_video(video_id)
        resolved = resolve(ctx.settings, "tts", ctx.project, doc)
        if resolved.profile is None:
            unknown_tts += 1
            continue
        try:
            provider = create_tts(resolved.profile)
        except ProviderError:
            unknown_tts += 1
            continue
        for seg in voiced:
            ctx.check_stop()
            voice = resolve_voice(seg, context, provider, resolved.value)
            key = segment_key(seg, resolved.profile, voice, doc.audio.speed)
            if needs_tts(ctx, doc, seg, key):
                tts += 1
                break
    counts = [f"{len(ids)} video"]
    if "asr" in steps:
        counts.append(f"{asr} cần nhận dạng")
    counts.append(f"{subtitles} đã có phụ đề")
    if "translate" in steps:
        counts.append(f"{translate} cần dịch")
    if "tts" in steps:
        counts.append(f"{tts} cần lồng tiếng")
        if unknown_tts:
            counts.append(f"{unknown_tts} chưa xác định được cấu hình TTS")
    return " · ".join(counts) + "\n\nCác bước đã chọn\n" + " → ".join(STEP_LABELS[s] for s in steps)


def confirm_run(parent, ctx: RunContext, ids: list[str], steps: list[str], only_missing: bool) -> bool:
    dialog = QtWidgets.QDialog(parent)
    dialog.setWindowTitle("Xem trước hàng đợi")
    layout = QtWidgets.QVBoxLayout(dialog)
    label = QtWidgets.QLabel("Đang kiểm tra phụ đề và lồng tiếng…")
    label.setWordWrap(True)
    layout.addWidget(label)
    buttons = QtWidgets.QDialogButtonBox()
    submit = buttons.addButton("Đưa vào hàng đợi", buttons.ButtonRole.AcceptRole)
    buttons.addButton("Hủy", buttons.ButtonRole.RejectRole)
    buttons.accepted.connect(dialog.accept)
    buttons.rejected.connect(dialog.reject)
    layout.addWidget(buttons)
    submit.setEnabled(False)
    dialog.resize(470, 160)

    def ready(summary):
        if not ctx.stop_event.is_set():
            label.setText("Dự kiến\n" + summary)
            submit.setEnabled(True)

    def failed(message):
        if not ctx.stop_event.is_set():
            label.setText(f"Không đọc được dữ liệu video: {message}")

    run_background(lambda *_: summarize_run(ctx, ids, steps, only_missing), ready, failed)
    try:
        return dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted
    finally:
        ctx.stop_event.set()
