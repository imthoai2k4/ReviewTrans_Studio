"""Standalone FunASR worker, also shipped as a data file in the frozen app.

Only this process imports PyTorch. Killing it releases models/GPU memory and
lets the desktop application's existing Stop button cancel inference/downloads.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import sys
from pathlib import Path

MODELS = ("paraformer-zh", "fsmn-vad", "ct-punc", "campplus")
EVENT_PREFIX = "REVIEWTRANS_FUNASR "


def event(percent: float, message: str) -> None:
    print(EVENT_PREFIX + json.dumps({"percent": percent, "message": message}, ensure_ascii=False), flush=True)


def download_models(root: Path) -> None:
    from huggingface_hub import snapshot_download

    for i, name in enumerate(MODELS):
        event(5 + i * 22, f"Tải funasr/{name} từ Hugging Face…")
        target = root / name
        snapshot_download(
            repo_id=f"funasr/{name}", revision="main", local_dir=str(target),
            allow_patterns=["*.pt", "*.bin", "*.yaml", "*.json", "*.mvn", "*.txt", "seg_dict", "jieba_usr_dict"],
        )
        weights = "campplus_cn_common.bin" if name == "campplus" else "model.pt"
        for filename in (weights, "config.yaml", "configuration.json"):
            if not (target / filename).is_file():
                raise RuntimeError(f"Model {name} thiếu {filename}.")
        # The ASR tokenizer requires the extensionless seg_dict dictionary.
        required = {"paraformer-zh": ("seg_dict", "tokens.json", "am.mvn"),
                    "ct-punc": ("tokens.json",), "fsmn-vad": ("am.mvn",), "campplus": ()}
        for filename in required[name]:
            if not (target / filename).is_file():
                raise RuntimeError(f"Model {name} thiếu {filename}.")
        (target / ".reviewtrans-complete").write_text("1\n", encoding="utf-8")
    event(100, "Đã tải Paraformer-zh + FSMN-VAD + CT-Punc + CAM++")


def voice_boundaries(result, samples, models, device, threads):
    """Compare short voice windows at pauses and punctuation boundaries.

    Sentence-level speaker labels cannot split a sentence already containing
    two speakers. Evaluate candidate word boundaries before assembling SRT.
    Scores are evidence, not character identities or guaranteed diarization.
    """
    import math
    import torch
    from funasr import AutoModel

    times, sentence_ends = [], set()
    for sentence in result.get("sentence_info", []):
        pairs = sentence.get("timestamp", [])
        if not pairs:
            return []
        for pair in pairs:
            if (not isinstance(pair, (list, tuple)) or len(pair) != 2
                    or not all(isinstance(t, (float, int)) and math.isfinite(t) for t in pair)
                    or pair[0] < 0 or pair[1] <= pair[0]):
                return []
            times.append(pair)
        sentence_ends.add(len(times) - 1)
    if len(times) < 4:
        return []
    model = AutoModel(
        model=str(models / "campplus"), hub="hf",
        init_param=str(models / "campplus/campplus_cn_common.bin"),
        device=device, ncpu=threads, disable_update=True,
        disable_pbar=True, trust_remote_code=False,
    )
    # Context must not extend through another pause into a different voice.
    block_start, starts = times[0][0], []
    for i, pair in enumerate(times):
        if i and pair[0] - times[i - 1][1] >= 100:
            block_start = pair[0]
        starts.append(block_start)
    block_end, ends = times[-1][1], [0] * len(times)
    for i in range(len(times) - 1, -1, -1):
        if i + 1 < len(times) and times[i + 1][0] - times[i][1] >= 100:
            block_end = times[i][1]
        ends[i] = block_end
    cache, boundaries = {}, []

    def embedding(start, end):
        key = (int(start * 16), int(end * 16))
        if key not in cache:
            audio = samples[key[0]:key[1]]
            cache[key] = model.generate(input=audio, fs=16000)[0]["spk_embedding"]
        return cache[key]

    for i in range(len(times) - 1):
        left_end, right_start = times[i][1], times[i + 1][0]
        gap = right_start - left_end
        if gap < 0 or (gap < 100 and i not in sentence_ends) or gap >= 800:
            continue
        left_start = max(starts[i], left_end - 1500)
        right_end = min(ends[i + 1], right_start + 1500)
        # Isolated syllables and very short windows are unreliable evidence.
        left_count = sum(a >= left_start and b <= left_end for a, b in times[max(0, i-30):i+1])
        right_count = sum(a >= right_start and b <= right_end for a, b in times[i+1:i+32])
        if min(left_end - left_start, right_end - right_start) < 400 or min(left_count, right_count) < 2:
            continue
        score = torch.nn.functional.cosine_similarity(
            embedding(left_start, left_end), embedding(right_start, right_end)).item()
        if math.isfinite(score):
            boundaries.append({"left_end": left_end, "right_start": right_start,
                               "similarity": round(score, 6)})
        if i % 20 == 0:
            event(85 + 14 * i / len(times), "Kiểm tra ranh giới giọng nói CAM++")
    return boundaries


def transcribe(args) -> dict:
    import soundfile as sf
    import torch
    from funasr import AutoModel

    device = "cuda:0" if args.device in ("auto", "cuda") and torch.cuda.is_available() else "cpu"
    if args.device == "cuda" and device == "cpu":
        event(5, "Không có PyTorch CUDA khả dụng; Paraformer chuyển sang CPU.")
    event(10, f"Nạp Paraformer-zh + FSMN-VAD + CT-Punc · {device}")
    torch.set_num_threads(args.threads)
    model = AutoModel(
        model=str(args.models / "paraformer-zh"), hub="hf",
        vad_model=str(args.models / "fsmn-vad"),
        vad_kwargs={"max_single_segment_time": 15000},
        punc_model=str(args.models / "ct-punc"),
        device=device, ncpu=args.threads, disable_update=True,
        disable_pbar=True, trust_remote_code=False,
    )
    # Pass samples directly: no dependence on torchaudio/torchcodec file decoders.
    samples, rate = sf.read(args.input, dtype="float32")
    if rate != 16000 or samples.ndim != 1:
        raise ValueError("Paraformer yêu cầu WAV mono 16 kHz.")
    if not len(samples):
        raise ValueError("File âm thanh rỗng.")
    event(25, f"Nhận dạng và căn timestamp theo lời nói · {device}")
    result = model.generate(
        input=samples, fs=16000, batch_size_s=15 if device != "cpu" else 1,
        sentence_timestamp=True, pred_timestamp=True, output_timestamp=True,
        return_raw_text=True,
    )
    if not isinstance(result, list) or len(result) != 1:
        raise RuntimeError("FunASR trả về kết quả không hợp lệ.")
    # Release ASR weights before loading the voice model on smaller GPUs.
    del model
    if device != "cpu":
        torch.cuda.empty_cache()
    event(85, "Căn ranh giới lời thoại bằng timestamp và CAM++")
    result[0]["voice_boundaries"] = voice_boundaries(
        result[0], samples, args.models, device, args.threads)
    return {"engine": "paraformer-zh", "device": device,
            "segmentation_version": 2,
            "funasr_version": importlib.metadata.version("funasr"), "result": result[0]}


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
        sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--models", type=Path)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    # Keep auxiliary caches inside the app's data directory too.
    if args.models:
        os.environ.setdefault("HF_HOME", str(args.models.parent / "huggingface"))
        os.environ.setdefault("MODELSCOPE_CACHE", str(args.models.parent / "modelscope"))
    if args.check:
        import torch
        import torchaudio
        from funasr import AutoModel

        assert AutoModel and torchaudio
        info = {"funasr_version": importlib.metadata.version("funasr"),
                "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
                "cuda_available": torch.cuda.is_available(),
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
                "python": str(sys.executable)}
        if args.output:
            args.output.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
        event(100, f"FunASR {info['funasr_version']} · PyTorch {torch.__version__} · CUDA={info['cuda_available']} · {info['gpu']}")
        return 0
    if not args.models:
        parser.error("--models is required")
    if args.download:
        download_models(args.models)
        return 0
    if not args.input or not args.output:
        parser.error("--input and --output are required")
    payload = transcribe(args)
    partial = args.output.with_suffix(".part.json")
    partial.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    partial.replace(args.output)
    event(100, f"Paraformer hoàn tất · {payload['device']}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"FunASR: {exc}", file=sys.stderr, flush=True)
        raise
