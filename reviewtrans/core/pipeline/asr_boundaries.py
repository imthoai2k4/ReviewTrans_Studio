"""Rebuild subtitle boundaries from measured token times and voice evidence.

Punctuation is a text prediction, not a speaker boundary. Never redistribute
words by character-count timing or infer a speaker from the translated text.
"""
from __future__ import annotations

import math
import re


def aligned_tokens(sentence: dict) -> list[dict] | None:
    """Map timed Chinese characters / English words to exact surface slices.

    Old caches without aligned tokens remain usable, unchanged. Ambiguous
    alignment must not silently drop, duplicate or move any recognized text.
    """
    text = str(sentence.get("text") or sentence.get("sentence") or "").strip()
    times = sentence.get("timestamp")
    raw = sentence.get("raw_text", "")
    if not isinstance(times, list) or not times or not isinstance(raw, str):
        return None
    words = raw.split()
    if len(words) != len(times):
        # Newer FunASR may return surface-form raw_text instead of spaced tokens.
        words = re.findall(r"[\u3400-\u9fff]|[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)*", raw)
    if len(words) != len(times):
        return None
    cursor, spans = 0, []
    for word, pair in zip(words, times):
        pos = text.find(word, cursor)
        if pos < 0 or any(c.isalnum() for c in text[cursor:pos]):
            return None
        try:
            start, end = map(float, pair)
        except (TypeError, ValueError):
            return None
        if not all(map(math.isfinite, (start, end))) or start < 0 or end <= start:
            return None
        if spans and start < spans[-1]["start"]:
            return None
        spans.append({"start": start, "end": end, "offset": pos})
        cursor = pos + len(word)
    if any(c.isalnum() for c in text[cursor:]):
        return None
    if spans[0]["start"] != sentence.get("start") or spans[-1]["end"] != sentence.get("end"):
        return None
    offsets = [0, *(t["offset"] for t in spans[1:]), len(text)]
    for i, token in enumerate(spans):
        # Preserve intervening punctuation and whitespace exactly.
        token["text"] = text[offsets[i]:offsets[i + 1]]
        token.pop("offset", None)
    return spans


def rebuild_sentences(result: dict, *, log=None) -> list[dict]:
    sentences = result.get("sentence_info", [])
    evidence = {}
    for boundary in result.get("voice_boundaries", []):
        try:
            key = (float(boundary["left_end"]), float(boundary["right_start"]))
            score = float(boundary["similarity"])
        except (KeyError, TypeError, ValueError):
            continue
        if all(map(math.isfinite, (*key, score))) and -1 <= score <= 1:
            evidence[key] = score
    output, split_count, merge_count = [], 0, 0
    previous_aligned = False
    for sentence in sentences:
        tokens = aligned_tokens(sentence)
        if tokens is None:
            output.append(dict(sentence))
            previous_aligned = False
            continue
        chunks, begin = [], 0
        for i in range(1, len(tokens)):
            left, right = tokens[i - 1], tokens[i]
            gap = right["start"] - left["end"]
            score = evidence.get((left["end"], right["start"]))
            if gap >= 800 or (gap >= 100 and score is not None and score <= .32):
                chunks.append(tokens[begin:i])
                begin = i
                split_count += 1
        chunks.append(tokens[begin:])
        for i, chunk in enumerate(chunks):
            current = {"start": chunk[0]["start"], "end": chunk[-1]["end"],
                       "text": "".join(t["text"] for t in chunk).strip()}
            # Only undo a weak punctuation boundary with positive voice
            # evidence. Never merge across one of the detected internal turns.
            if i == 0 and previous_aligned and output:
                prev = output[-1]
                gap = current["start"] - prev["end"]
                score = evidence.get((prev["end"], current["start"]))
                if (0 <= gap <= 180 and score is not None and score >= .45
                        and prev["text"].endswith((",", "，", "、"))
                        and current["end"] - prev["start"] <= 7000
                        and len(prev["text"]) + len(current["text"]) <= 80):
                    separator = " " if prev["text"].isascii() and current["text"].isascii() else ""
                    prev["text"] += separator + current["text"]
                    prev["end"] = current["end"]
                    merge_count += 1
                    continue
            output.append(current)
        previous_aligned = True
    if log and (split_count or merge_count):
        log(f"Căn ranh giới lời thoại: tách {split_count} chỗ nghỉ/đổi giọng, "
            f"nối {merge_count} mảnh câu có giọng tương đồng; giữ timestamp từng từ.")
    return output
