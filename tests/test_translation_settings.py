from __future__ import annotations

import json
import re
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from reviewtrans.core.config import ProviderProfile, SettingsStore
from reviewtrans.core.context import UPDATE_SYSTEM
from reviewtrans.core.models import Segment
from reviewtrans.core.pipeline import RunContext
from reviewtrans.core.pipeline.runner import VideoPipeline
from reviewtrans.core.store import ProjectStore
from reviewtrans.core.providers.translate.base import parse_items


@pytest.mark.parametrize("wrapped", [False, True])
def test_single_line_json_batch_keeps_only_translation(wrapped):
    raw = '[{"id": 1, "text": "Translation"}]'
    if wrapped:
        raw = "```json\n" + raw + "\n```"
    assert parse_items(raw, 1) == ["Translation"]


def _mock_api(monkeypatch):
    import openai

    def reply(**request):
        system, prompt = [message["content"] for message in request["messages"]]
        if system == UPDATE_SYSTEM:
            content = json.dumps({"summary": "Updated"})
        else:
            items = json.loads(re.search(r"Items \(JSON\):\n(.*)\n\nReturn", prompt, re.S).group(1))
            content = json.dumps([{"id": item["id"], "text": "VI:" + item["text"]} for item in items])
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

    client = Mock()
    client.chat.completions.create.side_effect = reply
    monkeypatch.setattr(openai, "OpenAI", Mock(return_value=client))
    return client.chat.completions.create


def _pipeline(tmp_path, options):
    settings = SettingsStore(tmp_path / "settings.json").settings
    profile = ProviderProfile(kind="openai", name="DeepSeek", base_url="https://api.deepseek.com", model="deepseek-v4.1-flash-1017", options=options)
    settings.translate_profiles = [profile]
    store, project = ProjectStore.create(tmp_path / "projects", "Batch")
    project.translate_profile = profile.id
    doc = store.add_video(project, "ep.mp4")
    doc.duration, doc.width, doc.height = 250, 320, 240
    store.save_video(doc)
    store.save_segments(doc.id, [Segment(id=i, start=i, end=i + 0.8, source=f"line{i}") for i in range(201)])
    logs = []
    ctx = RunContext(store=store, project=project, settings=settings, log=logs.append)
    return VideoPipeline(ctx, doc.id), logs


@pytest.mark.parametrize("subtitle,context", [("disabled", "enabled"), ("enabled", "disabled"), ("", ""), ("low", "high")])
def test_200_line_batches_and_separate_thinking_reach_api(tmp_path, home, monkeypatch, subtitle, context):
    api = _mock_api(monkeypatch)
    pipeline, logs = _pipeline(tmp_path, {"subtitle_thinking": subtitle, "context_thinking": context})
    pipeline.run(["translate", "context"])
    assert api.call_count == 3  # 200 câu + 1 câu + cập nhật ngữ cảnh.
    requests = [call.kwargs for call in api.call_args_list]
    for request, mode in zip(requests, [subtitle, subtitle, context]):
        if mode:
            assert request["extra_body"]["thinking"]["type"] == ("enabled" if mode in ("low", "high") else mode)
            if mode in ("low", "high"):
                assert request["reasoning_effort"] == mode
        else:
            assert "extra_body" not in request
    counts = []
    for request in requests[:2]:
        prompt = request["messages"][1]["content"]
        counts.append(len(json.loads(re.search(r"Items \(JSON\):\n(.*)\n\nReturn", prompt, re.S).group(1))))
    assert counts == [200, 1]
    assert all(seg.text == "VI:" + seg.source for seg in pipeline.ctx.store.load_segments(pipeline.video_id))
    assert pipeline.ctx.store.load_context().summary == "Updated"
    assert "Tối đa 200 câu/lần dịch." in logs


def test_explicit_context_profile_uses_its_own_thinking_setting(tmp_path, home, monkeypatch):
    api = _mock_api(monkeypatch)
    pipeline, logs = _pipeline(tmp_path, {"subtitle_thinking": "disabled", "context_thinking": "disabled"})
    profile = ProviderProfile(
        kind="openai", name="Context", model="context-model", base_url="https://api.deepseek.com",
        options={"subtitle_thinking": "disabled", "context_thinking": "enabled"},
    )
    pipeline.ctx.settings.translate_profiles.append(profile)
    pipeline.ctx.project.context_profile = profile.id
    pipeline.run(["translate", "context"])
    assert api.call_args_list[0].kwargs["extra_body"]["thinking"]["type"] == "disabled"
    assert api.call_args_list[-1].kwargs["extra_body"]["thinking"]["type"] == "enabled"
    assert api.call_args_list[-1].kwargs["model"] == "context-model"
