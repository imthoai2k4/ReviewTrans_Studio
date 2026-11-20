from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from reviewtrans.core.config import ProviderProfile
from reviewtrans.core.providers import ProviderError
from reviewtrans.core.providers.translate.llm import OpenAICompatTranslator
from reviewtrans.core.providers.translate.thinking import request_options, supported_modes


def router_profile():
    profile = ProviderProfile(kind="openai", base_url="https://openrouter.ai/api/v1", model="test/model")
    profile.options.update(reasoning_catalog_url=profile.base_url, reasoning_catalog={
        profile.model: {"supported": True, "required": False, "efforts": ["low", "medium", "high"]},
    })
    return profile


@pytest.mark.parametrize("mode", ["", "disabled", "enabled", "low", "medium", "high"])
def test_router_request_reaches_sdk(monkeypatch, mode):
    import openai

    profile = router_profile()
    profile.options.update(subtitle_thinking="disabled", context_thinking=mode)
    client = Mock()
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="result"))],
    )
    monkeypatch.setattr(openai, "OpenAI", Mock(return_value=client))
    translator = OpenAICompatTranslator(profile)
    translator.purpose = "context"
    assert translator._call("system", "prompt", "test-key") == "result"
    request = client.chat.completions.create.call_args.kwargs
    if not mode:
        assert "extra_body" not in request
    else:
        expected = {"enabled": mode == "enabled"} if mode in ("disabled", "enabled") else {"effort": mode}
        assert request["extra_body"] == {"reasoning": expected}


def test_deepseek_and_unknown_endpoint_do_not_receive_router_parameters():
    profile = ProviderProfile(base_url="https://api.deepseek.com", model="deepseek-v4-pro")
    assert request_options(profile, profile.model, "low") == {
        "extra_body": {"thinking": {"type": "enabled"}}, "reasoning_effort": "low",
    }
    with pytest.raises(ProviderError):
        request_options(profile, profile.model, "medium")
    profile.base_url = "https://api.deepseek.com.example.invalid/v1"
    with pytest.raises(ProviderError):
        request_options(profile, profile.model, "high")
    assert request_options(profile, profile.model, "disabled") == {"extra_body": {"thinking": {"type": "disabled"}}}


def test_router_unknown_nonreasoning_and_required_models():
    profile = router_profile()
    assert supported_modes(profile, "unknown") == {""}
    profile.options["reasoning_catalog"][profile.model]["required"] = True
    assert "disabled" not in supported_modes(profile, profile.model)
    with pytest.raises(ProviderError):
        request_options(profile, profile.model, "disabled")
    profile.options["reasoning_catalog"][profile.model]["supported"] = False
    assert supported_modes(profile, profile.model) == {""}
    profile.base_url += "/other"
    assert supported_modes(profile, profile.model) == {""}


def test_router_model_metadata_handles_mandatory_null_and_missing_efforts(monkeypatch):
    import openai

    profile = router_profile()
    client = Mock()
    client.models.list.return_value = [
        SimpleNamespace(id=name, model_dump=lambda data=data: data)
        for name, data in [
            ("required", {"reasoning": {"mandatory": True, "supported_efforts": ["high"]}}),
            ("all", {"reasoning": {"supported_efforts": None}}),
            ("toggle", {"supported_parameters": ["reasoning"]}),
            ("plain", {}),
        ]
    ]
    monkeypatch.setattr(openai, "OpenAI", Mock(return_value=client))
    assert OpenAICompatTranslator(profile).list_models() == ["all", "plain", "required", "toggle"]
    assert supported_modes(profile, "required") == {"", "enabled", "high"}
    assert {"low", "medium", "high"} <= supported_modes(profile, "all")
    assert supported_modes(profile, "toggle") == {"", "disabled", "enabled"}
    assert supported_modes(profile, "plain") == {""}


def test_editor_recommendations_and_model_switch_keep_saved_choices(qapp):
    from reviewtrans.core.config import TRANSLATE_KINDS
    from reviewtrans.ui.pages.providers import ProfileEditor

    profile = router_profile()
    editor = ProfileEditor(TRANSLATE_KINDS)
    editor.set_profile(profile)
    editor._recommend_thinking()
    assert profile.options["subtitle_thinking"] == "disabled"
    assert profile.options["context_thinking"] == "medium"
    editor.model.setEditText("unknown")
    combo = editor.thinking_combos["context_thinking"]
    assert combo.value() == "medium"
    assert combo.model().item(combo.findData("medium")).isEnabled()
    assert combo.model().item(combo.findData("")).isEnabled()
    editor.model.setEditText("test/model")
    assert combo.model().item(combo.findData("medium")).isEnabled()
    editor.close()


def test_editor_fetch_transfers_capabilities_and_persists_levels(qapp, monkeypatch):
    from reviewtrans.core.config import TRANSLATE_KINDS
    from reviewtrans.core.models import from_dict, to_dict
    from reviewtrans.ui.pages import providers

    profile = router_profile()
    profile.options.clear()
    editor = providers.ProfileEditor(TRANSLATE_KINDS)
    editor.set_profile(profile)

    def translator(snapshot):
        def fetch():
            snapshot.options.update(router_profile().options)
            return [snapshot.model]
        return SimpleNamespace(list_models=fetch)

    monkeypatch.setattr(providers, "create_translator", translator)
    monkeypatch.setattr(providers, "run_background", lambda work, done, fail: done(work(None, None)))
    monkeypatch.setattr(editor.model, "showPopup", lambda: None)
    editor._fetch_models()
    editor._recommend_thinking()
    restored = from_dict(ProviderProfile, to_dict(profile))
    assert restored.options["context_thinking"] == "medium"
    assert "medium" in supported_modes(restored, restored.model)
    assert restored.options["subtitle_thinking"] == "disabled"
    editor.close()


@pytest.mark.parametrize("catalog_state", ["not_loaded", "missing_metadata", "stale"])
def test_router_choices_are_selectable_and_sent_without_capability_metadata(qapp, monkeypatch, catalog_state):
    import openai
    from PyQt6 import QtCore, QtTest
    from reviewtrans.core.config import TRANSLATE_KINDS
    from reviewtrans.ui.pages.providers import ProfileEditor

    profile = router_profile()
    profile.model = "baidu/ernie-5.5-flash-preview:free"
    if catalog_state == "not_loaded":
        profile.options.clear()
    else:
        profile.options["reasoning_catalog"] = {
            profile.model: {"supported": False, "required": False, "efforts": []},
        }
        if catalog_state == "stale":
            profile.options["reasoning_catalog_url"] = "https://old.example/v1"
    client = Mock()
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
    )
    monkeypatch.setattr(openai, "OpenAI", Mock(return_value=client))
    editor = ProfileEditor(TRANSLATE_KINDS)
    editor.set_profile(profile)
    for purpose in ("subtitle", "context"):
        combo = editor.thinking_combos[f"{purpose}_thinking"]
        for mode in ("disabled", "enabled", "low", "medium", "high"):
            # Real key navigation skips disabled items, unlike setCurrentIndex.
            QtTest.QTest.keyClick(combo, QtCore.Qt.Key.Key_Down)
            assert combo.value() == mode
            assert profile.options[f"{purpose}_thinking"] == mode
            translator = OpenAICompatTranslator(profile)
            translator.purpose = purpose
            assert translator._call("system", "prompt", "test-key") == "ok"
            expected = {"enabled": mode == "enabled"} if mode in ("disabled", "enabled") else {"effort": mode}
            assert client.chat.completions.create.call_args.kwargs["extra_body"] == {"reasoning": expected}
    editor.close()
