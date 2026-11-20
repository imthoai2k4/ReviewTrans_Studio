from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QPA_FONTDIR", os.path.join(os.environ.get("WINDIR", "C:/Windows"), "Fonts"))

from PyQt6 import QtWidgets

from reviewtrans.core.config import (
    SETTINGS_VERSION, TRANSLATE_KINDS, TTS_KINDS, ProviderProfile, SettingsStore, default_settings,
)
from reviewtrans.core.models import to_dict
from reviewtrans.core.providers import ProviderError
from reviewtrans.core.providers.translate import create_translator
from reviewtrans.core.providers.tts import create_tts
from reviewtrans.ui.pages.providers import ProvidersPage
from reviewtrans.ui.state import AppState


class ApiSettingsTests(unittest.TestCase):
    def test_translation_batch_defaults_and_existing_settings_upgrade_to_200(self):
        self.assertEqual(default_settings().translate_batch_size, 200)
        for old, expected in ((60, 200), (80, 80), (25, 25), (300, 200)):
            with self.subTest(old=old), tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "settings.json"
                data = to_dict(default_settings())
                data.update(settings_version=3, translate_batch_size=old)
                path.write_text(json.dumps(data), encoding="utf-8")
                store = SettingsStore(path)
                self.assertEqual(store.settings.translate_batch_size, expected)
                saved = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(saved["translate_batch_size"], expected)
                self.assertEqual(saved["settings_version"], SETTINGS_VERSION)

    def test_defaults_and_factories_only_support_api_providers(self):
        settings = default_settings()
        self.assertEqual(set(TRANSLATE_KINDS), {"openai", "gemini", "anthropic"})
        self.assertEqual(set(TTS_KINDS), {"openai_speech", "vbee", "legacy_custom"})
        self.assertEqual(settings.tts_concurrency, 0)
        self.assertEqual(settings.translate_profiles[0].kind, "openai")
        self.assertEqual(settings.tts_profiles[0].kind, "openai_speech")
        self.assertNotIn("concurrency", settings.tts_profiles[0].options)
        for kind in ("google", "microsoft"):
            with self.subTest(kind=kind), self.assertRaises(ProviderError):
                create_translator(ProviderProfile(kind=kind))
        with self.assertRaises(ProviderError):
            create_tts(ProviderProfile(kind="edge"))

    def test_upgrade_removes_retired_providers_and_preserves_api_configuration(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "settings.json"
            settings = default_settings()
            saved_api = settings.tts_profiles[0]
            saved_api.name, saved_api.model = "OmniVoice", "omnivoice"
            saved_api.options = {"voice": "voice-1", "concurrency": "1"}
            settings.translate_profiles = [ProviderProfile(kind="google"), ProviderProfile(kind="microsoft")]
            edge = ProviderProfile(kind="edge")
            settings.tts_profiles.append(edge)
            settings.default_translate_profile = settings.translate_profiles[0].id
            settings.default_tts_profile = edge.id
            settings.tts_concurrency = 3
            settings.settings_version = 2
            path.write_text(json.dumps(to_dict(settings)), encoding="utf-8")
            store = SettingsStore(path)
            self.assertEqual([p.kind for p in store.settings.translate_profiles], ["openai"])
            self.assertEqual(store.settings.tts_profiles, [saved_api])
            self.assertEqual(store.settings.default_tts_profile, saved_api.id)
            self.assertEqual(store.settings.tts_concurrency, 0)
            saved = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(saved["settings_version"], SETTINGS_VERSION)
            self.assertEqual(saved["tts_profiles"][0]["options"], saved_api.options)
            self.assertEqual(SettingsStore(path).settings.default_tts_profile, saved_api.id)


class ProvidersPageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.state = AppState(SettingsStore(Path(self.temp.name) / "settings.json"))
        self.page = ProvidersPage(self.state)
        self.panel = self.page.panel
        self.editor = self.panel.editor
        self.addCleanup(self.close_page)

    def close_page(self):
        self.panel.save_timer.stop()
        self.page.close()
        self.page.deleteLater()
        self.app.processEvents()

    def test_both_categories_share_one_page_and_selection_changes_api_fields(self):
        self.assertFalse(self.page.findChildren(QtWidgets.QTabWidget))
        self.assertEqual(self.panel.list.count(), 2)
        self.assertEqual(self.panel.kind, "translate")
        self.assertEqual(set(self.editor.kinds), set(TRANSLATE_KINDS))
        self.assertNotIn("voice", self.editor.option_edits)
        self.panel.list.setCurrentRow(1)
        self.assertEqual(self.panel.kind, "tts")
        self.assertEqual(set(self.editor.kinds), set(TTS_KINDS))
        concurrency = self.editor.option_edits["concurrency"]
        self.assertEqual(concurrency.text(), "")
        self.assertEqual(concurrency.placeholderText(), "0")
        self.assertEqual(concurrency.validator().bottom(), 0)
        self.assertIn("voice", self.editor.option_edits)

    def test_subtitle_and_context_thinking_save_independently(self):
        self.editor.thinking_combos["subtitle_thinking"].set_value("disabled")
        self.editor.thinking_combos["context_thinking"].set_value("enabled")
        self.editor.option_edits["temperature"].setText("0.2")
        self.editor._commit()
        self.panel._save()
        saved = SettingsStore(self.state.settings_store.path).settings.translate_profiles[0]
        self.assertEqual(saved.options["subtitle_thinking"], "disabled")
        self.assertEqual(saved.options["context_thinking"], "enabled")
        self.assertEqual(saved.options["temperature"], "0.2")
        self.panel.list.setCurrentRow(1)
        self.assertFalse(self.editor.thinking_combos)
        self.panel.list.setCurrentRow(0)
        self.assertEqual(self.editor.thinking_combos["subtitle_thinking"].value(), "disabled")
        self.assertEqual(self.editor.thinking_combos["context_thinking"].value(), "enabled")
        self.editor.thinking_combos["subtitle_thinking"].set_value("")
        self.assertNotIn("subtitle_thinking", self.editor.profile.options)
        self.assertEqual(self.editor.profile.options["context_thinking"], "enabled")

    def test_thinking_choices_follow_selected_api_profile(self):
        first = self.editor.profile
        self.editor.thinking_combos["subtitle_thinking"].set_value("disabled")
        self.panel.add("translate")
        second = self.editor.profile
        self.assertEqual(self.editor.thinking_combos["subtitle_thinking"].value(), "")
        self.editor.thinking_combos["subtitle_thinking"].set_value("enabled")
        self.panel.refresh(first.id)
        self.assertEqual(self.editor.thinking_combos["subtitle_thinking"].value(), "disabled")
        self.panel.refresh(second.id)
        self.assertEqual(self.editor.thinking_combos["subtitle_thinking"].value(), "enabled")
        self.editor.kind.set_value("gemini")
        self.assertFalse(self.editor.thinking_combos)
        self.assertNotIn("subtitle_thinking", self.editor.profile.options)

    def test_add_edit_duplicate_and_defaults_do_not_mix_translation_with_tts(self):
        self.panel.add("tts")
        speech = self.editor.profile
        self.editor.name.setText("OmniVoice")
        self.editor.model.setEditText("omnivoice")
        self.editor.base_url.setText("https://example.invalid/v1")
        self.editor.option_edits["voice"].setText("voice-1")
        self.editor.option_edits["concurrency"].setText("0")
        self.editor._commit()
        combo = self.panel.active_profiles["tts"]
        combo.set_value(speech.id)
        combo.activated.emit(combo.currentIndex())
        self.assertEqual(self.state.settings.default_tts_profile, speech.id)
        self.panel.add("translate")
        translation = self.editor.profile
        self.editor.model.setEditText("translation-model")
        self.editor._commit()
        self.panel.duplicate()
        self.assertEqual(self.panel.kind, "translate")
        self.assertNotEqual(self.editor.profile.id, translation.id)
        self.assertEqual(self.editor.profile.model, "translation-model")
        self.panel._save()
        saved = SettingsStore(self.state.settings_store.path).settings
        self.assertEqual(len(saved.translate_profiles), 3)
        self.assertEqual(len(saved.tts_profiles), 2)
        persisted_speech = next(p for p in saved.tts_profiles if p.id == speech.id)
        self.assertEqual(persisted_speech.model, "omnivoice")
        self.assertEqual(persisted_speech.options, {"voice": "voice-1", "concurrency": "0"})

    def test_deleting_default_selects_another_api_in_the_same_category(self):
        self.panel.add("tts")
        combo = self.panel.active_profiles["tts"]
        combo.set_value(self.editor.profile.id)
        combo.activated.emit(combo.currentIndex())
        with patch("reviewtrans.ui.pages.providers.confirm", return_value=True):
            self.panel.delete()
        self.assertEqual(self.state.settings.default_tts_profile, self.state.settings.tts_profiles[0].id)
        self.assertEqual(combo.value(), self.state.settings.default_tts_profile)
        self.assertEqual(len(self.state.settings.translate_profiles), 1)

    def test_switch_api_saves_selection_without_switching_other_category(self):
        first = self.state.settings.default_translate_profile
        speech = self.state.settings.default_tts_profile
        self.panel.add("translate")
        second = self.editor.profile.id
        combo = self.panel.active_profiles["translate"]
        self.assertEqual(combo.value(), first)
        self.assertEqual(self.state.settings.default_translate_profile, first)
        combo.set_value(second)
        combo.activated.emit(combo.currentIndex())
        saved = SettingsStore(self.state.settings_store.path).settings
        self.assertEqual(saved.default_translate_profile, second)
        self.assertEqual(saved.default_tts_profile, speech)
        self.assertEqual(self.editor.profile.id, second)
        self.assertIn("Đang dùng", self.panel.list.currentItem().text())
        self.editor.name.setText("API mới đổi tên")
        self.editor._commit()
        self.assertEqual(combo.currentText(), "API mới đổi tên")
        self.panel.refresh(first)
        self.assertEqual(combo.value(), second)

    def test_empty_list_can_add_either_category(self):
        self.state.settings.translate_profiles.clear()
        self.state.settings.tts_profiles.clear()
        self.state.settings.default_translate_profile = ""
        self.state.settings.default_tts_profile = ""
        self.panel.refresh()
        self.assertIsNone(self.editor.profile)
        self.assertFalse(self.panel.test_button.isEnabled())
        self.assertFalse(self.panel.active_profiles["translate"].isEnabled())
        self.assertFalse(self.panel.active_profiles["tts"].isEnabled())
        self.panel.add("tts")
        self.assertEqual(self.panel.kind, "tts")
        self.assertEqual(self.state.settings.default_tts_profile, self.editor.profile.id)

    def test_kind_change_drops_options_from_previous_protocol(self):
        self.panel.list.setCurrentRow(1)
        self.editor.option_edits["voice"].setText("old-voice")
        self.editor._commit()
        self.editor.kind.set_value("vbee")
        self.assertEqual(self.editor.profile.kind, "vbee")
        self.assertEqual(self.editor.profile.options, {})
        self.assertIn("app_id", self.editor.option_edits)
        self.assertNotIn("instructions", self.editor.option_edits)

    def test_test_connection_routes_to_selected_api_and_ignores_stale_result(self):
        pending = []
        provider = Mock()
        provider.test.return_value = "translated"
        with (
            patch("reviewtrans.ui.pages.providers.run_background", side_effect=lambda *args: pending.append(args)),
            patch("reviewtrans.ui.pages.providers.create_translator", return_value=provider) as translation,
        ):
            self.panel.test()
            self.assertEqual(pending[0][0](None, None), "translated")
            translation.assert_called_once()
            self.panel.list.setCurrentRow(1)
            pending[0][1]("translated")
            self.assertEqual(self.panel.test_result.text(), "")
            self.assertTrue(self.panel.test_button.isEnabled())

    def test_tts_connection_test_uses_selected_speech_api(self):
        self.panel.list.setCurrentRow(1)
        pending = []
        provider = Mock()
        provider.test.return_value = Path(self.temp.name) / "test.wav"
        with (
            patch("reviewtrans.ui.pages.providers.run_background", side_effect=lambda *args: pending.append(args)),
            patch("reviewtrans.ui.pages.providers.create_tts", return_value=provider) as speech,
            patch("reviewtrans.ui.pages.providers.temp_dir", return_value=Path(self.temp.name)),
        ):
            self.panel.test()
            self.assertEqual(pending[0][0](None, None), provider.test.return_value)
            speech.assert_called_once()
            self.assertEqual(speech.call_args.args[0].id, self.state.settings.tts_profiles[0].id)
            provider.test.assert_called_once()


if __name__ == "__main__":
    unittest.main()
