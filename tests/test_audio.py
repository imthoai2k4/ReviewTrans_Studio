from __future__ import annotations

import math
import os
import tempfile
import unittest
import wave
from array import array
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QPA_FONTDIR", os.path.join(os.environ.get("WINDIR", "C:/Windows"), "Fonts"))

from PyQt6 import QtWidgets

from reviewtrans.core.config import AppSettings
from reviewtrans.core.models import AudioSettings, Segment, VideoDoc, from_dict, to_dict
from reviewtrans.core.paths import find_tool
from reviewtrans.core.pipeline import RunContext
from reviewtrans.core.pipeline.mix import run_mix
from reviewtrans.core.store import ProjectStore
from reviewtrans.ui.widgets.audio_panel import AudioPanel


class AudioPanelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.panel = AudioPanel()
        self.doc = VideoDoc()
        self.events = []
        self.panel.audioChanged.connect(self.events.append)
        self.panel.set_document(self.doc)
        self.addCleanup(self.panel.deleteLater)

    def test_fixed_volume_is_default_and_does_not_show_duck_options(self):
        self.assertEqual(self.doc.audio.original_mode, "keep")
        self.assertEqual(self.doc.audio.original_volume, 20)
        self.assertEqual(self.panel.original_volume.maximum(), 200)
        self.assertFalse(self.panel.original_form.isRowVisible(self.panel.duck_volume))
        self.assertFalse(self.panel.original_form.isRowVisible(self.panel.duck_release))
        self.assertEqual(self.events, [])

    def test_manual_gain_and_fade_changes_only_require_remixing(self):
        self.panel.original_volume.setValue(45)
        self.assertEqual(self.doc.audio.original_volume, 45)
        self.panel.original_mode.set_value("duck")
        self.assertTrue(self.panel.original_form.isRowVisible(self.panel.duck_release))
        self.panel.duck_release.setValue(1.2)
        self.assertEqual(self.doc.audio.duck_release, 1.2)
        self.assertTrue(self.events)
        self.assertFalse(any(self.events))
        restored = from_dict(AudioSettings, to_dict(self.doc.audio))
        self.assertEqual(restored, self.doc.audio)

    def test_mute_disables_gain_but_preserves_selected_level(self):
        self.panel.original_volume.setValue(35)
        self.panel.original_mode.set_value("mute")
        self.assertFalse(self.panel.original_volume.isEnabled())
        self.panel.original_mode.set_value("keep")
        self.assertTrue(self.panel.original_volume.isEnabled())
        self.assertEqual(self.doc.audio.original_volume, 35)

    def test_old_duck_settings_gain_smooth_transition_defaults(self):
        audio = from_dict(AudioSettings, {"original_mode": "duck", "original_volume": 100, "duck_volume": 20})
        self.assertEqual(audio.duck_attack, 0.15)
        self.assertEqual(audio.duck_release, 0.6)
        self.doc.audio = audio
        self.panel.set_document(self.doc)
        self.assertEqual(self.panel.original_mode.value(), "duck")
        self.assertTrue(self.panel.original_form.isRowVisible(self.panel.duck_attack))
        self.assertEqual(self.events, [])


def write_audio(path: Path, seconds: float, silent: bool = False) -> None:
    samples = array("h", (
        0 if silent else int(7000 * math.sin(2 * math.pi * 440 * frame / 48000))
        for frame in range(round(seconds * 48000))
    ))
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(48000)
        audio.writeframes(samples.tobytes())


def rms(path: Path, center: float, width: float = 0.04) -> float:
    with wave.open(str(path), "rb") as audio:
        audio.setpos(round((center - width / 2) * audio.getframerate()))
        samples = array("h", audio.readframes(round(width * audio.getframerate())))
    return math.sqrt(sum(float(value) ** 2 for value in samples) / len(samples))


@unittest.skipUnless(find_tool("ffmpeg"), "cần FFmpeg để đo âm lượng thực tế")
class AudioMixTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.source = root / "original.wav"
        write_audio(self.source, 4)
        store, project = ProjectStore.create(root, "Audio")
        self.doc = store.add_video(project, str(self.source))
        self.doc.duration = 4
        self.ctx = RunContext(store=store, project=project, settings=AppSettings())
        self.root = store.cache_dir(self.doc.id)
        speech = store.tts_dir(self.doc.id) / "speech.wav"
        write_audio(speech, 0.5, silent=True)
        self.segments = [Segment(id=1, start=1, end=1.5, text="Test", tts_file=speech.name, tts_duration=0.5)]
        self.doc.audio.original_mode = "keep"
        self.doc.audio.original_volume = 100
        self.baseline = run_mix(self.ctx, self.doc, [], target=self.root / "baseline.wav")

    def ratio(self, mixed: Path, time: float) -> float:
        return rms(mixed, time) / rms(self.baseline, time)

    def test_fixed_volume_stays_constant_before_during_and_after_tts(self):
        self.doc.audio.original_volume = 20
        mixed = run_mix(self.ctx, self.doc, self.segments, target=self.root / "fixed.wav")
        for time in (0.5, 1.2, 1.55, 1.9, 3.0):
            with self.subTest(time=time):
                self.assertAlmostEqual(self.ratio(mixed, time), 0.04, delta=0.002)

    def test_ducking_recovers_gradually_instead_of_jumping_to_full_volume(self):
        audio = self.doc.audio
        audio.original_mode, audio.duck_volume = "duck", 20
        audio.duck_attack, audio.duck_release = 0.2, 0.8
        mixed = run_mix(self.ctx, self.doc, self.segments, target=self.root / "duck.wav")
        self.assertAlmostEqual(self.ratio(mixed, 0.5), 1, delta=0.02)
        self.assertTrue(0.4 < self.ratio(mixed, 0.9) < 0.8)
        self.assertAlmostEqual(self.ratio(mixed, 1.2), 0.2, delta=0.03)
        self.assertLess(self.ratio(mixed, 1.54), 0.35)
        self.assertTrue(0.45 < self.ratio(mixed, 1.9) < 0.7)
        self.assertAlmostEqual(self.ratio(mixed, 2.6), 1, delta=0.02)

    def test_video_gain_can_be_boosted_above_original_level(self):
        self.doc.audio.original_volume = 150
        mixed = run_mix(self.ctx, self.doc, self.segments, target=self.root / "boost.wav")
        self.assertAlmostEqual(self.ratio(mixed, 1.2), 1.5, delta=0.02)
        self.assertAlmostEqual(self.ratio(mixed, 2.6), 1.5, delta=0.02)

    def test_disabling_tts_keeps_chosen_original_volume(self):
        self.doc.audio.original_mode = "duck"
        self.doc.audio.original_volume = 40
        self.doc.tracks.dub = False
        mixed = run_mix(self.ctx, self.doc, self.segments, target=self.root / "no-tts.wav")
        self.assertAlmostEqual(self.ratio(mixed, 1.2), 0.16, delta=0.02)

    def test_mute_outputs_silence_without_tts(self):
        self.doc.audio.original_mode = "mute"
        mixed = run_mix(self.ctx, self.doc, [], target=self.root / "mute.wav")
        self.assertEqual(rms(mixed, 1), 0)

    def test_zero_transition_times_remain_valid(self):
        self.doc.audio.original_mode = "duck"
        self.doc.audio.duck_attack = self.doc.audio.duck_release = 0
        mixed = run_mix(self.ctx, self.doc, self.segments, target=self.root / "no-fade.wav")
        self.assertAlmostEqual(self.ratio(mixed, 1.2), 0.2, delta=0.03)
        self.assertAlmostEqual(self.ratio(mixed, 2.6), 1, delta=0.02)

    def test_ducking_respects_manual_gain_when_tts_starts_at_zero(self):
        self.doc.audio.original_mode = "duck"
        self.doc.audio.original_volume = 50
        self.segments[0].start, self.segments[0].end = 0, 0.5
        mixed = run_mix(self.ctx, self.doc, self.segments, target=self.root / "relative-duck.wav")
        self.assertAlmostEqual(self.ratio(mixed, 0.2), 0.05, delta=0.005)
        self.assertAlmostEqual(self.ratio(mixed, 2.6), 0.25, delta=0.02)

    def test_short_gaps_between_tts_do_not_raise_original_volume(self):
        self.doc.audio.original_mode = "duck"
        self.segments.append(Segment(id=2, start=1.65, end=2.15, text="Next", tts_file="speech.wav", tts_duration=0.5))
        mixed = run_mix(self.ctx, self.doc, self.segments, target=self.root / "short-gap.wav")
        self.assertAlmostEqual(self.ratio(mixed, 1.57), 0.2, delta=0.03)
        self.assertAlmostEqual(self.ratio(mixed, 3.2), 1, delta=0.02)


if __name__ == "__main__":
    unittest.main()
