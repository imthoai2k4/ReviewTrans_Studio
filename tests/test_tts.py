from __future__ import annotations

import shutil
import math
import tempfile
import unittest
import wave
from array import array
from pathlib import Path
from unittest.mock import patch

from reviewtrans.core.config import AppSettings, ProviderProfile
from reviewtrans.core.models import Segment
from reviewtrans.core.pipeline import RunContext
from reviewtrans.core.pipeline import tts
from reviewtrans.core.proc import MediaInfo, StopRequested
from reviewtrans.core.store import ProjectStore


def write_wav(path: Path, frames: int = 8000) -> None:
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(8000)
        audio.writeframes(array("h", (int(5000 * math.sin(2 * math.pi * 440 * i / 8000)) for i in range(frames))).tobytes())


def probe_wav(path: Path) -> MediaInfo:
    with wave.open(str(path), "rb") as audio:
        return MediaInfo(duration=audio.getnframes() / audio.getframerate(), has_audio=True)


class TtsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store, project = ProjectStore.create(Path(self.temp.name), "TTS")
        self.doc = self.store.add_video(project, "video.mp4")
        self.profile = ProviderProfile(kind="openai_speech", name="test", options={"voice": "test"})
        settings = AppSettings(
            tts_profiles=[self.profile], default_tts_profile=self.profile.id, tts_concurrency=1,
        )
        self.logs = []
        self.ctx = RunContext(store=self.store, project=project, settings=settings, log=self.logs.append)
        self.tts_dir = self.store.tts_dir(self.doc.id)
        self.provider = unittest.mock.Mock(voice="test")
        self.provider.synthesize.side_effect = self.synthesize
        self.addCleanup(patch.stopall)
        patch.object(tts, "create_tts", return_value=self.provider).start()
        patch.object(tts, "_convert_to_wav", side_effect=lambda raw, target, ctx: shutil.copyfile(raw, target)).start()
        patch.object(tts, "probe", side_effect=probe_wav).start()

    def synthesize(self, text, voice, speed, base, stop):
        raw = base.with_suffix(".wav")
        write_wav(raw)
        return raw

    def segment(self, ident=1):
        return Segment(id=ident, start=ident, end=ident + 1, text=f"Line {ident}")

    def block_unlink(self, predicate):
        unlink = Path.unlink

        def remove(path, *args, **kwargs):
            if predicate(path):
                raise PermissionError(13, "[WinError 32] File is being used by another process", str(path))
            return unlink(path, *args, **kwargs)

        patch.object(Path, "unlink", remove).start()

    def test_locked_old_file_does_not_abort_remaining_segments_or_save(self):
        old = self.tts_dir / "old.wav"
        write_wav(old)
        segments = [self.segment(6), self.segment(7)]
        segments[0].tts_file, segments[0].tts_duration = old.name, 0.1
        self.block_unlink(lambda path: path == old)
        progress = []
        self.ctx.progress = lambda pct, message: progress.append(pct)

        result = tts.run_tts(
            self.ctx, self.doc, segments,
            save=lambda: self.store.save_segments(self.doc.id, segments),
        )

        self.assertEqual(result, (2, 0))
        self.assertTrue(old.exists())
        saved = self.store.load_segments(self.doc.id)
        self.assertEqual([s.tts_file for s in saved], [s.tts_file for s in segments])
        self.assertTrue(all(s.tts_duration > 0 and (self.tts_dir / s.tts_file).exists() for s in saved))
        self.assertNotEqual(segments[0].tts_file, old.name)
        self.assertEqual(progress[-1], 100)
        self.assertTrue(any("old.wav" in line and "WinError 32" in line for line in self.logs))

    def test_locked_raw_cleanup_does_not_retry_successful_synthesis(self):
        seg = self.segment()
        self.block_unlink(lambda path: path.name.startswith("raw_"))

        self.assertEqual(tts.run_tts(self.ctx, self.doc, [seg]), (1, 0))
        self.assertEqual(self.provider.synthesize.call_count, 1)
        self.assertGreater(seg.tts_duration, 0)
        self.assertTrue((self.tts_dir / seg.tts_file).exists())

    def test_force_with_same_key_does_not_overwrite_playing_file(self):
        seg = self.segment()
        tts.run_tts(self.ctx, self.doc, [seg])
        old = self.tts_dir / seg.tts_file
        original = old.read_bytes()
        key = seg.tts_key
        self.block_unlink(lambda path: path == old)

        self.assertEqual(tts.run_tts(self.ctx, self.doc, [seg], force=True), (1, 0))
        self.assertEqual(seg.tts_key, key)
        self.assertNotEqual(seg.tts_file, old.name)
        self.assertEqual(old.read_bytes(), original)
        self.assertEqual(tts.run_tts(self.ctx, self.doc, [seg]), (0, 0))
        self.assertEqual(self.provider.synthesize.call_count, 2)

    def test_empty_audio_never_replaces_existing_result(self):
        seg = self.segment()
        old = self.tts_dir / "old.wav"
        write_wav(old)
        seg.tts_file, seg.tts_duration, seg.tts_key = old.name, 0.1, "old-key"

        def empty(text, voice, speed, base, stop):
            raw = base.with_suffix(".wav")
            write_wav(raw, frames=0)
            return raw

        self.provider.synthesize.side_effect = empty
        self.assertEqual(tts.run_tts(self.ctx, self.doc, [seg], force=True), (0, 1))
        self.assertEqual((seg.tts_file, seg.tts_duration, seg.tts_key), (old.name, 0.1, "old-key"))
        self.assertEqual(self.provider.synthesize.call_count, 3)
        self.assertEqual(list(self.tts_dir.iterdir()), [old])

    def test_empty_first_response_is_retried(self):
        seg = self.segment()

        def response(text, voice, speed, base, stop):
            raw = base.with_suffix(".wav")
            write_wav(raw, frames=0 if self.provider.synthesize.call_count == 1 else 8000)
            return raw

        self.provider.synthesize.side_effect = response
        self.assertEqual(tts.run_tts(self.ctx, self.doc, [seg]), (1, 0))
        self.assertEqual(self.provider.synthesize.call_count, 2)
        self.assertGreater(seg.tts_duration, 0)

    def test_cached_zero_duration_audio_is_generated_again(self):
        seg = self.segment()
        seg.tts_file = "empty.wav"
        seg.tts_key = tts.segment_key(seg, self.profile, "test", self.doc.audio.speed)
        write_wav(self.tts_dir / seg.tts_file, frames=0)

        self.assertIsNone(tts.tts_path(self.ctx, self.doc, seg))
        self.assertEqual(tts.run_tts(self.ctx, self.doc, [seg]), (1, 0))
        self.assertGreater(seg.tts_duration, 0)

    def test_invalid_probe_results_are_not_published(self):
        for info in (
            MediaInfo(duration=1, has_audio=False),
            MediaInfo(duration=float("nan"), has_audio=True),
            MediaInfo(duration=float("inf"), has_audio=True),
        ):
            with self.subTest(info=info), patch.object(tts, "probe", return_value=info):
                seg = self.segment()
                self.assertEqual(tts.run_tts(self.ctx, self.doc, [seg]), (0, 1))
                self.assertEqual(seg.tts_file, "")
                self.assertFalse(list(self.tts_dir.iterdir()))

    def test_stop_is_propagated(self):
        self.provider.synthesize.side_effect = StopRequested()
        with self.assertRaises(StopRequested):
            tts.run_tts(self.ctx, self.doc, [self.segment()])

    def test_zero_or_blank_concurrency_uses_one_worker(self):
        executor = tts.ThreadPoolExecutor
        self.ctx.settings.tts_concurrency = 0
        for value in (None, "", "0", 0):
            with self.subTest(concurrency=value), patch.object(tts, "ThreadPoolExecutor", wraps=executor) as pool:
                self.profile.options["concurrency"] = value
                self.assertEqual(tts.run_tts(self.ctx, self.doc, [self.segment()], force=True), (1, 0))
                self.assertEqual(pool.call_args.kwargs["max_workers"], 1)


if __name__ == "__main__":
    unittest.main()
