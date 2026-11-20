from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from reviewtrans.core import paths
from reviewtrans.core.config import SettingsStore
from reviewtrans.core.pipeline import resources


class LocalDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "app"
        self.root.mkdir()
        self.home = Path(self.temp.name) / "user"
        self.home.mkdir()
        self.addCleanup(patch.stopall)
        patch.dict(os.environ, {"REVIEWTRANS_DATA_DIR": "", "REVIEWTRANS_PORTABLE": ""}).start()
        patch.object(sys, "frozen", False, create=True).start()
        patch.object(paths, "exe_dir", return_value=self.root).start()
        patch.object(paths, "app_root", return_value=self.root).start()
        patch.object(Path, "home", return_value=self.home).start()

    def test_source_data_is_local_even_from_another_working_directory(self):
        original = Path.cwd()
        try:
            os.chdir(self.home)
            self.assertEqual(paths.user_data_dir(), self.root / "data")
            self.assertEqual(paths.default_projects_root(), self.root / "data" / "projects")
        finally:
            os.chdir(original)
        self.assertFalse((self.home / paths.USER_DIR_NAME).exists())

    def test_resource_folders_share_local_data_root(self):
        for fn, name in (
            (paths.user_bin_dir, "bin"), (paths.models_dir, "models"),
            (paths.fonts_dir, "fonts"), (paths.temp_dir, "tmp"),
        ):
            with self.subTest(folder=name):
                self.assertEqual(fn(), self.root / "data" / name)
                self.assertTrue(fn().is_dir())

    def test_fresh_settings_do_not_import_old_configuration_or_projects(self):
        legacy = self.home / paths.USER_DIR_NAME
        legacy.mkdir()
        old_settings = {"default_whisper_model": "old-model", "recent_projects": ["old-project"]}
        (legacy / "settings.json").write_text(json.dumps(old_settings), encoding="utf-8")
        (legacy / "config.json").write_text('{"gemini_api_keys":["old-key"]}', encoding="utf-8")
        store = SettingsStore()
        self.assertEqual(store.path, self.root / "data" / "settings.json")
        self.assertEqual(store.settings.recent_projects, [])
        self.assertEqual(store.settings.last_project, "")
        self.assertEqual(store.settings.default_asr_engine, "paraformer")
        self.assertFalse(any(p.api_keys for p in store.settings.translate_profiles))
        self.assertEqual(Path(store.settings.projects_root), self.root / "data" / "projects")
        self.assertEqual(json.loads((legacy / "settings.json").read_text()), old_settings)

    def test_tools_and_models_are_discovered_in_new_folder(self):
        old_bin = self.home / paths.USER_DIR_NAME / "bin"
        old_bin.mkdir(parents=True)
        (old_bin / "ffmpeg.exe").write_bytes(b"old")
        with patch.object(paths, "IS_WINDOWS", True):
            self.assertIsNone(paths.find_tool("ffmpeg"))
            binary = paths.user_bin_dir() / "ffmpeg.exe"
            binary.write_bytes(b"new")
            self.assertEqual(paths.find_tool("ffmpeg"), binary)
            dll = paths.user_bin_dir() / "libmpv-2.dll"
            dll.write_bytes(b"new")
            self.assertEqual(paths.find_libmpv(), dll)
    def test_data_override_applies_to_resources_and_projects(self):
        custom = Path(self.temp.name) / "custom"
        with patch.dict(os.environ, {"REVIEWTRANS_DATA_DIR": str(custom)}):
            self.assertEqual(paths.user_data_dir(), custom)
            self.assertEqual(paths.models_dir(), custom / "models")
            self.assertEqual(paths.default_projects_root(), custom / "projects")

    def test_installed_and_portable_paths_remain_supported(self):
        with patch.object(sys, "frozen", True):
            self.assertEqual(paths.user_data_dir(), self.home / paths.USER_DIR_NAME)
            self.assertEqual(paths.default_projects_root(), self.home / "ReviewTrans Projects")
            (self.root / "portable.txt").write_text("portable")
            self.assertEqual(paths.user_data_dir(), self.root / "data")
            self.assertEqual(paths.default_projects_root(), self.root / "data" / "projects")

    def test_tool_downloads_extract_under_local_tmp_and_install_under_local_bin(self):
        archives = []

        def download(url, target, *args):
            self.assertTrue(target.is_relative_to(self.root / "data" / "tmp"))
            archives.append(target)
            with zipfile.ZipFile(target, "w") as archive:
                for name in ("ffmpeg.exe", "ffprobe.exe"):
                    archive.writestr(f"package/{name}", b"test")
            return target

        def extract(archive, dest):
            (dest / "libmpv-2.dll").write_bytes(b"test")

        with (
            patch.object(resources.platform, "system", return_value="Windows"),
            patch.object(resources, "download_file", side_effect=download),
            patch.object(resources, "_github_asset", return_value=("test.zip", "https://example.invalid/test.zip")),
            patch.object(resources, "extract_7z", side_effect=extract),
        ):
            for fn in (
                resources.download_ffmpeg, resources.download_libmpv,
            ):
                with self.subTest(download=fn.__name__):
                    installed = fn()
                    self.assertTrue(installed.is_relative_to(self.root / "data" / "bin"))
                    self.assertTrue(installed.exists())
        self.assertEqual(len(archives), 2)
        self.assertFalse(list(paths.temp_dir().iterdir()))
        self.assertFalse((self.home / paths.USER_DIR_NAME).exists())


if __name__ == "__main__":
    unittest.main()
