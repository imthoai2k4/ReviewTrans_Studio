from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from reviewtrans.core.pipeline import funasr


@pytest.mark.parametrize("flavor,free_gib,allowed", [
    ("cpu", 7, False), ("cpu", 8, True),
    ("cu126", 15, False), ("cu126", 16, True),
])
def test_install_checks_disk_before_downloading(tmp_path, monkeypatch, flavor, free_gib, allowed):
    monkeypatch.setattr(funasr, "user_data_dir", lambda: tmp_path)
    monkeypatch.setattr(funasr.shutil, "disk_usage", lambda _: SimpleNamespace(free=free_gib * 1024**3))
    if allowed:
        funasr.check_install_space(flavor)
    else:
        monkeypatch.setattr(funasr, "torch_flavor", lambda _: flavor)
        monkeypatch.setattr(funasr, "_install_environment", lambda: pytest.fail("Must stop before installation"))
        with pytest.raises(RuntimeError, match="Không đủ dung lượng"):
            funasr.install_funasr()


def test_installer_disables_pip_cache_and_user_packages(tmp_path, monkeypatch):
    monkeypatch.setattr(funasr, "temp_dir", lambda: tmp_path / "tmp")
    monkeypatch.setattr(funasr, "user_data_dir", lambda: tmp_path)
    env = funasr._install_environment()
    assert env["PIP_NO_CACHE_DIR"] == "1"
    assert env["PYTHONNOUSERSITE"] == "1"
    assert env["PIP_CONFIG_FILE"] == funasr.os.devnull
    assert Path(env["TEMP"]).is_relative_to(tmp_path)


@pytest.mark.parametrize("device,nvidia,cuda,expected", [
    ("auto", True, "13.0", "cu126"), ("auto", True, "12.6", "cu126"),
    ("auto", True, "11.8", "cpu"), ("auto", False, "", "cpu"),
    ("cpu", True, "13.0", "cpu"), ("cuda", False, "", "cu126"),
])
def test_install_selects_compatible_cuda_build(monkeypatch, device, nvidia, cuda, expected):
    monkeypatch.setattr(funasr, "load_cached", lambda: SimpleNamespace(
        gpus=[SimpleNamespace(vendor="nvidia" if nvidia else "intel")], cuda_version=cuda))
    monkeypatch.setattr(funasr, "runtime_info", lambda: {})
    assert funasr.torch_flavor(device) == expected


def test_runtime_and_models_resolve_after_moving_portable_folder(tmp_path, monkeypatch):
    relocated = tmp_path / "another-drive" / "ReviewTrans"
    (relocated / "asr-runtime").mkdir(parents=True)
    (relocated / "asr-runtime/python.exe").write_bytes(b"embedded python")
    (relocated / "asr-runtime/runtime-info.json").write_text(json.dumps({"torch_version": "2.8.0+cu126"}), "utf-8")
    (relocated / "asr-models").mkdir()
    monkeypatch.delenv("REVIEWTRANS_FUNASR_PYTHON", raising=False)
    monkeypatch.setattr(funasr.sys, "frozen", True, raising=False)
    monkeypatch.setattr(funasr, "exe_dir", lambda: relocated)
    monkeypatch.setattr(funasr, "user_data_dir", lambda: relocated / "data")
    monkeypatch.setattr(funasr, "temp_dir", lambda: relocated / "data/tmp")
    monkeypatch.setattr(funasr, "models_dir", lambda: relocated / "data/models")
    assert funasr.runtime_python() == relocated / "asr-runtime/python.exe"
    assert funasr.funasr_models_dir() == relocated / "asr-models"
    assert funasr.runtime_info()["torch_version"] == "2.8.0+cu126"
    work = funasr._install_environment()
    for field in ("TEMP", "TMP", "PIP_CACHE_DIR", "HF_HOME", "TORCH_HOME"):
        assert Path(work[field]).is_relative_to(relocated)


def test_user_installed_runtime_can_override_bundled_runtime(tmp_path, monkeypatch):
    (tmp_path / "data/funasr-runtime").mkdir(parents=True)
    managed = tmp_path / "data/funasr-runtime/python.exe"
    managed.write_bytes(b"updated")
    (tmp_path / "asr-runtime").mkdir()
    (tmp_path / "asr-runtime/python.exe").write_bytes(b"bundled")
    monkeypatch.delenv("REVIEWTRANS_FUNASR_PYTHON", raising=False)
    monkeypatch.setattr(funasr.sys, "frozen", True, raising=False)
    monkeypatch.setattr(funasr, "exe_dir", lambda: tmp_path)
    monkeypatch.setattr(funasr, "user_data_dir", lambda: tmp_path / "data")
    assert funasr.runtime_python() == managed


def test_bundle_includes_only_runtime_models_and_no_user_settings(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("reviewtrans_build", Path(__file__).parents[1] / "scripts/build.py")
    build = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build)
    runtime, models, dist = tmp_path / "data/funasr-runtime", tmp_path / "data/models/funasr", tmp_path / "dist"
    runtime.mkdir(parents=True)
    (runtime / "python.exe").write_bytes(b"python")
    (runtime / "python311._pth").write_text("python311.zip\n.\nLib/site-packages\nimport site\n", "utf-8")
    (runtime / "pyvenv.cfg.legacy").write_text("home = C:/old-python", "utf-8")
    (runtime / "runtime-info.json").write_text(json.dumps({
        "torch_version": "2.8.0+cu126", "cuda_version": "12.6",
        "gpu": "build-machine-gpu", "cuda_available": True, "python": "C:/build/python.exe"}), "utf-8")
    (runtime / "__pycache__").mkdir()
    (runtime / "__pycache__/old.pyc").write_bytes(b"cache")
    (tmp_path / "data/settings.json").write_text('{"api_keys":["secret"]}', "utf-8")
    for name in funasr.MODEL_NAMES:
        (models / name).mkdir(parents=True)
        (models / name / ".reviewtrans-complete").write_text("1", "utf-8")
        (models / name / "model.pt").write_bytes(b"weights")
    monkeypatch.setattr(build, "DIST", dist)
    build.bundle_asr(runtime, models)
    assert (dist / "asr-runtime/python.exe").is_file()
    assert (dist / "asr-models/paraformer-zh/model.pt").is_file()
    assert not list(dist.rglob("settings.json"))
    assert not list(dist.rglob("*.legacy"))
    assert not list(dist.rglob("__pycache__"))
    assert (dist / "portable.txt").is_file()
    info = json.loads((dist / "asr-runtime/runtime-info.json").read_text("utf-8"))
    assert info["cuda_version"] == "12.6"
    assert "gpu" not in info and "python" not in info and "cuda_available" not in info
