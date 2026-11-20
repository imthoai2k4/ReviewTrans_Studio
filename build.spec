# PyInstaller spec cho ReviewTrans Studio — thường được gọi qua scripts/build.py
# Công cụ ngoài (ffmpeg, ffprobe, libmpv) lấy từ thư mục REVIEWTRANS_TOOLS (mặc định build/tools).
import os
from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules, get_package_paths

app_dir = Path(SPECPATH)
tools_dir = Path(os.environ.get("REVIEWTRANS_TOOLS", app_dir / "build" / "tools"))
version_file = os.environ.get("REVIEWTRANS_VERSION_FILE") or None

datas = [(str(app_dir / "icon.ico"), "."), (str(app_dir / "icon.png"), ".")]
datas += [(str(app_dir / "scripts" / "funasr_worker.py"), "scripts"),
          (str(app_dir / "requirements-asr.txt"), ".")]
if tools_dir.is_dir():
    # để nguyên file .exe/.dll (datas) — không cho PyInstaller phân tích/sửa
    datas += [(str(path), "bin") for path in sorted(tools_dir.iterdir()) if path.name.lower() in ("ffmpeg.exe", "ffprobe.exe", "libmpv-2.dll")]

hiddenimports = (
    collect_submodules("reviewtrans")
    + collect_submodules("google.genai")
    + ["mpv", "py7zr"]
)

a = Analysis(
    ["main.py"],
    pathex=[str(app_dir)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=["tkinter", "pytest", "pyflakes", "PyInstaller"],
    noarchive=False,
)

# Python 3.11.0 ships VC runtime 14.32, while newer Qt wheels ship 14.44.
# Keep the matched Qt redistributables at the DLL search root; mixing these
# versions can make QtCore/QtWidgets fail with "procedure could not be found".
qt_bin = Path(get_package_paths("PyQt6")[1]) / "Qt6" / "bin"
qt_crt = sorted({p for pattern in ("vcruntime140*.dll", "msvcp140*.dll", "concrt140.dll")
                 for p in qt_bin.glob(pattern)})
qt_crt_names = {p.name.lower() for p in qt_crt}
a.binaries = [entry for entry in a.binaries if Path(entry[0]).name.lower() not in qt_crt_names]
a.binaries += [(p.name, str(p), "BINARY") for p in qt_crt]

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="ReviewTrans",
    icon=str(app_dir / "icon.ico"),
    version=version_file,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,  # UPX làm tăng cảnh báo nhầm của antivirus
    console=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="ReviewTrans",
)
