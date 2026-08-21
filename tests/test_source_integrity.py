"""Lightweight deployment checks that require only the Python standard library."""
from pathlib import Path
import ast

ROOT = Path(__file__).resolve().parents[1]

def test_python_sources_parse():
    for path in ROOT.glob("*.py"):
        ast.parse(path.read_text(encoding="utf-8"))

def test_device_is_defined_before_live_camera():
    lines = (ROOT / "app.py").read_text(encoding="utf-8").splitlines()
    device_line = next(i for i, line in enumerate(lines) if line.startswith("DEVICE = detect_device()"))
    live_line = next(i for i, line in enumerate(lines) if "LiveCameraService(BASE_DIR, DEVICE)" in line)
    assert device_line < live_line

def test_required_files_exist():
    for relative in [
        "app.py", "hubconfCustom.py", "live_camera.py",
        "templates/index.html", "static/css/styles.css",
        "weights/yolo11n.pt", "weights/anpr_car.pt",
    ]:
        assert (ROOT / relative).exists(), relative
