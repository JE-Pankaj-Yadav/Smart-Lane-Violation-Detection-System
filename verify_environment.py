"""Fail-fast deployment verification."""
import argparse, importlib.metadata, shutil, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
EXPECTED={"Flask":"3.1.3","Werkzeug":"3.1.3","python-dotenv":"1.1.1",
"numpy":"1.26.4","opencv-python-headless":"4.10.0.84","ultralytics":"8.4.121",
"imageio-ffmpeg":"0.6.0"}
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--dgx",action="store_true"); a=ap.parse_args()
    errors=[]
    print("Python:",sys.version.split()[0])
    if sys.version_info[:2]!=(3,10): errors.append("Python 3.10.x is required.")
    for pkg,wanted in EXPECTED.items():
        try:
            got=importlib.metadata.version(pkg); print(f"{pkg}: {got}")
            if got!=wanted: errors.append(f"{pkg} must be {wanted}, found {got}.")
        except importlib.metadata.PackageNotFoundError: errors.append(f"{pkg} is not installed.")
    try:
        import numpy,cv2
        print("NumPy:",numpy.__version__); print("OpenCV:",cv2.__version__)
        if not hasattr(cv2,"VideoCapture") or not hasattr(cv2,"dnn") or not hasattr(cv2.dnn,"DictValue"): errors.append("OpenCV import is incomplete or cv2.dnn.DictValue is missing.")
    except Exception as e: errors.append(f"OpenCV import failed: {e}")
    try:
        from ultralytics import YOLO; print("Ultralytics: OK")
    except Exception as e: errors.append(f"Ultralytics import failed: {e}")
    try:
        import torch
        print("PyTorch:",torch.__version__); print("CUDA:",torch.cuda.is_available())
        if a.dgx and torch.__version__!="2.5.0a0+e000cf0ad9.nv24.10":
            errors.append("DGX PyTorch does not match the verified NVIDIA build.")
    except Exception as e: errors.append(f"PyTorch import failed: {e}")
    ff=shutil.which("ffmpeg")
    if not ff:
        try:
            import imageio_ffmpeg; ff=imageio_ffmpeg.get_ffmpeg_exe()
        except Exception as e: errors.append(f"FFmpeg unavailable: {e}")
    print("FFmpeg:",ff or "NOT FOUND")
    if not ff: errors.append("FFmpeg is required.")
    model=ROOT/"weights"/"yolo11n.pt"; print("YOLO model:","OK" if model.is_file() else "MISSING")
    if not model.is_file(): errors.append(f"Required model missing: {model}")
    if errors:
        print("\nDEPLOYMENT VERIFICATION FAILED:")
        [print(" -",e) for e in errors]
        return 1
    print("\nDEPLOYMENT VERIFICATION: PASS"); return 0
if __name__=="__main__": raise SystemExit(main())
