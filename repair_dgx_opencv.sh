#!/usr/bin/env bash
set -euo pipefail
PYTHON_BIN="${PYTHON_BIN:-python3}"
SITE="$($PYTHON_BIN -c 'import site; print(site.getsitepackages()[0])')"
echo "Using Python: $PYTHON_BIN"
echo "Site-packages: $SITE"

# Fast path: do not reinstall when the exact clean stack is already working.
if $PYTHON_BIN - <<'PY'
try:
    import cv2, numpy
    assert numpy.__version__ == "1.26.4"
    assert cv2.__version__ == "4.10.0"
    assert hasattr(cv2.dnn, "DictValue")
except Exception:
    raise SystemExit(1)
PY
then
    echo "OpenCV/NumPy stack is already clean and verified."
    exit 0
fi

# Remove every pip OpenCV distribution and stale cv2 namespace left by older images.
$PYTHON_BIN -m pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python opencv-contrib-python-headless opencv || true
rm -rf "$SITE/cv2" "$SITE/opencv" "$SITE"/opencv_*.dist-info "$SITE"/opencv*.dist-info
$PYTHON_BIN -m pip install --no-cache-dir --force-reinstall numpy==1.26.4 opencv-python-headless==4.10.0.84

$PYTHON_BIN - <<'PY'
import cv2, numpy
print("NumPy:", numpy.__version__)
print("OpenCV:", cv2.__version__)
print("cv2 path:", cv2.__file__)
print("VideoCapture:", hasattr(cv2, "VideoCapture"))
print("DNN DictValue:", hasattr(cv2.dnn, "DictValue"))
assert numpy.__version__ == "1.26.4"
assert cv2.__version__ == "4.10.0"
assert hasattr(cv2, "VideoCapture")
assert hasattr(cv2.dnn, "DictValue")
PY
