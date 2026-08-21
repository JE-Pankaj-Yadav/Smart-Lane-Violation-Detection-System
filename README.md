# Wrong Lane Vehicle Detection — Deployment Ready

A Flask + YOLO/ByteTrack application for vehicle detection and wrong-lane analysis. The project supports uploaded traffic videos and live IP-camera streams.

## Current runtime

- Python 3.10
- Flask 3.1.3
- NumPy 1.26.4
- OpenCV headless 4.10.0.84
- Ultralytics 8.4.121
- imageio-ffmpeg 0.6.0
- CPU fallback and automatic CUDA selection
- DGX compatibility with the verified NVIDIA PyTorch build: `2.5.0a0+e000cf0ad9.nv24.10`

## Important Docker fix

This is a **Flask application listening on port 5000**. Do not map host port 5000 to container port 8501.

Use:

```bash
docker build -t smart-lane-violation-detection-system:latest .
docker run --rm -p 5000:5000 smart-lane-violation-detection-system:latest
```

Then open `http://localhost:5000`.

If you specifically want host port 8501:

```bash
docker run --rm -p 8501:5000 smart-lane-violation-detection-system:latest
```

Then open `http://localhost:8501`.

## Root cause of the reported Docker crash

The previous `app.py` created `LiveCameraService(BASE_DIR, DEVICE)` before `DEVICE` was defined. Python therefore stopped at startup with:

`NameError: name 'DEVICE' is not defined`

The service is now created only after `DEVICE = detect_device()`.

The earlier Flask background-thread application-context problem is also retained as a permanent fix: the processing worker enters `with app.app_context()` before any Flask-dependent work.

## Features

- Video upload with unique filenames
- MP4, AVI, MOV, MKV, WebM and WMV validation
- Streaming/frame-by-frame processing
- YOLO vehicle detection + ByteTrack
- Resolution-aware lane polygons
- Wrong-lane detection
- Configurable confidence, IoU, inference size and frame interval
- Real processing progress and statistics
- Browser-accessible processed-video URLs
- H.264/MP4 output through FFmpeg when available
- Output integrity verification
- CPU/GPU auto-selection
- Live RTSP/HTTP/HTTPS camera detection
- Responsive dashboard
- Structured logs and reports
- Cross-platform Windows/Linux paths

## Live IP Camera

Enter a stream URL reachable **from the machine/container running Flask**:

- `rtsp://user:password@192.168.1.100:554/stream`
- `http://192.168.1.100:8080/video`
- `https://camera.example/video`
- `0` for a webcam physically attached to the server

The browser does not open the RTSP stream directly. Flask/OpenCV opens it on the server and sends an MJPEG preview to the browser.

For Docker, the camera network must be reachable from inside the container. A camera visible only to your Windows PC is not automatically visible to a remote DGX/container.

For security, the API response redacts camera credentials from the displayed source URL.

## Configuration

Copy `.env.example` to `.env` and adjust values as needed:

- `DEVICE=auto|cpu|cuda`
- `CONFIDENCE_THRESHOLD`
- `IOU_THRESHOLD`
- `INFERENCE_SIZE`
- `FRAME_INTERVAL`
- `FFMPEG_PATH`
- `FFMPEG_PRESET`
- `FFMPEG_CRF`
- upload/output/temp/log directories
- optional e-mail alert settings

Do not commit `.env` or credentials.

## Windows

```powershell
run_windows.bat
```

Or manually:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-windows.txt
python verify_environment.py
python app.py
```

## Linux CPU

```bash
./run_linux.sh
```

## DGX

The DGX launcher preserves the preinstalled NVIDIA PyTorch build and repairs the known OpenCV/NumPy conflict before verification:

```bash
./run_dgx.sh
```

For a manual OpenCV repair:

```bash
./repair_dgx_opencv.sh
python3 verify_environment.py --dgx
```

The expected verification result is:

`DEPLOYMENT VERIFICATION: PASS`

## Docker CPU

```bash
docker build -t smart-lane-violation-detection-system:latest .
docker run --rm -p 5000:5000 smart-lane-violation-detection-system:latest
```

The included `Dockerfile` is the CPU deployment. `Dockerfile.gpu.example` is a template only; for the DGX environment, prefer the native `run_dgx.sh` flow so the verified NVIDIA PyTorch build is not replaced by an arbitrary wheel.

## Project directories

The application creates these directories automatically:

- `uploads/` — incoming files
- `outputs/` — final browser-ready videos
- `temp/` — intermediate video files
- `frames/` — alert snapshots
- `logs/` — application logs
- `static/reports/` — detection reports
- `weights/` — model files

Uploaded source videos are removed after processing completes or fails. Final output videos remain available until explicitly cleaned.

## Troubleshooting

### `NameError: name 'DEVICE' is not defined`

Use the updated `app.py`. This was caused by initializing `LiveCameraService` before device detection.

### `Working outside of application context`

The background processing worker now wraps processing in:

```python
with app.app_context():
    ...
```

and constructs browser URLs directly instead of calling `url_for()` from the worker.

### OpenCV `cv2.dnn.DictValue` error on DGX

Run:

```bash
./repair_dgx_opencv.sh
python3 verify_environment.py --dgx
```

The project pins NumPy 1.26.4 and OpenCV 4.10.0.84 to avoid the known incompatible OpenCV/NumPy stack.

### Browser does not play the output

Check that FFmpeg is available. The application prefers H.264 + `yuv420p` + `+faststart`, then verifies the generated file before returning success.

## Verification

Run:

```bash
python verify_environment.py
```

The application should not be considered deployment-ready until verification passes.
