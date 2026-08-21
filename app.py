import logging
import os
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path

import cv2
import torch
from dotenv import load_dotenv
from flask import Flask, Response, jsonify, render_template, request, send_from_directory
from werkzeug.utils import secure_filename

from hubconfCustom import VideoDetector
from live_camera import LiveCameraService

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / '.env')
UPLOAD_DIR = BASE_DIR / os.getenv("UPLOAD_DIR", "uploads")
OUTPUT_DIR = BASE_DIR / os.getenv("OUTPUT_DIR", "outputs")
TEMP_DIR = BASE_DIR / os.getenv("TEMP_DIR", "temp")
LOG_DIR = BASE_DIR / os.getenv("LOG_DIR", "logs")
REPORT_DIR = BASE_DIR / "static" / "reports"

for directory in (UPLOAD_DIR, OUTPUT_DIR, TEMP_DIR, LOG_DIR, REPORT_DIR):
    directory.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "app.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("wrong_lane")

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "change-this-secret-key")
app.config["MAX_CONTENT_LENGTH"] = None  # No arbitrary application upload-size limit.

ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "webm", "wmv"}
jobs = {}
jobs_lock = threading.Lock()
active_job_id = None
active_job_lock = threading.Lock()


def detect_device() -> str:
    requested = os.getenv("DEVICE", "auto").strip().lower()
    if requested in {"cpu", "cuda"}:
        if requested == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but unavailable; falling back to CPU.")
            return "cpu"
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


DEVICE = detect_device()

# DEVICE must be resolved before any service that consumes it is constructed.
live_camera = LiveCameraService(BASE_DIR, DEVICE)


def ffmpeg_path() -> str | None:
    """Find FFmpeg from config, PATH, or the bundled imageio-ffmpeg binary."""
    configured = os.getenv("FFMPEG_PATH", "").strip()
    if configured:
        p = Path(configured).expanduser()
        if p.is_file():
            return str(p)
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg
    try:
        import imageio_ffmpeg
        bundled = imageio_ffmpeg.get_ffmpeg_exe()
        if bundled and Path(bundled).is_file():
            return bundled
    except Exception:
        logger.exception("Bundled FFmpeg lookup failed")
    return None


def allowed_video(filename: str) -> bool:
    return bool(filename and "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS)


def safe_video_name(original: str) -> str:
    cleaned = secure_filename(original) or "uploaded_video"
    return f"{uuid.uuid4().hex}_{cleaned}"


def processing_settings(data: dict) -> dict:
    """Validate UI/API processing settings and keep values in safe ranges."""
    try:
        confidence = float(data.get("confidence", os.getenv("CONFIDENCE_THRESHOLD", "0.50")))
        iou = float(data.get("iou", os.getenv("IOU_THRESHOLD", "0.45")))
        imgsz = int(data.get("imgsz", os.getenv("INFERENCE_SIZE", "640")))
        frame_interval = int(data.get("frame_interval", os.getenv("FRAME_INTERVAL", "1")))
    except (TypeError, ValueError) as exc:
        raise ValueError("Detection settings must contain valid numeric values.") from exc

    if not 0.05 <= confidence <= 0.99:
        raise ValueError("Confidence must be between 0.05 and 0.99.")
    if not 0.05 <= iou <= 0.99:
        raise ValueError("IoU must be between 0.05 and 0.99.")
    if imgsz < 160 or imgsz > 1920:
        raise ValueError("Inference size must be between 160 and 1920.")
    if not 1 <= frame_interval <= 30:
        raise ValueError("Frame interval must be between 1 and 30.")

    return {
        "confidence": confidence,
        "iou": iou,
        "imgsz": imgsz,
        "frame_interval": frame_interval,
    }


def probe_video(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError("The uploaded file is not a readable video.")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration = frame_count / fps if fps > 0 else 0.0
    cap.release()

    if width <= 0 or height <= 0:
        raise ValueError("Video metadata could not be read.")
    return {
        "fps": fps or 30.0,
        "frames": frame_count,
        "width": width,
        "height": height,
        "duration": duration,
    }


def update_job(job_id: str, **values):
    with jobs_lock:
        if job_id in jobs:
            jobs[job_id].update(values)


def process_job(job_id: str, input_path: Path, settings: dict):
    global active_job_id
    start = time.perf_counter()
    # Background threads do not inherit Flask application/request context.
    with app.app_context():
        _process_job_context(job_id, input_path, settings, start)


def _process_job_context(job_id: str, input_path: Path, settings: dict, start: float):
    global active_job_id
    start = time.perf_counter()
    try:
        metadata = probe_video(input_path)
        update_job(job_id, status="processing", progress=0, metadata=metadata, device=DEVICE)
        logger.info("Processing started: %s | %s", job_id, metadata)

        detector = VideoDetector(
            model_path=BASE_DIR / "weights" / "yolo11n.pt",
            anpr_model_path=BASE_DIR / "weights" / "anpr_car.pt",
            device=DEVICE,
            confidence=float(settings.get("confidence", os.getenv("CONFIDENCE_THRESHOLD", "0.50"))),
            iou=float(settings.get("iou", os.getenv("IOU_THRESHOLD", "0.45"))),
            imgsz=int(settings.get("imgsz", os.getenv("INFERENCE_SIZE", "640"))),
            frame_interval=max(1, int(settings.get("frame_interval", os.getenv("FRAME_INTERVAL", "1")))),
            frames_dir=BASE_DIR / "frames",
        )

        raw_output = TEMP_DIR / f"{job_id}_annotated.mp4"
        detector.process_video(
            input_path=input_path,
            output_path=raw_output,
            progress_callback=lambda p: update_job(job_id, **p),
        )

        final_output = OUTPUT_DIR / f"{job_id}.mp4"
        final_path = finalize_video(raw_output, input_path, final_output, detector.fps or metadata["fps"])

        elapsed = time.perf_counter() - start
        report = detector.report_text()
        (REPORT_DIR / f"{job_id}_detections.txt").write_text(report, encoding="utf-8")

        update_job(
            job_id,
            status="completed",
            progress=100,
            # This function runs in a background thread, outside Flask's request
            # context. Build relative browser URLs directly instead of calling
            # url_for(), which requires an active Flask application/request context.
            output_url=f"/output/{final_path.name}",
            report_url=f"/api/report/{job_id}",
            processing_time=round(elapsed, 2),
            stats=detector.stats(),
            message="Video processed successfully.",
        )
        logger.info("Processing completed: %s in %.2fs", job_id, elapsed)
    except Exception as exc:
        logger.exception("Processing failed: %s", job_id)
        update_job(job_id, status="failed", message="Video processing failed.", error=str(exc))
    finally:
        try:
            input_path.unlink(missing_ok=True)
        except Exception:
            logger.warning("Could not remove upload: %s", input_path)
        try:
            (TEMP_DIR / f"{job_id}_annotated.mp4").unlink(missing_ok=True)
        except Exception:
            logger.warning("Could not remove temporary output for job: %s", job_id)
        with active_job_lock:
            if active_job_id == job_id:
                active_job_id = None


def _verify_output_video(path: Path) -> None:
    """Verify that the generated file can actually be opened by OpenCV."""
    if not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError("Processed video file was not created.")
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise RuntimeError("Processed video was created but cannot be opened.")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if width <= 0 or height <= 0 or frames <= 0:
            raise RuntimeError("Processed video failed integrity verification.")
    finally:
        cap.release()


def finalize_video(raw_output: Path, original: Path, final_output: Path, fps: float) -> Path:
    """Prefer browser-compatible H.264/AAC through FFmpeg and verify the result."""
    ffmpeg = ffmpeg_path()
    if ffmpeg:
        cmd = [
            ffmpeg, "-y",
            "-i", str(raw_output),
            "-i", str(original),
            "-map", "0:v:0",
            "-map", "1:a:0?",
            "-c:v", "libx264",
            "-preset", os.getenv("FFMPEG_PRESET", "veryfast"),
            "-crf", os.getenv("FFMPEG_CRF", "23"),
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", "128k",
            "-movflags", "+faststart",
            str(final_output),
        ]
        completed = subprocess.run(cmd, capture_output=True, text=True, timeout=None)
        if completed.returncode == 0 and final_output.exists() and final_output.stat().st_size > 0:
            try:
                _verify_output_video(final_output)
                raw_output.unlink(missing_ok=True)
                return final_output
            except Exception as exc:
                logger.error("FFmpeg output verification failed: %s", exc)
                final_output.unlink(missing_ok=True)
        logger.error("FFmpeg failed: %s", completed.stderr[-3000:])

    # OpenCV mp4v fallback. It is still verified before being exposed to the UI.
    shutil.move(str(raw_output), str(final_output))
    try:
        _verify_output_video(final_output)
    except Exception:
        final_output.unlink(missing_ok=True)
        raise RuntimeError(
            "Processed video was created but could not be verified. "
            "Install/enable FFmpeg for browser-compatible H.264 output."
        )
    return final_output


@app.get("/")
def index():
    return render_template("index.html", device=DEVICE, ffmpeg_available=bool(ffmpeg_path()))


@app.post("/api/upload")
def upload():
    video = request.files.get("video")
    if video is None or not video.filename:
        return jsonify(success=False, message="Please select a video file."), 400
    if not allowed_video(video.filename):
        return jsonify(success=False, message="Unsupported video format. Use MP4, AVI, MOV, MKV, WebM or WMV."), 415

    filename = safe_video_name(video.filename)
    destination = UPLOAD_DIR / filename
    try:
        logger.info("Upload started: %s", video.filename)
        # Werkzeug streams the multipart body to its temporary storage; save() then
        # writes the file to disk instead of loading it into application RAM.
        video.save(destination)
        metadata = probe_video(destination)
        logger.info("Upload completed: %s | %s", destination.name, metadata)
        return jsonify(
            success=True,
            message="Video uploaded successfully.",
            filename=destination.name,
            original_name=video.filename,
            metadata=metadata,
        )
    except Exception as exc:
        destination.unlink(missing_ok=True)
        logger.exception("Upload failed")
        return jsonify(success=False, message="The video could not be uploaded or read.", error=str(exc)), 400


@app.post("/api/process")
def start_processing():
    global active_job_id
    data = request.get_json(silent=True) or {}
    filename = data.get("filename", "")
    input_path = UPLOAD_DIR / Path(filename).name

    if not filename or not input_path.exists():
        return jsonify(success=False, message="Uploaded video not found. Please upload it again."), 404

    with active_job_lock:
        if active_job_id:
            with jobs_lock:
                current = jobs.get(active_job_id, {})
            if current.get("status") in {"queued", "processing"}:
                return jsonify(success=False, message="Another video is already being processed."), 409

        job_id = uuid.uuid4().hex
        active_job_id = job_id

    try:
        settings = processing_settings(data)
    except ValueError as exc:
        return jsonify(success=False, message=str(exc)), 400
    with jobs_lock:
        jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "progress": 0,
            "device": DEVICE,
            "message": "Processing queued.",
            "stats": {},
        }

    thread = threading.Thread(target=process_job, args=(job_id, input_path, settings), daemon=True)
    thread.start()
    return jsonify(success=True, job_id=job_id)


@app.get("/api/progress/<job_id>")
def progress(job_id):
    with jobs_lock:
        job = jobs.get(job_id)
    if not job:
        return jsonify(success=False, message="Job not found."), 404
    return jsonify(success=True, **job)


@app.get("/output/<path:filename>")
def serve_output(filename):
    return send_from_directory(OUTPUT_DIR, filename, conditional=True, max_age=0)


@app.get("/api/report/<job_id>")
def download_report(job_id):
    report = REPORT_DIR / f"{job_id}_detections.txt"
    if not report.exists():
        return jsonify(success=False, message="Report is not available yet."), 404
    return send_from_directory(REPORT_DIR, report.name, as_attachment=True, download_name="detections_summary.txt")


@app.post("/api/live/start")
def live_start():
    data = request.get_json(silent=True) or {}
    try:
        live_camera.start(
            data.get("source", ""),
            confidence=float(data.get("confidence", 0.50)),
            iou=float(data.get("iou", 0.45)),
            imgsz=int(data.get("imgsz", 640)),
            frame_interval=max(1, int(data.get("frame_interval", 1))),
        )
        return jsonify(success=True, message="Live camera started.", **live_camera.snapshot())
    except Exception as exc:
        logger.exception("Live camera start failed")
        return jsonify(success=False, message=str(exc)), 400


@app.post("/api/live/stop")
def live_stop():
    live_camera.stop()
    return jsonify(success=True, message="Live camera stopped.", **live_camera.snapshot())


@app.get("/api/live/status")
def live_status():
    return jsonify(success=True, **live_camera.snapshot())


@app.get("/live/feed")
def live_feed():
    return Response(live_camera.mjpeg(), mimetype="multipart/x-mixed-replace; boundary=frame", headers={"Cache-Control": "no-cache, no-store, must-revalidate", "Pragma": "no-cache"})


@app.get("/health")
def health():
    model_path = BASE_DIR / "weights" / "yolo11n.pt"
    return jsonify(
        success=True,
        device=DEVICE,
        cuda_available=bool(torch.cuda.is_available()),
        ffmpeg_available=bool(ffmpeg_path()),
        model_available=model_path.is_file(),
    )


@app.errorhandler(413)
def too_large(_):
    return jsonify(success=False, message="The server rejected the upload before processing. Check the web server/proxy configuration."), 413


if __name__ == "__main__":
    logger.info("Starting Wrong Lane Vehicle Detection | device=%s | ffmpeg=%s", DEVICE, ffmpeg_path())
    app.run(debug=False, host="0.0.0.0", port=int(os.getenv("PORT", "5000")), threaded=True)
