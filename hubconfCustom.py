import logging
import os
import smtplib
import threading
import time
from datetime import datetime
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

import cv2
import numpy as np
import torch
from dotenv import load_dotenv
from ultralytics import YOLO

logger = logging.getLogger("wrong_lane.detector")
load_dotenv(Path(__file__).resolve().parent / ".env")

_email_lock = threading.Lock()
_last_email_time = 0.0


class VideoDetector:
    """Streaming YOLO/ByteTrack detector with resolution-aware lane regions."""

    def __init__(self, model_path, anpr_model_path, device="auto", confidence=0.5,
                 iou=0.45, imgsz=640, frame_interval=1, frames_dir=None):
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.confidence = float(confidence)
        self.iou = float(iou)
        self.imgsz = int(imgsz)
        self.frame_interval = max(1, int(frame_interval))
        self.model = YOLO(str(model_path))
        self.model.to(self.device)
        self.anpr_model = YOLO(str(anpr_model_path)) if Path(anpr_model_path).exists() else None
        if self.anpr_model is not None:
            self.anpr_model.to(self.device)

        self.wup = {}
        self.wrongway = set()
        self.anpr_processed = set()
        self.vehicle_number = set()
        self.vehicle_count = set()
        self.frames_processed = 0
        self.total_frames = 0
        self.fps = 0.0
        self.video_fps = 0.0
        self.processing_fps = 0.0
        self.processing_started = 0.0
        self.report = []
        self.frames_dir = Path(frames_dir) if frames_dir else Path(__file__).resolve().parent / "frames"

    @staticmethod
    def _lane_polygons(width, height):
        # Original regions were designed for 640x360. Scale them to the current
        # frame instead of assuming every video has that exact resolution.
        sx, sy = width / 640.0, height / 360.0
        area1 = [(399, 257), (177, 257), (290, 213), (406, 213)]
        area2 = [(391, 326), (0, 326), (91, 290), (398, 290)]
        scale = lambda pts: np.array([(int(x * sx), int(y * sy)) for x, y in pts], np.int32)
        return scale(area1), scale(area2)

    def _send_email_async(self, obj_id, frame):
        global _last_email_time
        enabled = os.getenv("ALERT_EMAIL_ENABLED", "false").lower() == "true"
        recipient = os.getenv("EMAIL_RECIPIENT", "").strip()
        sender = os.getenv("EMAIL_SENDER", "").strip()
        password = os.getenv("SENDER_PASSWORD", "").strip()
        if not enabled or not recipient or not sender or not password:
            return

        with _email_lock:
            now = time.time()
            cooldown = int(os.getenv("EMAIL_COOLDOWN_SECONDS", "600"))
            if now - _last_email_time < cooldown:
                return
            _last_email_time = now

        def send():
            try:
                frames_dir = self.frames_dir
                frames_dir.mkdir(parents=True, exist_ok=True)
                path = frames_dir / f"violation_{datetime.now():%Y%m%d_%H%M%S}_{obj_id}.jpeg"
                cv2.imwrite(str(path), frame)
                msg = MIMEMultipart()
                msg["From"], msg["To"], msg["Subject"] = sender, recipient, "Lane Violation Detected"
                msg.attach(MIMEText(f"Wrong Side Vehicle Detected: Vehicle_{obj_id}", "plain"))
                with open(path, "rb") as f:
                    part = MIMEImage(f.read(), _subtype="jpeg")
                msg.attach(part)
                with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=20) as server:
                    server.login(sender, password)
                    server.sendmail(sender, recipient, msg.as_string())
                logger.info("Violation email sent to configured recipient.")
            except Exception:
                logger.exception("Violation email failed")

        threading.Thread(target=send, daemon=True).start()

    def _anpr(self, obj_id, vehicle_crop, full_frame):
        if self.anpr_model is None or obj_id in self.anpr_processed:
            return
        self.anpr_processed.add(obj_id)
        try:
            results = self.anpr_model.predict(
                source=vehicle_crop,
                imgsz=320,
                conf=0.5,
                device=self.device,
                half=(self.device == "cuda"),
                verbose=False,
            )
            if not results or results[0].boxes is None:
                return
            # The current supplied ANPR model detects plate regions but does not
            # provide OCR text. Do not fabricate a plate number.
            if len(results[0].boxes) > 0:
                label = f"Vehicle_{obj_id}"
                if label not in self.vehicle_number:
                    self.vehicle_number.add(label)
                    self.report.append(
                        f"[{datetime.now():%Y-%m-%d %H:%M:%S}] Plate region detected: {label}"
                    )
                    self._send_email_async(obj_id, full_frame.copy())
        except Exception:
            logger.exception("ANPR processing failed for vehicle %s", obj_id)

    def process_video(self, input_path, output_path, progress_callback=None):
        """Stream-process a video without loading it into memory."""
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise ValueError("Could not open the uploaded video.")

        self.video_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        if self.video_fps <= 0:
            self.video_fps = 30.0
        self.fps = self.video_fps
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if width <= 0 or height <= 0:
            cap.release()
            raise ValueError("Invalid video dimensions.")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.video_fps,
            (width, height),
        )
        if not writer.isOpened():
            cap.release()
            raise RuntimeError("Could not initialize video output writer.")

        area1, area2 = self._lane_polygons(width, height)
        self.processing_started = time.perf_counter()
        frame_index = 0
        last_progress = 0.0
        last_detections = []

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame_index += 1

                should_infer = ((frame_index - 1) % self.frame_interval == 0)

                if should_infer:
                    inference_start = time.perf_counter()
                    results = self.model.track(
                        source=frame,
                        persist=True,
                        tracker="bytetrack.yaml",
                        conf=self.confidence,
                        iou=self.iou,
                        imgsz=self.imgsz,
                        device=self.device,
                        half=(self.device == "cuda"),
                        verbose=False,
                    )
                    inference_elapsed = max(time.perf_counter() - inference_start, 1e-6)
                    self.processing_fps = 1.0 / inference_elapsed
                    self.frames_processed += 1

                    annotated = frame.copy()
                    cv2.polylines(annotated, [area1], True, (255, 255, 255), 2)
                    cv2.polylines(annotated, [area2], True, (255, 255, 255), 2)
                    current_detections = []

                    result = results[0] if results else None
                    if result is not None and result.boxes is not None:
                        boxes = result.boxes
                        xyxy = boxes.xyxy.cpu().numpy()
                        cls_ids = boxes.cls.int().cpu().tolist()
                        confs = boxes.conf.cpu().tolist()
                        ids = (
                            boxes.id.int().cpu().tolist()
                            if boxes.id is not None
                            else [None] * len(xyxy)
                        )
                        names = self.model.names

                        for box, obj_id, cls_id, confidence in zip(xyxy, ids, cls_ids, confs):
                            cls_name = str(names[cls_id])
                            if cls_name.lower() not in {"car", "truck", "bus", "motorcycle"}:
                                continue
                            if confidence < self.confidence:
                                continue

                            x1, y1, x2, y2 = map(int, box)
                            x1, y1 = max(0, x1), max(0, y1)
                            x2, y2 = min(width - 1, x2), min(height - 1, y2)
                            if x2 <= x1 or y2 <= y1:
                                continue

                            # ByteTrack IDs are required for reliable wrong-lane state.
                            # If an ID is unavailable, still draw the detection rather
                            # than silently dropping the vehicle from the output.
                            cx, cy = (x1 + x2) // 2, y2
                            wrong = False
                            if obj_id is not None:
                                if cv2.pointPolygonTest(area1, (cx, cy), False) >= 0:
                                    self.wup[obj_id] = True
                                wrong = (
                                    obj_id in self.wup
                                    and cv2.pointPolygonTest(area2, (cx, cy), False) >= 0
                                )
                                self.vehicle_count.add(obj_id)

                                if wrong:
                                    self.wrongway.add(obj_id)
                                    crop = frame[y1:y2, x1:x2]
                                    if crop.size:
                                        self._anpr(obj_id, crop.copy(), frame)

                            color = (0, 0, 255) if wrong else (0, 255, 0)
                            id_text = f" | ID:{obj_id}" if obj_id is not None else ""
                            label = (
                                f"WRONG LANE{id_text} | {confidence:.2f}"
                                if wrong
                                else f"{cls_name}{id_text} | {confidence:.2f}"
                            )
                            current_detections.append((x1, y1, x2, y2, color, label))
                            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(
                                annotated,
                                label,
                                (x1, max(20, y1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.55,
                                color,
                                2,
                                cv2.LINE_AA,
                            )

                    last_detections = current_detections
                else:
                    # Keep every source frame in the output. Reuse the most recent
                    # tracked boxes when inference is intentionally skipped.
                    annotated = frame.copy()
                    cv2.polylines(annotated, [area1], True, (255, 255, 255), 2)
                    cv2.polylines(annotated, [area2], True, (255, 255, 255), 2)
                    for x1, y1, x2, y2, color, label in last_detections:
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(
                            annotated,
                            label,
                            (x1, max(20, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            color,
                            2,
                            cv2.LINE_AA,
                        )

                cv2.putText(
                    annotated,
                    f"Vehicles: {len(self.vehicle_count)} | Wrong Lane: {len(self.wrongway)} | Device: {self.device.upper()}",
                    (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                writer.write(annotated)

                if progress_callback and self.total_frames:
                    progress = min(99.0, round(frame_index / self.total_frames * 100, 1))
                    if progress != last_progress or frame_index % 10 == 0:
                        elapsed = max(time.perf_counter() - self.processing_started, 1e-6)
                        overall_fps = frame_index / elapsed
                        progress_callback({
                            "progress": progress,
                            "frames_processed": self.frames_processed,
                            "total_frames": self.total_frames,
                            "processing_fps": round(overall_fps, 2),
                            "stats": self.stats(),
                        })
                        last_progress = progress
        finally:
            cap.release()
            writer.release()

        if not output_path.exists() or output_path.stat().st_size == 0:
            raise RuntimeError("Processing finished but no valid output video was created.")
        return output_path

    def stats(self):
        elapsed = max(time.perf_counter() - self.processing_started, 1e-6) if self.processing_started else 0
        return {
            "total_vehicles": len(self.vehicle_count),
            "wrong_lane_vehicles": len(self.wrongway),
            "frames_processed": self.frames_processed,
            "total_frames": self.total_frames,
            "processing_fps": round(self.frames_processed / elapsed, 2) if self.frames_processed else 0,
            "video_fps": round(self.video_fps, 2),
            "device": self.device.upper(),
        }

    def report_text(self):
        if not self.report:
            return "No ANPR/plate-region detections were recorded.\n"
        return "\n".join(self.report) + "\n"
