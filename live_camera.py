"""Live IP-camera detection service for the Wrong Lane application.

The browser receives an MJPEG stream from Flask. The server opens the camera
URL, runs YOLO/ByteTrack, annotates each frame, and publishes the latest JPEG.
Supported sources are http://, https:// and rtsp:// URLs (and numeric webcam
indices for a camera attached directly to the server).
"""
from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from urllib.parse import urlparse

import cv2

from hubconfCustom import VideoDetector

logger = logging.getLogger("wrong_lane.live")


class LiveCameraService:
    def __init__(self, base_dir: Path, device: str):
        self.base_dir = Path(base_dir)
        self.device = device
        self.lock = threading.RLock()
        self.condition = threading.Condition(self.lock)
        self.thread = None
        self.stop_event = threading.Event()
        self.latest_jpeg = None
        self.sequence = 0
        self.source = None
        self.status = "stopped"
        self.error = None
        self.stats = {
            "frames": 0,
            "fps": 0.0,
            "vehicles": 0,
            "wrong_lane_vehicles": 0,
            "device": device.upper(),
            "source": None,
            "width": 0,
            "height": 0,
        }

    @staticmethod
    def validate_source(source: str) -> str:
        source = (source or "").strip()
        if not source:
            raise ValueError("Please enter an IP camera URL.")
        if source.isdigit():
            return source
        parsed = urlparse(source)
        if parsed.scheme.lower() not in {"rtsp", "http", "https"} or not parsed.netloc:
            raise ValueError("Camera URL must use rtsp://, http:// or https://.")
        return source

    def start(self, source: str, confidence=0.50, iou=0.45, imgsz=640, frame_interval=1):
        source = self.validate_source(source)
        with self.lock:
            if self.thread and self.thread.is_alive():
                raise RuntimeError("A live camera is already running. Stop it first.")
            self.stop_event.clear()
            self.latest_jpeg = None
            self.sequence = 0
            self.source = source
            self.error = None
            self.status = "starting"
            self.stats = {
                "frames": 0, "fps": 0.0, "vehicles": 0,
                "wrong_lane_vehicles": 0, "device": self.device.upper(),
                "source": source, "width": 0, "height": 0,
            }
            self.thread = threading.Thread(
                target=self._worker,
                args=(source, float(confidence), float(iou), int(imgsz), max(1, int(frame_interval))),
                daemon=True,
                name="live-camera",
            )
            self.thread.start()

    def stop(self):
        self.stop_event.set()
        with self.condition:
            self.status = "stopping"
            self.condition.notify_all()
        thread = self.thread
        if thread and thread.is_alive():
            thread.join(timeout=5)
        with self.lock:
            self.status = "stopped"
            self.thread = None
            self.condition.notify_all()

    @staticmethod
    def _safe_source(source):
        if not source:
            return source
        try:
            parsed = urlparse(source)
            if parsed.scheme in {"rtsp", "http", "https"} and parsed.hostname:
                host = parsed.hostname
                if parsed.port:
                    host = f"{host}:{parsed.port}"
                return f"{parsed.scheme}://{host}{parsed.path or ''}"
        except Exception:
            pass
        return source

    def snapshot(self):
        with self.lock:
            return {
                "status": self.status,
                "source": self._safe_source(self.source),
                "error": self.error,
                "sequence": self.sequence,
                "stats": dict(self.stats),
            }

    def wait_frame(self, last_sequence=0, timeout=5.0):
        deadline = time.monotonic() + timeout
        with self.condition:
            while self.sequence <= last_sequence and not self.stop_event.is_set():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self.condition.wait(remaining)
            return self.sequence, self.latest_jpeg, self.status

    def _open_capture(self, source):
        capture_source = int(source) if source.isdigit() else source
        cap = cv2.VideoCapture(capture_source)
        # Reduce latency for network streams when the backend supports it.
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        if not cap.isOpened():
            cap.release()
            raise RuntimeError("Could not connect to the camera. Check the IP, URL, credentials and network reachability from the server.")
        return cap

    def _worker(self, source, confidence, iou, imgsz, frame_interval):
        cap = None
        detector = None
        try:
            cap = self._open_capture(source)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            with self.lock:
                self.status = "running"
                self.stats.update(width=width, height=height)
                self.condition.notify_all()

            detector = VideoDetector(
                model_path=self.base_dir / "weights" / "yolo11n.pt",
                anpr_model_path=self.base_dir / "weights" / "anpr_car.pt",
                device=self.device,
                confidence=confidence,
                iou=iou,
                imgsz=imgsz,
                frame_interval=frame_interval,
                frames_dir=self.base_dir / "frames",
            )
            frame_index = 0
            started = time.perf_counter()
            area1 = area2 = None

            while not self.stop_event.is_set():
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError("The camera stream stopped or returned an unreadable frame.")
                frame_index += 1
                if frame_interval > 1 and (frame_index - 1) % frame_interval != 0:
                    continue

                h, w = frame.shape[:2]
                if area1 is None or detector.stats().get("total_frames", 0) == 0:
                    area1, area2 = detector._lane_polygons(w, h)

                results = detector.model.track(
                    source=frame,
                    persist=True,
                    tracker="bytetrack.yaml",
                    conf=confidence,
                    iou=iou,
                    imgsz=imgsz,
                    device=self.device,
                    half=(self.device == "cuda"),
                    verbose=False,
                )
                annotated = frame.copy()
                cv2.polylines(annotated, [area1], True, (255, 255, 255), 2)
                cv2.polylines(annotated, [area2], True, (255, 255, 255), 2)

                result = results[0] if results else None
                if result is not None and result.boxes is not None and result.boxes.id is not None:
                    boxes = result.boxes
                    ids = boxes.id.int().cpu().tolist()
                    xyxy = boxes.xyxy.cpu().numpy()
                    cls_ids = boxes.cls.int().cpu().tolist()
                    confs = boxes.conf.cpu().tolist()
                    names = detector.model.names
                    for box, obj_id, cls_id, score in zip(xyxy, ids, cls_ids, confs):
                        cls_name = str(names[cls_id])
                        if cls_name.lower() not in {"car", "truck", "bus", "motorcycle"} or score < confidence:
                            continue
                        x1, y1, x2, y2 = map(int, box)
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w - 1, x2), min(h - 1, y2)
                        if x2 <= x1 or y2 <= y1:
                            continue
                        cx, cy = (x1 + x2) // 2, y2
                        if cv2.pointPolygonTest(area1, (cx, cy), False) >= 0:
                            detector.wup[obj_id] = True
                        wrong = obj_id in detector.wup and cv2.pointPolygonTest(area2, (cx, cy), False) >= 0
                        detector.vehicle_count.add(obj_id)
                        if wrong:
                            detector.wrongway.add(obj_id)
                            color = (0, 0, 255)
                            label = f"WRONG LANE | ID:{obj_id} | {score:.2f}"
                        else:
                            color = (0, 255, 0)
                            label = f"{cls_name} | ID:{obj_id} | {score:.2f}"
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(annotated, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

                elapsed = max(time.perf_counter() - started, 1e-6)
                fps = frame_index / elapsed
                cv2.putText(
                    annotated,
                    f"LIVE | Vehicles: {len(detector.vehicle_count)} | Wrong Lane: {len(detector.wrongway)} | {self.device.upper()} | {fps:.1f} FPS",
                    (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 255), 2, cv2.LINE_AA,
                )
                ok, encoded = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
                if not ok:
                    continue
                with self.condition:
                    self.latest_jpeg = encoded.tobytes()
                    self.sequence += 1
                    self.stats.update(
                        frames=frame_index,
                        fps=round(fps, 2),
                        vehicles=len(detector.vehicle_count),
                        wrong_lane_vehicles=len(detector.wrongway),
                    )
                    self.condition.notify_all()
        except Exception as exc:
            logger.exception("Live camera failed")
            with self.condition:
                self.error = str(exc)
                self.status = "error"
                self.condition.notify_all()
        finally:
            if cap is not None:
                cap.release()
            with self.lock:
                if self.status != "error":
                    self.status = "stopped"
                self.condition.notify_all()

    def mjpeg(self):
        last = 0
        while True:
            seq, jpeg, status = self.wait_frame(last, timeout=5)
            if jpeg:
                last = seq
                yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                       + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n")
            elif status in {"stopped", "error"}:
                break
