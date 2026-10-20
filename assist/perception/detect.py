"""YOLO-World open-vocab detection + depth lift to meters/bearing."""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from ..capture import median_depth_in_box
from ..debuglog import log, log_exc

DEFAULT_CLASSES = [
    "person",
    "car",
    "bus",
    "truck",
    "traffic light",
    "door",
    "chair",
    "table",
    "stairs",
    "pole",
    "wall",
    "sofa",
    "plant",
    "bottle",
    "cup",
    "laptop",
    "bag",
]


# One YOLO pass on the GPU at a time. The live detector and the on-demand
# finder each have their own model; running both on MPS at once is not safe
# to rely on. The finder holds this for its whole pass, which also pauses
# live labeling (the detector skips instead of waiting).
YOLO_LOCK = threading.Lock()


def iter_boxes(results) -> List[tuple]:
    """(x1, y1, x2, y2, class_name, score) for each box of an Ultralytics result."""
    if not results:
        return []
    r0 = results[0]
    if r0.boxes is None or len(r0.boxes) == 0:
        return []
    names = r0.names
    out = []
    for box in r0.boxes:
        x1, y1, x2, y2 = map(float, box.xyxy[0].cpu().numpy())
        cls_id = int(box.cls[0].item())
        score = float(box.conf[0].item())
        label = (
            names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(names[cls_id])
        )
        out.append((x1, y1, x2, y2, label, score))
    return out


@dataclass
class DetectedObject:
    label: str
    dist_m: float
    bearing: str
    conf_depth: int
    box: tuple  # x1,y1,x2,y2
    score: float

    def as_dict(self) -> dict:
        return {
            "label": self.label,
            "dist_m": round(self.dist_m, 2),
            "bearing": self.bearing,
            "conf_depth": self.conf_depth,
            "score": round(self.score, 2),
        }


def bearing_from_cx(cx: float, width: float) -> str:
    """Map horizontal center of box to side label."""
    r = cx / max(width, 1)
    if r < 0.28:
        return "left"
    if r < 0.40:
        return "center-left"
    if r < 0.60:
        return "center"
    if r < 0.72:
        return "center-right"
    return "right"


class DetectionHold:
    """Keep last detections for a few empty frames (YOLO every-N gap)."""

    def __init__(self, ttl_frames: int = 3):
        self.ttl_frames = ttl_frames
        self._last: List[DetectedObject] = []
        self._age = 0

    def update(self, objects: List[DetectedObject]) -> List[DetectedObject]:
        if objects:
            self._last = objects
            self._age = 0
            return objects
        self._age += 1
        if self._age <= self.ttl_frames and self._last:
            return list(self._last)
        self._last = []
        return []


def yolo_device() -> str:
    """YOLO_DEVICE env (cpu / mps / auto). Auto picks the Apple GPU if present.

    On CPU a YOLO-World pass is ~200 ms and pins ~170% CPU on an M2 — on a
    fanless MacBook Air that throttles the whole app within minutes. On MPS
    the same pass is ~25 ms at ~1/3 of the CPU.
    """
    want = (os.getenv("YOLO_DEVICE") or "auto").strip().lower()
    if want != "auto":
        return want
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


class ObjectDetector:
    """Lazy-loads Ultralytics YOLO-World. Safe to construct without GPU."""

    def __init__(
        self,
        classes: Optional[List[str]] = None,
        conf: float = 0.25,
        model_name: str = "yolov8s-worldv2.pt",
    ):
        self.classes = classes or DEFAULT_CLASSES
        self.conf = conf
        self.model_name = model_name
        self._model = None
        self._failed = False
        self.device = yolo_device()

    def _ensure_model(self):
        if self._model is not None or self._failed:
            return
        try:
            from ultralytics import YOLO

            self._model = YOLO(self.model_name)
            # Set open-vocab vocabulary
            self._model.set_classes(self.classes)
            print(f"YOLO-World loaded: {self.model_name}, classes={self.classes}")
            log("detect", "YOLO-World loaded", model=self.model_name, device=self.device)
        except Exception as e:
            print(f"YOLO-World unavailable ({e}). Object detection disabled.")
            log_exc("detect", "YOLO-World unavailable", e)
            self._failed = True

    def warmup(self, shape: tuple = (1920, 1440, 3)) -> None:
        """Load the model and run one pass so the first real frame is not slow.

        The first MPS pass compiles kernels (seconds); doing it at startup keeps
        that hitch out of the live demo.
        """
        dummy = np.zeros(shape, dtype=np.uint8)
        depth = np.ones(shape[:2], dtype=np.float32)
        conf = np.full(shape[:2], 2, dtype=np.uint8)
        self.detect(dummy, depth, conf)

    def _predict(self, rgb_bgr: np.ndarray):
        try:
            return self._model.predict(
                rgb_bgr, conf=self.conf, verbose=False, imgsz=640, device=self.device
            )
        except Exception as exc:
            if self.device == "cpu":
                raise
            # A GPU backend hiccup must not cost us detection for the whole
            # session: drop to CPU once and keep going.
            log_exc("detect", f"YOLO on {self.device} failed, falling back to cpu", exc)
            self.device = "cpu"
            return self._model.predict(
                rgb_bgr, conf=self.conf, verbose=False, imgsz=640, device=self.device
            )

    def detect(
        self,
        rgb_bgr: np.ndarray,
        depth: np.ndarray,
        conf_map: np.ndarray,
        block: bool = True,
    ) -> Optional[List[DetectedObject]]:
        """Label one frame. ``block=False`` returns None at once if another
        YOLO pass (the finder) holds YOLO_LOCK."""
        if not YOLO_LOCK.acquire(blocking=block):
            return None
        try:
            self._ensure_model()
            if self._model is None:
                return []
            try:
                boxes = iter_boxes(self._predict(rgb_bgr))
            except Exception as e:
                log_exc("detect", "YOLO predict failed", e)
                return []
        finally:
            YOLO_LOCK.release()

        w = rgb_bgr.shape[1]
        out: List[DetectedObject] = []
        for x1, y1, x2, y2, label, score in boxes:
            z, cmed = median_depth_in_box(depth, conf_map, x1, y1, x2, y2)
            if z is None:
                continue
            out.append(
                DetectedObject(
                    label=label,
                    dist_m=z,
                    bearing=bearing_from_cx(0.5 * (x1 + x2), w),
                    conf_depth=cmed,
                    box=(int(x1), int(y1), int(x2), int(y2)),
                    score=score,
                )
            )

        # Keep more detections; no class-type filtering. LLM ranks importance.
        out.sort(key=lambda o: o.dist_m)
        return out[:15]
