from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

import config_rt_dlc_live as config
from dlclive import DLCLive


# =========================
# Logging
# =========================
def setup_logger() -> logging.Logger:
    logger = logging.getLogger("rt_dlc_live")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    return logger


# =========================
# Frame sources
# =========================
@dataclass
class FramePacket:
    frame_id: int
    frame: np.ndarray
    capture_ts: float


class FrameSource(ABC):
    @abstractmethod
    def open(self) -> None:
        ...

    @abstractmethod
    def read(self) -> tuple[bool, Optional[FramePacket]]:
        ...

    @abstractmethod
    def release(self) -> None:
        ...


class CameraSource(FrameSource):
    def __init__(self, camera_index: int) -> None:
        self.camera_index = camera_index
        self.cap: cv2.VideoCapture | None = None
        self.frame_id = 0

    def open(self) -> None:
        self.cap = cv2.VideoCapture(self.camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_H)
        self.cap.set(cv2.CAP_PROP_FPS, config.TARGET_VIDEO_FPS)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera index {self.camera_index}")

    def read(self) -> tuple[bool, Optional[FramePacket]]:
        if self.cap is None:
            raise RuntimeError("CameraSource is not opened.")

        ret, frame = self.cap.read()
        if not ret:
            return False, None

        self.frame_id += 1
        return True, FramePacket(self.frame_id, frame, time.time())

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None


class VideoFileSource(FrameSource):
    def __init__(self, video_path: Path, target_fps: float | None, skip_if_behind: bool) -> None:
        self.video_path = video_path
        self.target_fps = target_fps
        self.skip_if_behind = skip_if_behind
        self.cap: cv2.VideoCapture | None = None
        self.frame_id = 0
        self.start_perf: float | None = None
        self.frame_interval = (1.0 / target_fps) if target_fps and target_fps > 0 else None

    def open(self) -> None:
        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video file: {self.video_path}")
        self.start_perf = time.perf_counter()

    def read(self) -> tuple[bool, Optional[FramePacket]]:
        if self.cap is None:
            raise RuntimeError("VideoFileSource is not opened.")

        ret, frame = self.cap.read()
        if not ret:
            return False, None

        self.frame_id += 1

        if self.frame_interval is not None and self.start_perf is not None:
            expected_elapsed = self.frame_id * self.frame_interval
            actual_elapsed = time.perf_counter() - self.start_perf
            lag = actual_elapsed - expected_elapsed

            if lag < 0:
                time.sleep(-lag)
            elif lag > self.frame_interval and self.skip_if_behind:
                # drop one extra frame to catch up
                _ = self.cap.read()

        return True, FramePacket(self.frame_id, frame, time.time())

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None


def build_frame_source() -> FrameSource:
    if config.USE_VIDEO_FILE:
        return VideoFileSource(
            video_path=Path(config.VIDEO_FILE_PATH),
            target_fps=config.VIDEO_TARGET_FPS,
            skip_if_behind=config.VIDEO_SKIP_IF_BEHIND,
        )
    return CameraSource(config.CAM_INDEX)


# =========================
# Geometry / preprocess
# =========================
def resolve_roi(frame: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    """
    Returns:
        cropped_frame, (x_offset, y_offset)
    """
    if not config.USE_ROI:
        return frame, (0, 0)

    x1, y1, x2, y2 = config.ROI
    h, w = frame.shape[:2]

    if 0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h:
        return frame[y1:y2, x1:x2], (x1, y1)

    roi_w = max(0, x2 - x1)
    roi_h = max(0, y2 - y1)

    # if frame already equals ROI size, accept as-is
    if w == roi_w and h == roi_h:
        return frame, (0, 0)

    raise ValueError(
        f"Invalid ROI={config.ROI} for frame size {(w, h)}. "
        "Either disable USE_ROI or provide ROI in the source-frame coordinates."
    )


def resize_for_infer(frame: np.ndarray) -> np.ndarray:
    return cv2.resize(frame, (config.INFER_W, config.INFER_H), interpolation=cv2.INTER_LINEAR)


def map_points_from_infer_to_display(
    points: dict[str, dict[str, float | None]],
    roi_shape: tuple[int, int],
    infer_shape: tuple[int, int],
    roi_offset: tuple[int, int],
) -> dict[str, dict[str, float | None]]:
    roi_h, roi_w = roi_shape
    infer_h, infer_w = infer_shape
    x_off, y_off = roi_offset

    sx = roi_w / float(infer_w)
    sy = roi_h / float(infer_h)

    mapped: dict[str, dict[str, float | None]] = {}
    for name, p in points.items():
        x = p["x"]
        y = p["y"]
        l = p["likelihood"]

        if x is None or y is None:
            mapped[name] = {"x": None, "y": None, "likelihood": l}
            continue

        mapped[name] = {
            "x": float(x * sx + x_off),
            "y": float(y * sy + y_off),
            "likelihood": l,
        }
    return mapped


# =========================
# DLC-Live wrapper
# =========================
class DLCLivePredictor:
    def __init__(self, model_path: str, model_type: str, body_parts: list[str]) -> None:
        self.model_path = model_path
        self.model_type = model_type
        self.body_parts = body_parts
        self.live = DLCLive(self.model_path, model_type=self.model_type)
        self.initialized = False

    def init_on_frame(self, frame_bgr: np.ndarray) -> None:
        if self.initialized:
            return

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        _ = self.live.init_inference(frame_rgb)
        self.initialized = True

    def predict_frame(self, frame_bgr: np.ndarray) -> dict[str, dict[str, float | None]]:
        if not self.initialized:
            self.init_on_frame(frame_bgr)

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pose = self.live.get_pose(frame_rgb)

        pose = np.asarray(pose)
        if pose.ndim != 2 or pose.shape[1] < 3:
            raise RuntimeError(f"Unexpected DLCLive pose shape: {pose.shape}")

        n = min(len(self.body_parts), pose.shape[0])
        points: dict[str, dict[str, float | None]] = {}

        for i, name in enumerate(self.body_parts):
            if i >= n:
                points[name] = {"x": None, "y": None, "likelihood": None}
                continue

            x, y, l = pose[i, 0], pose[i, 1], pose[i, 2]
            points[name] = {
                "x": float(x),
                "y": float(y),
                "likelihood": float(l),
            }

        return points


# =========================
# Overlay
# =========================
def draw_overlay(
    frame: np.ndarray,
    points: dict[str, dict[str, float | None]],
    fps_cam: float,
    fps_dlc: float,
) -> np.ndarray:
    out = frame.copy()

    for name, p in points.items():
        x, y, l = p["x"], p["y"], p["likelihood"]
        if x is None or y is None or l is None:
            continue
        if l < config.CONF_THRESH_DRAW:
            continue

        cv2.circle(out, (int(x), int(y)), 4, (0, 255, 0), -1)

        if config.DRAW_NAMES:
            label = name
            if config.DRAW_CONF:
                label += f" {l:.2f}"
            cv2.putText(
                out,
                label,
                (int(x) + 6, int(y) - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 255, 0),
                1,
            )

    y0 = 22
    cv2.putText(
        out,
        f"CAM FPS: {fps_cam:.1f} | DLC FPS: {fps_dlc:.1f}",
        (10, y0),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
    )
    return out


# =========================
# Main
# =========================
def validate_config() -> None:
    if config.INFER_W <= 0 or config.INFER_H <= 0:
        raise ValueError("INFER_W and INFER_H must be positive.")
    if config.SHOW_SCALE <= 0:
        raise ValueError("SHOW_SCALE must be positive.")
    if config.MODEL_TYPE not in {"pytorch", "base"}:
        # "base" here is only left as a fallback string, but your case is pytorch
        raise ValueError("MODEL_TYPE should be 'pytorch' for your exported model.")
    if not Path(config.MODEL_PATH).exists():
        raise FileNotFoundError(f"MODEL_PATH does not exist: {config.MODEL_PATH}")
    if config.USE_VIDEO_FILE and not Path(config.VIDEO_FILE_PATH).exists():
        raise FileNotFoundError(f"VIDEO_FILE_PATH does not exist: {config.VIDEO_FILE_PATH}")


def main() -> None:
    validate_config()
    logger = setup_logger()

    logger.info("Starting rt_dlc_live.py")
    logger.info("MODEL_PATH=%s", config.MODEL_PATH)
    logger.info("MODEL_TYPE=%s", config.MODEL_TYPE)

    source = build_frame_source()
    source.open()
    logger.info("Source opened: %s", type(source).__name__)

    predictor = DLCLivePredictor(
        model_path=config.MODEL_PATH,
        model_type=config.MODEL_TYPE,
        body_parts=config.BODY_PARTS,
    )

    prev_cam_ts: Optional[float] = None
    prev_infer_end_ts: Optional[float] = None
    fps_cam = 0.0
    fps_dlc = 0.0

    total_visible = 0
    total_points = 0

    while True:
        ret, packet = source.read()
        if not ret or packet is None:
            logger.info("Source ended.")
            break

        frame = packet.frame
        frame_id = packet.frame_id
        capture_ts = packet.capture_ts

        if prev_cam_ts is not None:
            dt = capture_ts - prev_cam_ts
            if dt > 0:
                fps_cam = 1.0 / dt
        prev_cam_ts = capture_ts

        roi_frame, roi_offset = resolve_roi(frame)
        infer_frame = resize_for_infer(roi_frame)

        t0 = time.perf_counter()
        points_infer = predictor.predict_frame(infer_frame)
        t1 = time.perf_counter()

        infer_time_ms = (t1 - t0) * 1000.0
        infer_end_ts = time.time()

        if prev_infer_end_ts is not None:
            dt_inf = infer_end_ts - prev_infer_end_ts
            if dt_inf > 0:
                fps_dlc = 1.0 / dt_inf
        prev_infer_end_ts = infer_end_ts

        if config.SHOW_FULL_FRAME:
            draw_points = map_points_from_infer_to_display(
                points_infer,
                roi_shape=(roi_frame.shape[0], roi_frame.shape[1]),
                infer_shape=(infer_frame.shape[0], infer_frame.shape[1]),
                roi_offset=roi_offset,
            )
            display = frame.copy()
        else:
            draw_points = points_infer
            display = infer_frame.copy()

        visible_now = 0
        for p in draw_points.values():
            l = p["likelihood"]
            if p["x"] is not None and p["y"] is not None and l is not None and l >= config.CONF_THRESH_DRAW:
                visible_now += 1

        total_visible += visible_now
        total_points += len(draw_points)

        display = draw_overlay(display, draw_points, fps_cam, fps_dlc)

        if config.SHOW_SCALE != 1.0:
            display = cv2.resize(
                display,
                None,
                fx=config.SHOW_SCALE,
                fy=config.SHOW_SCALE,
                interpolation=cv2.INTER_AREA,
            )

        cv2.imshow(config.WINDOW_NAME, display)
        key = cv2.waitKey(1) & 0xFF

        if frame_id % max(1, config.LOG_EVERY_N_FRAMES) == 0:
            visible_pct = 100.0 * total_visible / max(1, total_points)
            logger.info(
                "frame_id=%d cam_fps=%.1f dlc_fps=%.1f infer_time=%.2fms visible=%.1f%%",
                frame_id,
                fps_cam,
                fps_dlc,
                infer_time_ms,
                visible_pct,
            )

        if key in (27, ord("q")):
            logger.info("Exit requested by user.")
            break

    source.release()
    cv2.destroyAllWindows()
    logger.info("Finished.")


if __name__ == "__main__":
    main()