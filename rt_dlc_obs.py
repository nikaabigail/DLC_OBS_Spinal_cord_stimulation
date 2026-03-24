from __future__ import annotations

import csv
import logging
import math
import logging
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Event, Lock, Thread
from typing import Optional

import cv2
import numpy as np
import torch
import yaml

import config_rt_dlc as config


# =========================
# Frame source abstraction
# =========================
@dataclass
class FramePacket:
    frame_id: int
    frame: np.ndarray
    capture_ts: float


@dataclass
class PredictionPacket:
    frame_id: int
    infer_start_ts: float
    infer_end_ts: float
    points: dict[str, dict[str, float | None]]
    roi_shape: tuple[int, int]
    infer_shape: tuple[int, int]
    roi_offset: tuple[int, int]


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
        return True, FramePacket(frame_id=self.frame_id, frame=frame, capture_ts=time.time())

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
        self.start_ts: float | None = None
        self.frame_interval = (1.0 / target_fps) if target_fps and target_fps > 0 else None

    def open(self) -> None:
        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video file: {self.video_path}")
        self.start_ts = time.perf_counter()

    def read(self) -> tuple[bool, Optional[FramePacket]]:
        if self.cap is None:
            raise RuntimeError("VideoFileSource is not opened.")

        ret, frame = self.cap.read()
        if not ret:
            return False, None

        self.frame_id += 1

        if self.frame_interval is not None and self.start_ts is not None:
            expected_elapsed = self.frame_id * self.frame_interval
            actual_elapsed = time.perf_counter() - self.start_ts
            lag = actual_elapsed - expected_elapsed

            if lag < 0:
                time.sleep(-lag)
            elif lag > self.frame_interval and self.skip_if_behind:
                # drop one stale frame to catch up
                _ = self.cap.read()

        return True, FramePacket(frame_id=self.frame_id, frame=frame, capture_ts=time.time())

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None


def build_frame_source() -> FrameSource:
    if getattr(config, "USE_VIDEO_FILE", False):
        return VideoFileSource(
            video_path=Path(config.VIDEO_FILE_PATH),
            target_fps=getattr(config, "VIDEO_TARGET_FPS", None),
            skip_if_behind=getattr(config, "VIDEO_SKIP_IF_BEHIND", True),
        )
    return CameraSource(camera_index=config.CAM_INDEX)


# =========================
# Geometry / helper utils
# =========================
def crop_roi(frame: np.ndarray) -> np.ndarray:
    if not config.USE_ROI:
        return frame
    x1, y1, x2, y2 = config.ROI
    h, w = frame.shape[:2]
    if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
        raise ValueError(
            f"Invalid ROI={config.ROI} for frame size {(w, h)}. "
            "Expected 0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height."
        )
    return frame[y1:y2, x1:x2]


def detect_content_roi(
    frame: np.ndarray,
    black_thresh: int = 12,
    min_row_fill: float = 0.08,
    min_col_fill: float = 0.03,
) -> tuple[int, int, int, int] | None:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask = gray > black_thresh

    row_fill = mask.mean(axis=1)
    col_fill = mask.mean(axis=0)

    row_idx = np.where(row_fill >= min_row_fill)[0]
    col_idx = np.where(col_fill >= min_col_fill)[0]
    if len(row_idx) == 0 or len(col_idx) == 0:
        return None

    y1, y2 = int(row_idx.min()), int(row_idx.max()) + 1
    x1, x2 = int(col_idx.min()), int(col_idx.max()) + 1

    if x2 - x1 < 32 or y2 - y1 < 32:
        return None
    return x1, y1, x2, y2


def resize_for_infer(frame: np.ndarray) -> np.ndarray:
    return cv2.resize(frame, (config.INFER_W, config.INFER_H), interpolation=cv2.INTER_LINEAR)


def safe_angle_deg(
    a: tuple[float, float],
    b: tuple[float, float],
    c: tuple[float, float],
) -> Optional[float]:
    ba = np.array(a, dtype=np.float32) - np.array(b, dtype=np.float32)
    bc = np.array(c, dtype=np.float32) - np.array(b, dtype=np.float32)

    n1 = np.linalg.norm(ba)
    n2 = np.linalg.norm(bc)
    if n1 < 1e-6 or n2 < 1e-6:
        return None

    cos_val = float(np.dot(ba, bc) / (n1 * n2))
    cos_val = float(np.clip(cos_val, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_val)))


def dist2d(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    return float(math.hypot(p1[0] - p2[0], p1[1] - p2[1]))


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


def validate_runtime_config() -> None:
    if config.INFER_W <= 0 or config.INFER_H <= 0:
        raise ValueError("INFER_W and INFER_H must be positive.")
    if config.SHOW_SCALE <= 0:
        raise ValueError("SHOW_SCALE must be positive.")
    if config.DESPIKE_THRESHOLD_PX <= 0:
        raise ValueError("DESPIKE_THRESHOLD_PX must be positive.")
    if config.MAX_HOLD_FRAMES < 0:
        raise ValueError("MAX_HOLD_FRAMES must be >= 0.")
    if config.MEDIAN_WINDOW <= 0:
        raise ValueError("MEDIAN_WINDOW must be positive.")
    if not (0.0 <= config.CONF_THRESH_USE <= 1.0):
        raise ValueError("CONF_THRESH_USE must be in [0, 1].")
    if not (0.0 <= config.CONF_THRESH_DRAW <= 1.0):
        raise ValueError("CONF_THRESH_DRAW must be in [0, 1].")
    if getattr(config, "USE_ROI", False):
        roi = getattr(config, "ROI", None)
        if not (isinstance(roi, tuple) and len(roi) == 4 and all(isinstance(v, int) for v in roi)):
            raise ValueError("ROI must be a tuple of 4 integers: (x1, y1, x2, y2).")
    if getattr(config, "INFER_QUEUE_MAXSIZE", 2) <= 0:
        raise ValueError("INFER_QUEUE_MAXSIZE must be positive.")
    if getattr(config, "DISPLAY_DELAY_MS", 0) < 0:
        raise ValueError("DISPLAY_DELAY_MS must be >= 0.")
    if getattr(config, "MAX_RESULT_HISTORY", 10) <= 0:
        raise ValueError("MAX_RESULT_HISTORY must be positive.")
    if getattr(config, "INFER_EVERY_N_FRAMES", 1) <= 0:
        raise ValueError("INFER_EVERY_N_FRAMES must be positive.")
    if getattr(config, "TARGET_INFER_FPS", 1.0) <= 0:
        raise ValueError("TARGET_INFER_FPS must be positive.")


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("rt_dlc")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    try:
        fh = logging.FileHandler(config.LOG_PATH, encoding="utf-8")
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    except Exception as exc:
        logger.warning("Cannot open log file %s: %s", getattr(config, "LOG_PATH", ""), exc)

    return logger


# =========================
# Filtering
# =========================
@dataclass
class PointState:
    last_good_xy: Optional[tuple[float, float]] = None
    last_good_frame_idx: Optional[int] = None
    x_hist: deque | None = None
    y_hist: deque | None = None

    def __post_init__(self) -> None:
        if self.x_hist is None:
            self.x_hist = deque(maxlen=config.MEDIAN_WINDOW)
        if self.y_hist is None:
            self.y_hist = deque(maxlen=config.MEDIAN_WINDOW)


class OnlinePointFilter:
    def __init__(self) -> None:
        self.states: dict[str, PointState] = defaultdict(PointState)

    def process_point(
        self,
        name: str,
        x: Optional[float],
        y: Optional[float],
        likelihood: Optional[float],
        frame_idx: int,
    ) -> tuple[Optional[float], Optional[float], Optional[float]]:
        state = self.states[name]

        is_good = (
            x is not None
            and y is not None
            and likelihood is not None
            and ((not config.ENABLE_PCUTOFF) or likelihood >= config.CONF_THRESH_USE)
        )

        if is_good and config.ENABLE_DESPIKE:
            current_xy = (float(x), float(y))
            if state.last_good_xy is not None:
                jump = dist2d(current_xy, state.last_good_xy)
                if jump > config.DESPIKE_THRESHOLD_PX:
                    is_good = False

        if is_good:
            current_xy = (float(x), float(y))
            state.last_good_xy = current_xy
            state.last_good_frame_idx = frame_idx
            state.x_hist.append(current_xy[0])
            state.y_hist.append(current_xy[1])
            x_med = float(np.median(np.array(state.x_hist, dtype=np.float32)))
            y_med = float(np.median(np.array(state.y_hist, dtype=np.float32)))
            return x_med, y_med, float(likelihood)

        if config.ENABLE_HOLD and state.last_good_xy is not None and state.last_good_frame_idx is not None:
            gap = frame_idx - state.last_good_frame_idx
            if gap <= config.MAX_HOLD_FRAMES:
                hold_x, hold_y = state.last_good_xy
                state.x_hist.append(hold_x)
                state.y_hist.append(hold_y)
                x_med = float(np.median(np.array(state.x_hist, dtype=np.float32)))
                y_med = float(np.median(np.array(state.y_hist, dtype=np.float32)))
                return x_med, y_med, float(config.CONF_THRESH_USE + 0.01)

        return None, None, None


@dataclass
class InferJob:
    frame_idx: int
    infer_frame: np.ndarray
    roi_shape: tuple[int, int]
    infer_shape: tuple[int, int]
    roi_offset: tuple[int, int]


@dataclass
class InferResult:
    frame_idx: int
    points: dict[str, dict[str, float | None]]
    roi_shape: tuple[int, int]
    infer_shape: tuple[int, int]
    roi_offset: tuple[int, int]
    ts: float


# =========================
# DLC wrapper
# =========================
class DLCRealtimePredictor:
    def __init__(self) -> None:
        self.snapshot_path = Path(config.DLC_SNAPSHOT)
        self.pytorch_cfg_path = Path(config.DLC_PYTORCH_CFG)

        if not self.pytorch_cfg_path.exists():
            raise FileNotFoundError(f"PyTorch config not found: {self.pytorch_cfg_path}")
        if not self.snapshot_path.exists():
            raise FileNotFoundError(f"DLC snapshot not found: {self.snapshot_path}")

        self.runner = None
        self.model_cfg: dict | None = None
        self.all_bodyparts: list[str] = []
        self.bodypart_to_idx: dict[str, int] = {}
        self._init_predictor()

    def _init_predictor(self) -> None:
        from deeplabcut.pose_estimation_pytorch.apis import get_pose_inference_runner

        with open(self.pytorch_cfg_path, "r", encoding="utf-8") as f:
            self.model_cfg = yaml.safe_load(f)

        self.all_bodyparts = list(self.model_cfg["metadata"]["bodyparts"])
        self.bodypart_to_idx = {name: i for i, name in enumerate(self.all_bodyparts)}
        print("[INFO] Bodyparts order:", self.all_bodyparts)

        self.runner = get_pose_inference_runner(
            model_config=self.model_cfg,
            snapshot_path=str(self.snapshot_path),
            batch_size=1,
            device=config.DEVICE,
        )
        print("[INFO] DLC pose inference runner initialized.")

    @staticmethod
    def _normalize_poses(poses_raw: object) -> np.ndarray:
        poses = np.asarray(poses_raw)
        if poses.ndim == 3 and poses.shape[0] == 1:
            poses = poses[0]
        elif poses.ndim == 3 and poses.shape[1] == 1:
            poses = poses[:, 0, :]
        elif poses.ndim == 1 and poses.shape[0] >= 3:
            poses = poses.reshape(1, -1)

        if poses.ndim != 2 or poses.shape[1] < 3:
            raise RuntimeError(f"Unexpected poses shape: {poses.shape}. Expected [N,3] or [1,N,3].")
        return poses

    def predict_frame(self, frame_bgr: np.ndarray) -> dict[str, dict[str, float | None]]:
        if self.runner is None:
            raise RuntimeError("Runner is not initialized.")

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if self.runner.preprocessor is not None:
            inputs, context = self.runner.preprocessor(frame_rgb, {})
        else:
            inputs = torch.as_tensor(frame_rgb)
            context = {}

        model_kwargs = context.get("model_kwargs", {})
        with torch.inference_mode():
            raw = self.runner.predict(inputs, **model_kwargs)

        raw0 = raw[0] if isinstance(raw, (list, tuple)) and len(raw) > 0 else raw
        if not (isinstance(raw0, dict) and "bodypart" in raw0 and "poses" in raw0["bodypart"]):
            raise RuntimeError(f"Unexpected pred output format: type={type(raw0)} repr={repr(raw0)[:300]}")

        poses = self._normalize_poses(raw0["bodypart"]["poses"])
        points: dict[str, dict[str, float | None]] = {}

        for point_name in config.USE_POINTS:
            idx = self.bodypart_to_idx.get(point_name)
            if idx is None or idx >= len(poses):
                points[point_name] = {"x": None, "y": None, "likelihood": None}
                continue
            points[point_name] = {
                "x": float(poses[idx, 0]),
                "y": float(poses[idx, 1]),
                "likelihood": float(poses[idx, 2]),
            }
        return points

    @staticmethod
    def _normalize_poses(poses_raw: object) -> np.ndarray:
        poses = np.asarray(poses_raw)

        # Частый случай DLC: batch из 1 элемента -> (1, N, 3)
        if poses.ndim == 3 and poses.shape[0] == 1:
            poses = poses[0]
        # Реже: (N, 1, 3)
        elif poses.ndim == 3 and poses.shape[1] == 1:
            poses = poses[:, 0, :]
        # Совсем вырожденный случай: (3,) для одной точки
        elif poses.ndim == 1 and poses.shape[0] >= 3:
            poses = poses.reshape(1, -1)

        if poses.ndim != 2 or poses.shape[1] < 3:
            raise RuntimeError(
                f"Unexpected poses shape: {poses.shape}. Expected [N,3] or [1,N,3]."
            )

        return poses


# =========================
# Overlay
# =========================
def draw_overlay(
    frame: np.ndarray,
    points: dict[str, dict[str, float | None]],
    fps_cam: float,
    fps_dlc: float,
    skip_rate: float,
    hind_angle: Optional[float],
    processing_active: bool,
    motion_score: float,
) -> np.ndarray:
    out = frame.copy()

    if config.DRAW_POINTS:
        for point_name, p in points.items():
            x, y, l = p["x"], p["y"], p["likelihood"]
            if x is None or y is None or l is None:
                continue
            if l < config.CONF_THRESH_DRAW:
                continue
            cv2.circle(out, (int(x), int(y)), 4, (0, 255, 0), -1)
            if config.DRAW_NAMES:
                label = point_name + (f" {l:.2f}" if config.DRAW_CONF else "")
                cv2.putText(out, label, (int(x) + 5, int(y) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    y0 = 20
    cv2.putText(out, f"STATUS: {'ACTIVE' if processing_active else 'WAITING'}", (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    y0 += 25
    cv2.putText(out, f"MOTION: {motion_score:.3f}", (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)
    y0 += 25
    cv2.putText(out, f"CAM FPS: {fps_cam:.1f} | DLC FPS: {fps_dlc:.1f} | SKIP RATE: {skip_rate:.1f}%", (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    if config.DRAW_HIND_ANGLE and hind_angle is not None:
        cv2.putText(out, f"Hind angle: {hind_angle:.1f}", (10, y0 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    return out


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("rt_dlc")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    fh = logging.FileHandler(config.LOG_PATH, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def validate_runtime_config() -> None:
    if config.INFER_W <= 0 or config.INFER_H <= 0:
        raise ValueError("INFER_W and INFER_H must be positive.")
    if config.INFER_EVERY_N_FRAMES <= 0:
        raise ValueError("INFER_EVERY_N_FRAMES must be positive.")
    if config.TARGET_INFER_FPS <= 0:
        raise ValueError("TARGET_INFER_FPS must be positive.")
    if config.DISPLAY_BUFFER_MS < 0:
        raise ValueError("DISPLAY_BUFFER_MS must be >= 0.")
    if config.MAX_FRAME_BUFFER <= 0 or config.MAX_PRED_BUFFER <= 0:
        raise ValueError("MAX_FRAME_BUFFER and MAX_PRED_BUFFER must be positive.")
    if config.USE_VIDEO_FILE and not Path(config.VIDEO_FILE_PATH).exists():
        raise FileNotFoundError(f"VIDEO_FILE_PATH does not exist: {config.VIDEO_FILE_PATH}")


def main() -> None:
    validate_runtime_config()
    logger = setup_logger()
    source = build_frame_source()
    source.open()

    predictor = DLCRealtimePredictor()
    point_filter = OnlinePointFilter()

    stats = defaultdict(float)
    skip_reasons = defaultdict(int)
    likelihood_hist = [0, 0, 0, 0, 0]
    bodypart_lik_sum: dict[str, float] = defaultdict(float)
    bodypart_lik_cnt: dict[str, int] = defaultdict(int)

    prev_frame_gray: Optional[np.ndarray] = None
    prev_motion_gray: Optional[np.ndarray] = None
    prev_cam_ts: Optional[float] = None
    prev_infer_end_ts: Optional[float] = None
    prev_infer_ts_for_fps: Optional[float] = None
    fps_dlc = 0.0
    frame_buffer: deque[FramePacket] = deque(maxlen=config.MAX_FRAME_BUFFER)
    pred_buffer: deque[PredictionPacket] = deque(maxlen=config.MAX_PRED_BUFFER)

    csv_header_written = False

    logger.info("Pipeline started. source=%s roi=%s infer_size=(%s,%s)", type(source).__name__, config.ROI if config.USE_ROI else "full", config.INFER_W, config.INFER_H)

    while True:
        t_cap0 = time.perf_counter()
        ret, packet = source.read()
        t_cap1 = time.perf_counter()
        if not ret or packet is None:
            break

        frame = packet.frame
        frame_id = packet.frame_id
        capture_ts = packet.capture_ts
        frame_buffer.append(packet)
        stats["frames"] += 1
        stats["t_capture_ms"] += (t_cap1 - t_cap0) * 1000.0

        if prev_cam_ts is not None:
            dt = capture_ts - prev_cam_ts
            if dt > 0:
                stats["fps_cam"] = 1.0 / dt
        prev_cam_ts = capture_ts

        # preprocess
        t_pre0 = time.perf_counter()
        roi_offset = (config.ROI[0], config.ROI[1]) if config.USE_ROI else (0, 0)
        roi = crop_roi(frame)
        infer_frame = resize_for_infer(roi)
        t_pre1 = time.perf_counter()
        stats["t_pre_ms"] += (t_pre1 - t_pre0) * 1000.0

        # motion
        motion_score = 0.0
        gray_small = cv2.cvtColor(infer_frame, cv2.COLOR_BGR2GRAY)
        if prev_motion_gray is not None:
            motion_score = float(cv2.absdiff(gray_small, prev_motion_gray).mean())
        prev_motion_gray = gray_small

        # admission control
        run_dlc = True
        if frame_id % max(1, config.INFER_EVERY_N_FRAMES) != 0:
            run_dlc = False
            skip_reasons["skip_n"] += 1
        if run_dlc and config.SUPPRESS_LOW_MOTION and motion_score < config.LOW_MOTION_THRESHOLD:
            run_dlc = False
            skip_reasons["skip_motion"] += 1
        if run_dlc and config.SKIP_NEAR_DUPLICATE_FRAMES:
            if prev_frame_gray is not None:
                dup_score = float(cv2.absdiff(gray_small, prev_frame_gray).mean())
                if dup_score < config.DUPLICATE_FRAME_THRESHOLD:
                    run_dlc = False
                    skip_reasons["skip_duplicate"] += 1
            prev_frame_gray = gray_small

        if run_dlc and prev_infer_end_ts is not None:
            min_dt = 1.0 / max(1.0, config.TARGET_INFER_FPS)
            if (time.time() - prev_infer_end_ts) < min_dt:
                run_dlc = False
                skip_reasons["skip_fps"] += 1

        infer_start_ts = 0.0
        infer_end_ts = 0.0

        if run_dlc:
            infer_start_ts = time.time()
            t_inf0 = time.perf_counter()
            raw_points = predictor.predict_frame(infer_frame)
            t_inf1 = time.perf_counter()
            infer_end_ts = time.time()
            prev_infer_end_ts = infer_end_ts
            if prev_infer_ts_for_fps is not None:
                dt_inf = infer_end_ts - prev_infer_ts_for_fps
                if dt_inf > 0:
                    fps_dlc = 1.0 / dt_inf
            prev_infer_ts_for_fps = infer_end_ts
            stats["infer_frames"] += 1
            stats["t_infer_ms"] += (t_inf1 - t_inf0) * 1000.0

            t_post0 = time.perf_counter()
            filtered_points: dict[str, dict[str, float | None]] = {}
            for name, p in raw_points.items():
                raw_l = p["likelihood"]
                if raw_l is not None:
                    bodypart_lik_sum[name] += raw_l
                    bodypart_lik_cnt[name] += 1
                    if raw_l < 0.2:
                        likelihood_hist[0] += 1
                    elif raw_l < 0.4:
                        likelihood_hist[1] += 1
                    elif raw_l < 0.6:
                        likelihood_hist[2] += 1
                    elif raw_l < 0.8:
                        likelihood_hist[3] += 1
                    else:
                        likelihood_hist[4] += 1
                    if raw_l >= config.CONF_THRESH_DRAW:
                        stats["raw_visible"] += 1

                x, y, l = point_filter.process_point(name, p["x"], p["y"], p["likelihood"], frame_id)
                filtered_points[name] = {"x": x, "y": y, "likelihood": l}
                if l is not None and l >= config.CONF_THRESH_DRAW and x is not None and y is not None:
                    stats["filtered_visible"] += 1
                stats["total_points"] += 1

            t_post1 = time.perf_counter()
            stats["t_post_ms"] += (t_post1 - t_post0) * 1000.0
            pred_buffer.append(
                PredictionPacket(
                    frame_id=frame_id,
                    infer_start_ts=infer_start_ts,
                    infer_end_ts=infer_end_ts,
                    points=filtered_points,
                    roi_shape=(roi.shape[0], roi.shape[1]),
                    infer_shape=(infer_frame.shape[0], infer_frame.shape[1]),
                    roi_offset=roi_offset,
                )
            )
        else:
            skip_reasons["skip_total"] += 1

        display_ts = time.time()
        target_display_ts = display_ts - (config.DISPLAY_BUFFER_MS / 1000.0)

        display_packet: Optional[FramePacket] = None
        while len(frame_buffer) >= 2 and frame_buffer[1].capture_ts <= target_display_ts:
            frame_buffer.popleft()

        if frame_buffer:
            if frame_buffer[0].capture_ts <= target_display_ts:
                display_packet = frame_buffer[0]
            else:
                display_packet = frame_buffer[0]  # fallback: oldest available

        if display_packet is None:
            continue

        matched_pred: Optional[PredictionPacket] = None
        exact_match = 0
        for pred in reversed(pred_buffer):
            if pred.frame_id == display_packet.frame_id:
                matched_pred = pred
                exact_match = 1
                break
        if matched_pred is None:
            for pred in reversed(pred_buffer):
                if pred.frame_id <= display_packet.frame_id:
                    matched_pred = pred
                    break

        if matched_pred is not None:
            processed_points = matched_pred.points
            pred_frame_id = matched_pred.frame_id
            pred_age_ms = (display_ts - matched_pred.infer_end_ts) * 1000.0
            map_roi_shape = matched_pred.roi_shape
            map_infer_shape = matched_pred.infer_shape
            map_roi_offset = matched_pred.roi_offset
        else:
            processed_points = {p: {"x": None, "y": None, "likelihood": None} for p in config.USE_POINTS}
            pred_frame_id = -1
            pred_age_ms = -1.0
            map_roi_shape = (roi.shape[0], roi.shape[1])
            map_infer_shape = (infer_frame.shape[0], infer_frame.shape[1])
            map_roi_offset = roi_offset

        display_frame_id = display_packet.frame_id
        frame_delta = display_frame_id - pred_frame_id if pred_frame_id >= 0 else -1
        display_buffer_ms_actual = (display_ts - display_packet.capture_ts) * 1000.0

        # compute feature
        hind_angle = None
        if config.COMPUTE_HIND_ANGLE:
            p1, p2, p3 = config.HIND_ANGLE_POINTS
            a, b, c = processed_points.get(p1), processed_points.get(p2), processed_points.get(p3)
            if a and b and c and all(v is not None for v in [a["x"], a["y"], b["x"], b["y"], c["x"], c["y"]]):
                hind_angle = safe_angle_deg((float(a["x"]), float(a["y"])), (float(b["x"]), float(b["y"])), (float(c["x"]), float(c["y"])))

        # draw
        t_draw0 = time.perf_counter()
        if config.SHOW_FULL_FRAME:
            display_points = map_points_from_infer_to_display(
                processed_points,
                roi_shape=map_roi_shape,
                infer_shape=map_infer_shape,
                roi_offset=map_roi_offset,
            )
            display = display_packet.frame.copy()
        else:
            display_points = processed_points
            display = resize_for_infer(crop_roi(display_packet.frame))

        skip_rate = (skip_reasons["skip_total"] / max(1.0, stats["frames"])) * 100.0

        display = draw_overlay(display, display_points, stats.get("fps_cam", 0.0), fps_dlc, skip_rate, hind_angle, True, motion_score)
        t_draw1 = time.perf_counter()
        stats["t_draw_ms"] += (t_draw1 - t_draw0) * 1000.0

        t_disp0 = time.perf_counter()
        if config.SHOW_SCALE != 1.0:
            display = cv2.resize(display, None, fx=config.SHOW_SCALE, fy=config.SHOW_SCALE, interpolation=cv2.INTER_AREA)
        cv2.imshow(config.WINDOW_NAME, display)
        key = cv2.waitKey(1) & 0xFF
        display_ts = time.time()
        t_disp1 = time.perf_counter()
        stats["t_display_ms"] += (t_disp1 - t_disp0) * 1000.0

        stats["latency_ms"] += (display_ts - display_packet.capture_ts) * 1000.0
        stats["latency_count"] += 1

        if frame_id % max(1, config.LOG_EVERY_N_FRAMES) == 0:
            raw_vis = stats["raw_visible"] / max(1.0, stats["total_points"]) * 100.0
            filt_vis = stats["filtered_visible"] / max(1.0, stats["total_points"]) * 100.0
            lat_ms = stats["latency_ms"] / max(1.0, stats["latency_count"])

            bp_txt = ",".join(
                f"{bp}:{bodypart_lik_sum[bp]/max(1, bodypart_lik_cnt[bp]):.2f}"
                for bp in config.USE_POINTS
                if bodypart_lik_cnt[bp] > 0
            )
            logger.info(
                "frame_id=%d raw_visible=%.1f%% filtered_visible=%.1f%% skip_rate=%.1f%% "
                "skip_n=%d skip_motion=%d skip_duplicate=%d skip_fps=%d "
                "latency=%.1fms t_capture=%.2f t_pre=%.2f t_infer=%.2f t_post=%.2f t_draw=%.2f t_display=%.2f "
                "display_frame_id=%d pred_frame_id=%d frame_delta=%d exact_match=%d "
                "display_buffer_ms_actual=%.1f pred_age_ms=%.1f "
                "hist=%s roi=%s infer=%sx%s infer_start=%.3f infer_end=%.3f display_ts=%.3f bp_lik={%s}",
                frame_id,
                raw_vis,
                filt_vis,
                skip_rate,
                int(skip_reasons.get("skip_n", 0)),
                int(skip_reasons.get("skip_motion", 0)),
                int(skip_reasons.get("skip_duplicate", 0)),
                int(skip_reasons.get("skip_fps", 0)),
                lat_ms,
                stats["t_capture_ms"] / max(1.0, stats["frames"]),
                stats["t_pre_ms"] / max(1.0, stats["frames"]),
                stats["t_infer_ms"] / max(1.0, stats["infer_frames"]),
                stats["t_post_ms"] / max(1.0, stats["infer_frames"]),
                stats["t_draw_ms"] / max(1.0, stats["frames"]),
                stats["t_display_ms"] / max(1.0, stats["frames"]),
                display_frame_id,
                pred_frame_id,
                frame_delta,
                exact_match,
                display_buffer_ms_actual,
                pred_age_ms,
                likelihood_hist,
                config.ROI if config.USE_ROI else "full",
                config.INFER_W,
                config.INFER_H,
                infer_start_ts,
                infer_end_ts,
                display_ts,
                bp_txt,
            )

            if getattr(config, "ENABLE_BENCHMARK_LOG_ROW", False):
                path = Path(config.BENCHMARK_CSV_PATH)
                with open(path, "a", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    if not csv_header_written:
                        w.writerow([
                            "frame_id", "raw_visible", "filtered_visible", "skip_rate",
                            "skip_n", "skip_motion", "skip_duplicate", "skip_fps",
                            "latency_ms", "t_capture_ms", "t_pre_ms", "t_infer_ms", "t_post_ms", "t_draw_ms", "t_display_ms",
                            "display_frame_id", "pred_frame_id", "frame_delta", "exact_match",
                            "display_buffer_ms_actual", "pred_age_ms",
                            "hist_lt02", "hist_02_04", "hist_04_06", "hist_06_08", "hist_ge08",
                            "roi", "infer_w", "infer_h",
                        ])
                        csv_header_written = True
                    w.writerow([
                        frame_id, raw_vis, filt_vis, skip_rate,
                        skip_reasons.get("skip_n", 0), skip_reasons.get("skip_motion", 0), skip_reasons.get("skip_duplicate", 0), skip_reasons.get("skip_fps", 0),
                        lat_ms,
                        stats["t_capture_ms"] / max(1.0, stats["frames"]),
                        stats["t_pre_ms"] / max(1.0, stats["frames"]),
                        stats["t_infer_ms"] / max(1.0, stats["infer_frames"]),
                        stats["t_post_ms"] / max(1.0, stats["infer_frames"]),
                        stats["t_draw_ms"] / max(1.0, stats["frames"]),
                        stats["t_display_ms"] / max(1.0, stats["frames"]),
                        display_frame_id, pred_frame_id, frame_delta, exact_match, display_buffer_ms_actual, pred_age_ms,
                        *likelihood_hist,
                        config.ROI if config.USE_ROI else "full",
                        config.INFER_W,
                        config.INFER_H,
                    ])

        if key in (27, ord("q")):
            break

    source.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
