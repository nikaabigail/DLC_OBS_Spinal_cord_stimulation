from __future__ import annotations

import math
import logging
import time
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


def detect_content_roi(frame: np.ndarray, black_thresh: int = 12) -> tuple[int, int, int, int] | None:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask = gray > black_thresh
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    if x2 - x1 < 32 or y2 - y1 < 32:
        return None
    return x1, y1, x2, y2


def resize_for_infer(frame: np.ndarray) -> np.ndarray:
    return cv2.resize(
        frame,
        (config.INFER_W, config.INFER_H),
        interpolation=cv2.INTER_LINEAR,
    )


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
    """
    Перевод координат из infer-frame обратно в full frame / ROI frame.

    roi_shape: (h, w) ROI после crop
    infer_shape: (h, w) infer_frame
    roi_offset: (x_offset, y_offset) ROI внутри полного кадра
    """
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
# Online point filter
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
            and likelihood >= config.CONF_THRESH_USE
        )

        if is_good:
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

        if state.last_good_xy is not None and state.last_good_frame_idx is not None:
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
# DLC realtime wrapper
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
        try:
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
        except Exception as e:
            raise RuntimeError(
                "Не удалось инициализировать DLC pose inference runner."
            ) from e

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

        if isinstance(raw, (list, tuple)) and len(raw) > 0:
            raw0 = raw[0]
        else:
            raw0 = raw

        if not (isinstance(raw0, dict) and "bodypart" in raw0 and "poses" in raw0["bodypart"]):
            raise RuntimeError(
                f"Неожиданный формат pred output: type={type(raw0)}, repr={repr(raw0)[:500]}"
            )

        poses = self._normalize_poses(raw0["bodypart"]["poses"])
        points: dict[str, dict[str, float | None]] = {}

        for point_name in config.USE_POINTS:
            idx = self.bodypart_to_idx.get(point_name)
            if idx is None:
                points[point_name] = {"x": None, "y": None, "likelihood": None}
                continue

            if idx >= len(poses):
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
    fps_skip: float,
    hind_angle: Optional[float],
    processing_active: bool,
    motion_score: float,
) -> np.ndarray:
    out = frame.copy()

    if config.DRAW_POINTS:
        for point_name, p in points.items():
            x = p["x"]
            y = p["y"]
            l = p["likelihood"]

            if x is None or y is None or l is None:
                continue
            if l < config.CONF_THRESH_DRAW:
                continue

            cv2.circle(out, (int(x), int(y)), 4, (0, 255, 0), -1)

            if config.DRAW_NAMES:
                label = point_name
                if config.DRAW_CONF:
                    label += f" {l:.2f}"
                cv2.putText(
                    out,
                    label,
                    (int(x) + 5, int(y) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )

    y0 = 20
    status_text = "ACTIVE" if processing_active else "WAITING FOR PLAY"
    status_color = (0, 255, 0) if processing_active else (0, 165, 255)
    cv2.putText(
        out,
        f"STATUS: {status_text}",
        (10, y0),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        status_color,
        2,
        cv2.LINE_AA,
    )
    y0 += 25

    cv2.putText(
        out,
        f"MOTION: {motion_score:.3f}",
        (10, y0),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 200, 0),
        2,
        cv2.LINE_AA,
    )
    y0 += 25

    if config.DRAW_FPS:
        cv2.putText(
            out,
            f"CAM FPS: {fps_cam:.1f} | DLC FPS: {fps_dlc:.1f} | SKIP FPS: {fps_skip:.1f}",
            (10, y0),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y0 += 25

    if config.DRAW_HIND_ANGLE and hind_angle is not None:
        cv2.putText(
            out,
            f"Hind angle: {hind_angle:.1f} deg",
            (10, y0),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )

    return out


# =========================
# Main loop
# =========================
def main() -> None:
    validate_runtime_config()
    logger = setup_logger()

    cap = cv2.VideoCapture(config.CAM_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_H)
    cap.set(cv2.CAP_PROP_FPS, config.TARGET_VIDEO_FPS)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {config.CAM_INDEX}")

    actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    logger.info(
        "Camera opened: index=%s requested=(%sx%s @ %.1f) actual=(%.0fx%.0f @ %.1f)",
        config.CAM_INDEX,
        config.FRAME_W,
        config.FRAME_H,
        config.TARGET_VIDEO_FPS,
        actual_w,
        actual_h,
        actual_fps,
    )
    if actual_fps > 0 and actual_fps < config.TARGET_VIDEO_FPS * 0.8:
        logger.warning(
            "Actual camera FPS is much lower than requested (%.1f vs %.1f). "
            "Source/backend likely capped.",
            actual_fps,
            config.TARGET_VIDEO_FPS,
        )

    predictor = DLCRealtimePredictor()
    point_filter = OnlinePointFilter()

    frame_idx = 0
    prev_cam_ts: Optional[float] = None
    prev_skip_ts: Optional[float] = None
    fps_cam = 0.0
    fps_dlc = 0.0
    fps_skip = 0.0

    prev_gray_for_motion: Optional[np.ndarray] = None
    prev_gray_for_dup: Optional[np.ndarray] = None
    motion_score = 0.0
    processing_active = not getattr(config, "AUTO_START_ON_MOTION", False)

    infer_queue: Queue[InferJob] = Queue(maxsize=config.INFER_QUEUE_MAXSIZE)
    result_lock = Lock()
    stop_event = Event()
    latest_result: Optional[InferResult] = None
    results_by_frame: dict[int, InferResult] = {}

    display_delay_frames = int(round(
        getattr(config, "DISPLAY_DELAY_MS", 0) / 1000.0 * max(1.0, config.TARGET_VIDEO_FPS)
    ))
    frame_buffer: deque[tuple[int, np.ndarray, tuple[int, int], tuple[int, int], tuple[int, int], float, bool]] = deque()
    stats = {
        "infer_jobs": 0,
        "infer_frames": 0,
        "exact_match": 0,
        "fallback_match": 0,
        "empty_match": 0,
        "visible_points": 0,
        "total_points": 0,
    }

    def infer_worker() -> None:
        nonlocal fps_dlc, latest_result
        prev_dlc_ts_local: Optional[float] = None

        while not stop_event.is_set():
            try:
                job = infer_queue.get(timeout=0.1)
            except Empty:
                continue

            # Anti-backlog: берем самый свежий кадр.
            while True:
                try:
                    job = infer_queue.get_nowait()
                except Empty:
                    break

            raw_points = predictor.predict_frame(job.infer_frame)
            stats["infer_frames"] += 1

            processed_points: dict[str, dict[str, float | None]] = {}
            for point_name, p in raw_points.items():
                x, y, l = point_filter.process_point(
                    name=point_name,
                    x=p["x"],
                    y=p["y"],
                    likelihood=p["likelihood"],
                    frame_idx=job.frame_idx,
                )
                processed_points[point_name] = {"x": x, "y": y, "likelihood": l}
                stats["total_points"] += 1
                if x is not None and y is not None and l is not None and l >= config.CONF_THRESH_DRAW:
                    stats["visible_points"] += 1

            t_done = time.time()
            if prev_dlc_ts_local is not None:
                dt = t_done - prev_dlc_ts_local
                if dt > 0:
                    fps_dlc = 1.0 / dt
            prev_dlc_ts_local = t_done

            result = InferResult(
                frame_idx=job.frame_idx,
                points=processed_points,
                roi_shape=job.roi_shape,
                infer_shape=job.infer_shape,
                roi_offset=job.roi_offset,
                ts=t_done,
            )

            with result_lock:
                latest_result = result
                results_by_frame[job.frame_idx] = result
                if len(results_by_frame) > config.MAX_RESULT_HISTORY:
                    for old_idx in sorted(results_by_frame.keys())[: len(results_by_frame) - config.MAX_RESULT_HISTORY]:
                        del results_by_frame[old_idx]

    worker = Thread(target=infer_worker, name="dlc-infer-worker", daemon=True)
    worker.start()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Empty frame from OBS camera.")
                break

            ts_frame = time.time()
            if prev_cam_ts is not None:
                dt = ts_frame - prev_cam_ts
                if dt > 0:
                    fps_cam = 1.0 / dt
            prev_cam_ts = ts_frame

            frame_idx += 1

            if config.USE_ROI:
                x1, y1, x2, y2 = config.ROI
                roi_offset = (x1, y1)
            else:
                roi_offset = (0, 0)

            if frame_idx == 1 and getattr(config, "AUTO_DETECT_CONTENT_ROI", False):
                auto_roi = detect_content_roi(frame, black_thresh=getattr(config, "CONTENT_BLACK_THRESH", 12))
                if auto_roi is not None:
                    config.ROI = auto_roi
                    config.USE_ROI = True
                    x1, y1, x2, y2 = config.ROI
                    roi_offset = (x1, y1)
                    logger.info("AUTO_DETECT_CONTENT_ROI enabled. Using ROI=%s", config.ROI)
                else:
                    logger.warning("AUTO_DETECT_CONTENT_ROI could not find content. Keeping configured ROI.")

            roi = crop_roi(frame)
            infer_frame = resize_for_infer(roi)

            if getattr(config, "AUTO_START_ON_MOTION", False):
                gray_small = cv2.cvtColor(infer_frame, cv2.COLOR_BGR2GRAY)
                if prev_gray_for_motion is None:
                    prev_gray_for_motion = gray_small
                else:
                    diff = cv2.absdiff(gray_small, prev_gray_for_motion)
                    motion_score = float(diff.mean())
                    prev_gray_for_motion = gray_small

                    if (
                        not processing_active
                        and motion_score > getattr(config, "FRAME_DIFF_THRESHOLD", 0.5)
                    ):
                        processing_active = True
                        print(
                            f"[INFO] Playback detected. Starting DLC processing. "
                            f"motion_score={motion_score:.3f}"
                        )

            run_dlc = processing_active
            if processing_active and getattr(config, "SKIP_NEAR_DUPLICATE_FRAMES", True):
                gray_dup = cv2.cvtColor(infer_frame, cv2.COLOR_BGR2GRAY)
                if prev_gray_for_dup is None:
                    prev_gray_for_dup = gray_dup
                else:
                    dup_diff = cv2.absdiff(gray_dup, prev_gray_for_dup)
                    dup_score = float(dup_diff.mean())

                    if dup_score < getattr(config, "DUPLICATE_FRAME_THRESHOLD", 0.15):
                        run_dlc = False
                        t_skip = time.time()
                        if prev_skip_ts is not None:
                            dt = t_skip - prev_skip_ts
                            if dt > 0:
                                fps_skip = 1.0 / dt
                        prev_skip_ts = t_skip
                    else:
                        prev_gray_for_dup = gray_dup

            if run_dlc:
                job = InferJob(
                    frame_idx=frame_idx,
                    infer_frame=infer_frame.copy(),
                    roi_shape=(roi.shape[0], roi.shape[1]),
                    infer_shape=(infer_frame.shape[0], infer_frame.shape[1]),
                    roi_offset=roi_offset,
                )
                try:
                    infer_queue.put_nowait(job)
                    stats["infer_jobs"] += 1
                except Full:
                    try:
                        _ = infer_queue.get_nowait()
                    except Empty:
                        pass
                    infer_queue.put_nowait(job)
                    stats["infer_jobs"] += 1

            frame_buffer.append(
                (
                    frame_idx,
                    frame.copy(),
                    (roi.shape[0], roi.shape[1]),
                    (infer_frame.shape[0], infer_frame.shape[1]),
                    roi_offset,
                    motion_score,
                    processing_active,
                )
            )

            if len(frame_buffer) <= display_delay_frames:
                continue

            disp_idx, disp_frame, disp_roi_shape, disp_infer_shape, disp_roi_offset, disp_motion, disp_active = frame_buffer.popleft()

            with result_lock:
                result_exact = results_by_frame.pop(disp_idx, None)
                result_fallback = latest_result

            if result_exact is not None:
                processed_points = result_exact.points
                map_meta = result_exact
                stats["exact_match"] += 1
            elif result_fallback is not None and disp_active:
                processed_points = result_fallback.points
                map_meta = InferResult(
                    frame_idx=disp_idx,
                    points=processed_points,
                    roi_shape=disp_roi_shape,
                    infer_shape=disp_infer_shape,
                    roi_offset=disp_roi_offset,
                    ts=time.time(),
                )
                stats["fallback_match"] += 1
            else:
                processed_points = {
                    point_name: {"x": None, "y": None, "likelihood": None}
                    for point_name in config.USE_POINTS
                }
                map_meta = InferResult(
                    frame_idx=disp_idx,
                    points=processed_points,
                    roi_shape=disp_roi_shape,
                    infer_shape=disp_infer_shape,
                    roi_offset=disp_roi_offset,
                    ts=time.time(),
                )
                stats["empty_match"] += 1

            hind_angle = None
            if disp_active and config.COMPUTE_HIND_ANGLE:
                p1, p2, p3 = config.HIND_ANGLE_POINTS
                a = processed_points.get(p1)
                b = processed_points.get(p2)
                c = processed_points.get(p3)

                if a and b and c:
                    if (
                        a["x"] is not None and a["y"] is not None
                        and b["x"] is not None and b["y"] is not None
                        and c["x"] is not None and c["y"] is not None
                    ):
                        hind_angle = safe_angle_deg(
                            (float(a["x"]), float(a["y"])),
                            (float(b["x"]), float(b["y"])),
                            (float(c["x"]), float(c["y"])),
                        )

            if getattr(config, "SHOW_FULL_FRAME", True):
                display_points = map_points_from_infer_to_display(
                    processed_points,
                    roi_shape=map_meta.roi_shape,
                    infer_shape=map_meta.infer_shape,
                    roi_offset=map_meta.roi_offset,
                )
                display = disp_frame.copy()
            else:
                display_points = processed_points
                display = resize_for_infer(crop_roi(disp_frame))

            display = draw_overlay(
                display,
                display_points,
                fps_cam,
                fps_dlc,
                fps_skip,
                hind_angle,
                disp_active,
                disp_motion,
            )

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
            if key in (27, ord("q")):
                break

            if frame_idx % max(1, getattr(config, "LOG_EVERY_N_FRAMES", 30)) == 0:
                visibility = (
                    stats["visible_points"] / stats["total_points"] * 100.0
                    if stats["total_points"] > 0
                    else 0.0
                )
                logger.info(
                    "frame=%d cam_fps=%.1f dlc_fps=%.1f skip_fps=%.1f q=%d "
                    "jobs=%d infer=%d exact=%d fallback=%d empty=%d visible=%.1f%% motion=%.3f",
                    frame_idx,
                    fps_cam,
                    fps_dlc,
                    fps_skip,
                    infer_queue.qsize(),
                    stats["infer_jobs"],
                    stats["infer_frames"],
                    stats["exact_match"],
                    stats["fallback_match"],
                    stats["empty_match"],
                    visibility,
                    motion_score,
                )
    finally:
        stop_event.set()
        worker.join(timeout=1.0)
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
