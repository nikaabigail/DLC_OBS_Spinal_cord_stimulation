from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Optional

import cv2

import config_rt_dlc as config
from rt_dlc_obs import (
    DLCRealtimePredictor,
    OnlinePointFilter,
    PredictionPacket,
    VideoFileSource,
    draw_overlay,
    evaluate_triplet,
    map_points_from_infer_to_display,
    resolve_roi,
    safe_angle_deg,
    setup_logger,
    validate_runtime_config,
)


@dataclass
class InferTask:
    stream_name: str
    frame_id: int
    infer_frame: object
    roi_shape: tuple[int, int]
    infer_shape: tuple[int, int]
    roi_offset: tuple[int, int]
    submit_ts: float


@dataclass
class StreamRuntime:
    name: str
    source: VideoFileSource
    point_filter: OnlinePointFilter = field(default_factory=OnlinePointFilter)
    pred_buffer: deque[PredictionPacket] = field(default_factory=lambda: deque(maxlen=getattr(config, "MAX_PRED_BUFFER", 8)))
    pred_lock: Lock = field(default_factory=Lock)
    task_lock: Lock = field(default_factory=Lock)
    latest_task: Optional[InferTask] = None

    fps_cam: float = 0.0
    prev_cam_ts: Optional[float] = None
    last_infer_submit_perf: Optional[float] = None
    last_seen_infer_end_ts: Optional[float] = None
    fps_dlc: float = 0.0

    last_overlay_points: Optional[dict[str, dict[str, float | None]]] = None
    last_overlay_hind_angle: Optional[float] = None
    last_overlay_ts: Optional[float] = None

    skip_busy_overwrite: int = 0
    submit_attempts: int = 0


def extrapolate_points(
    prev_packet: PredictionPacket,
    curr_packet: PredictionPacket,
    now_ts: float,
    max_extrapolate_ms: float,
) -> dict[str, dict[str, float | None]]:
    dt_pred = curr_packet.infer_end_ts - prev_packet.infer_end_ts
    if dt_pred <= 1e-6:
        return curr_packet.points

    dt_now = max(0.0, now_ts - curr_packet.infer_end_ts)
    dt_now = min(dt_now, max_extrapolate_ms / 1000.0)

    out: dict[str, dict[str, float | None]] = {}
    for name, cur in curr_packet.points.items():
        prev = prev_packet.points.get(name, {"x": None, "y": None, "likelihood": None})
        cx, cy, cl = cur["x"], cur["y"], cur["likelihood"]
        px, py = prev["x"], prev["y"]
        if cx is None or cy is None or cl is None or px is None or py is None:
            out[name] = dict(cur)
            continue
        vx = (float(cx) - float(px)) / dt_pred
        vy = (float(cy) - float(py)) / dt_pred
        out[name] = {
            "x": float(cx) + vx * dt_now,
            "y": float(cy) + vy * dt_now,
            "likelihood": cl,
        }
    return out


def resize_for_infer_dual(frame):
    w = int(getattr(config, "DUAL_INFER_W", config.INFER_W))
    h = int(getattr(config, "DUAL_INFER_H", config.INFER_H))
    return cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)


def _score_side(points: dict[str, dict[str, float | None]], side_points: tuple[str, str, str]) -> tuple[int, float]:
    cnt = 0
    lik = 0.0
    for name in side_points:
        p = points.get(name, {})
        x, y, l = p.get("x"), p.get("y"), p.get("likelihood")
        if x is not None and y is not None and l is not None and l >= config.CONF_THRESH_DRAW:
            cnt += 1
            lik += float(l)
    return cnt, lik


def pick_side(points: dict[str, dict[str, float | None]]) -> tuple[str, tuple[str, str, str]]:
    side_sets = getattr(config, "SIDE_POINT_SETS", {})
    best_side, best_triplet = "fallback", tuple(config.USE_POINTS[:3])
    best_metric = (-1, -1.0)
    for side_name, names in side_sets.items():
        if len(names) != 3:
            continue
        metric = _score_side(points, tuple(names))
        if metric > best_metric:
            best_metric = metric
            best_side, best_triplet = side_name, tuple(names)
    return best_side, best_triplet


def normalize_selected_triplet(
    points: dict[str, dict[str, float | None]],
    triplet_names: tuple[str, str, str],
) -> dict[str, dict[str, float | None]]:
    hip, ankle, toes = triplet_names
    return {
        "hip": dict(points.get(hip, {"x": None, "y": None, "likelihood": None})),
        "ankle": dict(points.get(ankle, {"x": None, "y": None, "likelihood": None})),
        "toes": dict(points.get(toes, {"x": None, "y": None, "likelihood": None})),
    }


def inference_worker(
    stop_event: Event,
    predictor: DLCRealtimePredictor,
    runtimes: list[StreamRuntime],
):
    rr_idx = 0
    while not stop_event.is_set():
        task: Optional[InferTask] = None
        runtime: Optional[StreamRuntime] = None

        for _ in range(len(runtimes)):
            runtime = runtimes[rr_idx]
            rr_idx = (rr_idx + 1) % len(runtimes)
            with runtime.task_lock:
                if runtime.latest_task is not None:
                    task = runtime.latest_task
                    runtime.latest_task = None
                    break

        if task is None or runtime is None:
            time.sleep(0.001)
            continue

        raw_points = predictor.predict_frame(task.infer_frame)
        filtered_points: dict[str, dict[str, float | None]] = {}
        for name, p in raw_points.items():
            x, y, l = runtime.point_filter.process_point(name, p["x"], p["y"], p["likelihood"], task.frame_id)
            filtered_points[name] = {"x": x, "y": y, "likelihood": l}

        packet = PredictionPacket(
            frame_id=task.frame_id,
            infer_start_ts=task.submit_ts,
            infer_end_ts=time.time(),
            points=filtered_points,
            roi_shape=task.roi_shape,
            infer_shape=task.infer_shape,
            roi_offset=task.roi_offset,
        )
        with runtime.pred_lock:
            runtime.pred_buffer.append(packet)


def main() -> None:
    validate_runtime_config()
    logger = setup_logger()

    if not getattr(config, "USE_DUAL_VIDEO_FILES", False):
        raise ValueError("Set USE_DUAL_VIDEO_FILES=True to run dual stream mode.")

    paths = [Path(p) for p in getattr(config, "VIDEO_FILE_PATHS", [])]
    if len(paths) != 2:
        raise ValueError("VIDEO_FILE_PATHS must contain exactly 2 paths.")

    side_sets = getattr(config, "SIDE_POINT_SETS", {})
    candidate_points = sorted({p for names in side_sets.values() for p in names})
    if not candidate_points:
        raise ValueError("SIDE_POINT_SETS is empty.")

    # Dual mode: predictor must return both side candidates.
    config.USE_POINTS = candidate_points

    left_source = VideoFileSource(paths[0], target_fps=getattr(config, "VIDEO_TARGET_FPS", None), skip_if_behind=getattr(config, "VIDEO_SKIP_IF_BEHIND", True))
    right_source = VideoFileSource(paths[1], target_fps=getattr(config, "VIDEO_TARGET_FPS", None), skip_if_behind=getattr(config, "VIDEO_SKIP_IF_BEHIND", True))
    left_source.open()
    right_source.open()

    left = StreamRuntime(name="left_cam", source=left_source)
    right = StreamRuntime(name="right_cam", source=right_source)
    runtimes = [left, right]

    predictor = DLCRealtimePredictor()
    stop_event = Event()
    worker = Thread(target=inference_worker, daemon=True, args=(stop_event, predictor, runtimes))
    worker.start()

    logger.info(
        "Dual async pipeline started. left=%s right=%s infer=%sx%s points=%s",
        paths[0],
        paths[1],
        int(getattr(config, "DUAL_INFER_W", config.INFER_W)),
        int(getattr(config, "DUAL_INFER_H", config.INFER_H)),
        candidate_points,
    )

    try:
        while True:
            displays: dict[str, object] = {}
            ended = False

            for rt in runtimes:
                ok, packet = rt.source.read()
                if not ok or packet is None:
                    ended = True
                    break

                frame = packet.frame
                frame_id = packet.frame_id
                now = packet.capture_ts

                if rt.prev_cam_ts is not None:
                    dt = now - rt.prev_cam_ts
                    if dt > 0:
                        rt.fps_cam = 1.0 / dt
                rt.prev_cam_ts = now

                roi, roi_offset = resolve_roi(frame)
                infer_frame = resize_for_infer_dual(roi)

                # Scheduler gate: inference cadence and per-stream soft-cap.
                run_infer = frame_id % max(1, int(getattr(config, "INFER_EVERY_N_FRAMES", 1))) == 0
                if run_infer and rt.last_infer_submit_perf is not None:
                    min_dt = 1.0 / max(1.0, float(getattr(config, "DUAL_TARGET_INFER_FPS_PER_STREAM", 10.0)))
                    if (time.perf_counter() - rt.last_infer_submit_perf) < min_dt:
                        run_infer = False

                if run_infer:
                    rt.submit_attempts += 1
                    task = InferTask(
                        stream_name=rt.name,
                        frame_id=frame_id,
                        infer_frame=infer_frame.copy(),
                        roi_shape=(roi.shape[0], roi.shape[1]),
                        infer_shape=(infer_frame.shape[0], infer_frame.shape[1]),
                        roi_offset=roi_offset,
                        submit_ts=time.time(),
                    )
                    with rt.task_lock:
                        if rt.latest_task is not None:
                            rt.skip_busy_overwrite += 1
                        rt.latest_task = task  # latest-only slot (old task is overwritten)
                    rt.last_infer_submit_perf = time.perf_counter()

                with rt.pred_lock:
                    preds = list(rt.pred_buffer)

                matched = None
                prev_matched = None
                for pred in reversed(preds):
                    if pred.frame_id <= frame_id:
                        matched = pred
                        break
                if matched is not None:
                    for pred in reversed(preds):
                        if pred.frame_id < matched.frame_id:
                            prev_matched = pred
                            break

                if matched is None:
                    processed_points = {p: {"x": None, "y": None, "likelihood": None} for p in candidate_points}
                    map_roi_shape = (roi.shape[0], roi.shape[1])
                    map_infer_shape = (infer_frame.shape[0], infer_frame.shape[1])
                    map_roi_offset = roi_offset
                else:
                    if rt.last_seen_infer_end_ts is not None:
                        dt_inf = matched.infer_end_ts - rt.last_seen_infer_end_ts
                        if dt_inf > 0:
                            rt.fps_dlc = 1.0 / dt_inf
                    rt.last_seen_infer_end_ts = matched.infer_end_ts

                    processed_points = matched.points
                    if (
                        bool(getattr(config, "DUAL_ENABLE_EXTRAPOLATION", True))
                        and prev_matched is not None
                    ):
                        processed_points = extrapolate_points(
                            prev_packet=prev_matched,
                            curr_packet=matched,
                            now_ts=time.time(),
                            max_extrapolate_ms=float(getattr(config, "DUAL_EXTRAPOLATE_MAX_MS", 80.0)),
                        )
                    map_roi_shape = matched.roi_shape
                    map_infer_shape = matched.infer_shape
                    map_roi_offset = matched.roi_offset

                picked_side, picked_triplet = pick_side(processed_points)
                canonical = normalize_selected_triplet(processed_points, picked_triplet)
                draw_points, has_triplet, reason = evaluate_triplet(canonical, ["hip", "ankle", "toes"])

                hind_angle = None
                overlay_source = "live"
                if has_triplet:
                    a, b, c = draw_points["hip"], draw_points["ankle"], draw_points["toes"]
                    hind_angle = safe_angle_deg((float(a["x"]), float(a["y"])), (float(b["x"]), float(b["y"])), (float(c["x"]), float(c["y"])))
                    rt.last_overlay_points = {k: dict(v) for k, v in draw_points.items()}
                    rt.last_overlay_hind_angle = hind_angle
                    rt.last_overlay_ts = time.time()
                else:
                    hold_ms = float(getattr(config, "OVERLAY_HOLD_MS", 0.0))
                    if (
                        hold_ms > 0
                        and rt.last_overlay_points is not None
                        and rt.last_overlay_ts is not None
                        and (time.time() - rt.last_overlay_ts) * 1000.0 <= hold_ms
                    ):
                        draw_points = {k: dict(v) for k, v in rt.last_overlay_points.items()}
                        hind_angle = rt.last_overlay_hind_angle
                        overlay_source = "hold"
                    else:
                        draw_points = {"hip": {"x": None, "y": None, "likelihood": None}, "ankle": {"x": None, "y": None, "likelihood": None}, "toes": {"x": None, "y": None, "likelihood": None}}
                        overlay_source = "none"

                display_points = map_points_from_infer_to_display(
                    draw_points,
                    roi_shape=map_roi_shape,
                    infer_shape=map_infer_shape,
                    roi_offset=map_roi_offset,
                )
                display = draw_overlay(
                    frame,
                    display_points,
                    fps_cam=rt.fps_cam,
                    fps_dlc=rt.fps_dlc,
                    skip_rate=(rt.skip_busy_overwrite / max(1, rt.submit_attempts)) * 100.0,
                    hind_angle=hind_angle,
                    buffer_status_text=None,
                    processing_active=True,
                    motion_score=0.0,
                )
                cv2.putText(display, f"Stream: {rt.name} | side: {picked_side}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                if frame_id % max(1, int(getattr(config, "LOG_EVERY_N_FRAMES", 30))) == 0:
                    logger.info(
                        "stream=%s frame=%d side=%s infer=%s slot_overwrite=%d draw=%d triplet=%s source=%s reason=%s",
                        rt.name,
                        frame_id,
                        picked_side,
                        run_infer,
                        rt.skip_busy_overwrite,
                        sum(1 for p in display_points.values() if p["x"] is not None and p["y"] is not None and p["likelihood"] is not None),
                        has_triplet,
                        overlay_source,
                        reason,
                    )

                if config.SHOW_SCALE != 1.0:
                    display = cv2.resize(display, None, fx=config.SHOW_SCALE, fy=config.SHOW_SCALE, interpolation=cv2.INTER_AREA)
                displays[rt.name] = display

            if ended:
                break

            cv2.imshow(f"{config.WINDOW_NAME} | left_cam", displays["left_cam"])
            cv2.imshow(f"{config.WINDOW_NAME} | right_cam", displays["right_cam"])
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break

    finally:
        stop_event.set()
        worker.join(timeout=1.0)
        left_source.release()
        right_source.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
