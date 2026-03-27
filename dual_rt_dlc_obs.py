from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2

import config_rt_dlc as config
from rt_dlc_obs import (
    OnlinePointFilter,
    DLCRealtimePredictor,
    VideoFileSource,
    draw_overlay,
    evaluate_triplet,
    map_points_from_infer_to_display,
    resolve_roi,
    resize_for_infer,
    safe_angle_deg,
    setup_logger,
    validate_runtime_config,
)


@dataclass
class StreamState:
    name: str
    source: VideoFileSource
    point_filter: OnlinePointFilter
    frame_id: int = 0
    fps_cam: float = 0.0
    prev_cam_ts: Optional[float] = None
    last_overlay_points: Optional[dict[str, dict[str, float | None]]] = None
    last_overlay_hind_angle: Optional[float] = None
    last_overlay_ts: Optional[float] = None


def _canonical_triplet(points: dict[str, dict[str, float | None]], side_points: tuple[str, str, str]) -> dict[str, dict[str, float | None]]:
    hip, ankle, toes = side_points
    return {
        "hip": dict(points.get(hip, {"x": None, "y": None, "likelihood": None})),
        "ankle": dict(points.get(ankle, {"x": None, "y": None, "likelihood": None})),
        "toes": dict(points.get(toes, {"x": None, "y": None, "likelihood": None})),
    }


def _score_side(points: dict[str, dict[str, float | None]], side_points: tuple[str, str, str]) -> tuple[int, float]:
    score_cnt = 0
    score_lik = 0.0
    for n in side_points:
        p = points.get(n, {})
        x, y, l = p.get("x"), p.get("y"), p.get("likelihood")
        if x is not None and y is not None and l is not None and l >= config.CONF_THRESH_DRAW:
            score_cnt += 1
            score_lik += float(l)
    return score_cnt, score_lik


def pick_side(points: dict[str, dict[str, float | None]]) -> tuple[str, tuple[str, str, str]]:
    side_sets = getattr(config, "SIDE_POINT_SETS", {})
    if not side_sets:
        side_sets = {"right": tuple(config.USE_POINTS)}

    best_side = None
    best_points = None
    best_metric = (-1, -1.0)

    for side_name, names in side_sets.items():
        if len(names) != 3:
            continue
        cnt, lik = _score_side(points, tuple(names))
        metric = (cnt, lik)
        if metric > best_metric:
            best_metric = metric
            best_side = side_name
            best_points = tuple(names)

    if best_side is None or best_points is None:
        fallback = tuple(config.USE_POINTS[:3])
        return "fallback", fallback

    return best_side, best_points


def process_stream_frame(
    predictor: DLCRealtimePredictor,
    stream: StreamState,
    logger,
) -> tuple[bool, Optional[object]]:
    ret, packet = stream.source.read()
    if not ret or packet is None:
        return False, None

    frame = packet.frame
    stream.frame_id = packet.frame_id
    capture_ts = packet.capture_ts

    if stream.prev_cam_ts is not None:
        dt = capture_ts - stream.prev_cam_ts
        if dt > 0:
            stream.fps_cam = 1.0 / dt
    stream.prev_cam_ts = capture_ts

    roi, roi_offset = resolve_roi(frame)
    infer_frame = resize_for_infer(roi)

    raw_points = predictor.predict_frame(infer_frame)

    filtered_points: dict[str, dict[str, float | None]] = {}
    for name, p in raw_points.items():
        x, y, l = stream.point_filter.process_point(name, p["x"], p["y"], p["likelihood"], stream.frame_id)
        filtered_points[name] = {"x": x, "y": y, "likelihood": l}

    picked_side, picked_side_points = pick_side(filtered_points)
    side_points_dict = _canonical_triplet(filtered_points, picked_side_points)
    draw_points, has_triplet, reason = evaluate_triplet(side_points_dict, ["hip", "ankle", "toes"])
    overlay_source = "live"

    hind_angle = None
    if has_triplet:
        a, b, c = draw_points["hip"], draw_points["ankle"], draw_points["toes"]
        hind_angle = safe_angle_deg((float(a["x"]), float(a["y"])), (float(b["x"]), float(b["y"])), (float(c["x"]), float(c["y"])))
        stream.last_overlay_points = {k: dict(v) for k, v in draw_points.items()}
        stream.last_overlay_hind_angle = hind_angle
        stream.last_overlay_ts = time.time()
    else:
        hold_ms = float(getattr(config, "OVERLAY_HOLD_MS", 0.0))
        now = time.time()
        if (
            hold_ms > 0
            and stream.last_overlay_points is not None
            and stream.last_overlay_ts is not None
            and (now - stream.last_overlay_ts) * 1000.0 <= hold_ms
        ):
            draw_points = {k: dict(v) for k, v in stream.last_overlay_points.items()}
            hind_angle = stream.last_overlay_hind_angle
            overlay_source = "hold"
        else:
            draw_points = {"hip": {"x": None, "y": None, "likelihood": None}, "ankle": {"x": None, "y": None, "likelihood": None}, "toes": {"x": None, "y": None, "likelihood": None}}
            overlay_source = "none"

    display_points = map_points_from_infer_to_display(
        draw_points,
        roi_shape=(roi.shape[0], roi.shape[1]),
        infer_shape=(infer_frame.shape[0], infer_frame.shape[1]),
        roi_offset=roi_offset,
    )

    display = draw_overlay(
        frame,
        display_points,
        fps_cam=stream.fps_cam,
        fps_dlc=0.0,
        skip_rate=0.0,
        hind_angle=hind_angle,
        processing_active=True,
        motion_score=0.0,
    )

    logger.info(
        "stream=%s frame=%d side=%s raw=%d filt=%d draw=%d triplet=%s source=%s reason=%s",
        stream.name,
        stream.frame_id,
        picked_side,
        len(raw_points),
        sum(1 for p in filtered_points.values() if p["x"] is not None and p["y"] is not None and p["likelihood"] is not None),
        sum(1 for p in display_points.values() if p["x"] is not None and p["y"] is not None and p["likelihood"] is not None),
        has_triplet,
        overlay_source,
        reason,
    )

    cv2.putText(display, f"Stream: {stream.name} | side: {picked_side}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return True, display


def main() -> None:
    validate_runtime_config()
    logger = setup_logger()

    if not getattr(config, "USE_DUAL_VIDEO_FILES", False):
        raise ValueError("Set USE_DUAL_VIDEO_FILES=True to run dual stream mode.")

    paths = [Path(p) for p in getattr(config, "VIDEO_FILE_PATHS", [])]
    if len(paths) != 2:
        raise ValueError("VIDEO_FILE_PATHS must contain exactly 2 video paths.")

    side_sets = getattr(config, "SIDE_POINT_SETS", {})
    candidate_points = sorted({p for names in side_sets.values() for p in names})
    if not candidate_points:
        raise ValueError("SIDE_POINT_SETS is empty. Provide left/right triplets.")

    # Predictor must output both sides; temporarily override USE_POINTS with union.
    config.USE_POINTS = candidate_points

    source_l = VideoFileSource(paths[0], target_fps=getattr(config, "VIDEO_TARGET_FPS", None), skip_if_behind=getattr(config, "VIDEO_SKIP_IF_BEHIND", True))
    source_r = VideoFileSource(paths[1], target_fps=getattr(config, "VIDEO_TARGET_FPS", None), skip_if_behind=getattr(config, "VIDEO_SKIP_IF_BEHIND", True))
    source_l.open()
    source_r.open()

    predictor = DLCRealtimePredictor()
    stream_l = StreamState("left_cam", source_l, OnlinePointFilter())
    stream_r = StreamState("right_cam", source_r, OnlinePointFilter())

    logger.info("Dual pipeline started. left=%s right=%s points=%s", paths[0], paths[1], candidate_points)

    try:
        while True:
            ok_l, frame_l = process_stream_frame(predictor, stream_l, logger)
            ok_r, frame_r = process_stream_frame(predictor, stream_r, logger)
            if not ok_l or not ok_r or frame_l is None or frame_r is None:
                break

            if frame_l.shape[0] != frame_r.shape[0]:
                frame_r = cv2.resize(frame_r, (frame_r.shape[1], frame_l.shape[0]))
            combined = cv2.hconcat([frame_l, frame_r])

            if config.SHOW_SCALE != 1.0:
                combined = cv2.resize(combined, None, fx=config.SHOW_SCALE, fy=config.SHOW_SCALE, interpolation=cv2.INTER_AREA)

            cv2.imshow(f"{config.WINDOW_NAME} | dual", combined)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
    finally:
        source_l.release()
        source_r.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
