from __future__ import annotations

import argparse
import csv
import random
import statistics
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SimPrediction:
    frame_id: int
    ready_ts: float


def run_single_sim(
    *,
    display_buffer_ms: float,
    duration_s: float,
    capture_fps: float,
    infer_fps: float,
    infer_latency_ms: float,
    infer_jitter_ms: float,
    max_frame_buffer: int,
) -> dict[str, float]:
    dt = 1.0 / max(capture_fps, 1e-6)
    frame_packets: list[tuple[int, float]] = []
    pred_queue: list[SimPrediction] = []

    pred_next_ready = 0.0
    pred_cooldown = 1.0 / max(infer_fps, 1e-6)

    t = 0.0
    frame_id = 0
    exact_match_count = 0
    samples = 0
    display_buffer_actual_ms: list[float] = []
    pred_age_ms: list[float] = []

    while t < duration_s:
        frame_id += 1
        frame_packets.append((frame_id, t))
        if len(frame_packets) > max(1, int(max_frame_buffer)):
            frame_packets = frame_packets[-int(max(1, max_frame_buffer)) :]

        if t >= pred_next_ready:
            jitter = random.uniform(-infer_jitter_ms, infer_jitter_ms)
            ready_ts = t + max(0.0, (infer_latency_ms + jitter) / 1000.0)
            pred_queue.append(SimPrediction(frame_id=frame_id, ready_ts=ready_ts))
            pred_next_ready = t + pred_cooldown

        target_ts = t - (display_buffer_ms / 1000.0)

        display_idx = 0
        while display_idx + 1 < len(frame_packets) and frame_packets[display_idx + 1][1] <= target_ts:
            display_idx += 1
        display_frame_id, display_frame_ts = frame_packets[display_idx]
        display_buffer_actual_ms.append((t - display_frame_ts) * 1000.0)

        ready_preds = [p for p in pred_queue if p.ready_ts <= t]
        matched = None
        for p in reversed(ready_preds):
            if p.frame_id == display_frame_id:
                matched = p
                exact_match_count += 1
                break
        if matched is None:
            for p in reversed(ready_preds):
                if p.frame_id <= display_frame_id:
                    matched = p
                    break
        if matched is not None:
            pred_age_ms.append((t - matched.ready_ts) * 1000.0)

        samples += 1
        t += dt

    return {
        "samples": float(samples),
        "exact_match_pct": (exact_match_count / max(samples, 1)) * 100.0,
        "display_buffer_mean_ms": statistics.fmean(display_buffer_actual_ms) if display_buffer_actual_ms else 0.0,
        "pred_age_mean_ms": statistics.fmean(pred_age_ms) if pred_age_ms else 0.0,
        "pred_age_p95_ms": sorted(pred_age_ms)[int(0.95 * (len(pred_age_ms) - 1))] if pred_age_ms else 0.0,
    }


def simulate(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    buffers = [float(v) for v in args.buffers]
    print(
        "buffer_ms,exact_match_pct,display_buffer_mean_ms,pred_age_mean_ms,pred_age_p95_ms"
    )
    for b_ms in buffers:
        runs = [
            run_single_sim(
                display_buffer_ms=b_ms,
                duration_s=args.duration_s,
                capture_fps=args.capture_fps,
                infer_fps=args.infer_fps,
                infer_latency_ms=args.infer_latency_ms,
                infer_jitter_ms=args.infer_jitter_ms,
                max_frame_buffer=args.max_frame_buffer,
            )
            for _ in range(args.repeats)
        ]
        print(
            f"{b_ms:.1f},"
            f"{statistics.fmean(r['exact_match_pct'] for r in runs):.2f},"
            f"{statistics.fmean(r['display_buffer_mean_ms'] for r in runs):.2f},"
            f"{statistics.fmean(r['pred_age_mean_ms'] for r in runs):.2f},"
            f"{statistics.fmean(r['pred_age_p95_ms'] for r in runs):.2f}"
        )


def summarize_csv(args: argparse.Namespace) -> None:
    path = Path(args.csv)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    with open(path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print("CSV has no rows.")
        return

    def _col(name: str) -> list[float]:
        out: list[float] = []
        for r in rows:
            if name in r and r[name] not in {"", None}:
                out.append(float(r[name]))
        return out

    display_buf = _col("display_buffer_ms_actual")
    pred_age = _col("pred_age_ms")
    exact_match = _col("exact_match")
    frame_delta = _col("frame_delta")

    print(f"rows={len(rows)}")
    if display_buf:
        print(f"display_buffer_ms_actual_mean={statistics.fmean(display_buf):.2f}")
    if pred_age:
        print(f"pred_age_ms_mean={statistics.fmean(pred_age):.2f}")
    if exact_match:
        print(f"exact_match_pct={100.0 * (sum(exact_match) / len(exact_match)):.2f}")
    if frame_delta:
        print(f"frame_delta_mean={statistics.fmean(frame_delta):.2f}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Проверка буферизации online-пайплайна (симуляция и сводка benchmark CSV)."
    )
    sub = p.add_subparsers(dest="mode", required=True)

    sim = sub.add_parser("simulate", help="Многократная симуляция для набора DISPLAY_BUFFER_MS.")
    sim.add_argument("--buffers", nargs="+", default=["0", "20", "40", "80", "120"])
    sim.add_argument("--repeats", type=int, default=5)
    sim.add_argument("--duration-s", type=float, default=20.0)
    sim.add_argument("--capture-fps", type=float, default=60.0)
    sim.add_argument("--infer-fps", type=float, default=25.0)
    sim.add_argument("--infer-latency-ms", type=float, default=35.0)
    sim.add_argument("--infer-jitter-ms", type=float, default=10.0)
    sim.add_argument("--max-frame-buffer", type=int, default=8)
    sim.add_argument("--seed", type=int, default=42)
    sim.set_defaults(func=simulate)

    csvp = sub.add_parser("summarize-csv", help="Сводка по rt_dlc_obs benchmark CSV.")
    csvp.add_argument("--csv", required=True)
    csvp.set_defaults(func=summarize_csv)

    return p


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
