from __future__ import annotations

import ast
import math
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Any

import config_rt_dlc as config
import rt_dlc_obs


def iter_config_keys() -> list[str]:
    keys: list[str] = []
    for name, value in vars(config).items():
        if not name.isupper():
            continue
        if callable(value):
            continue
        keys.append(name)
    return sorted(keys)


def parse_value(raw: str, current_value: Any) -> Any:
    if isinstance(current_value, str):
        return raw
    if isinstance(current_value, Path):
        return Path(raw)
    if isinstance(current_value, bool):
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(current_value, int) and not isinstance(current_value, bool):
        return int(raw)
    if isinstance(current_value, float):
        return float(raw)
    # tuple/list/dict/Path-like complex values
    return ast.literal_eval(raw)


class ConfigGUI:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("DLC Runtime Config GUI")
        self.root.geometry("1200x850")

        self.keys = iter_config_keys()
        self.vars: dict[str, tk.Variable] = {}
        self.widgets: dict[str, ttk.Entry] = {}
        self.validation_label = ttk.Label(self.root, text="", foreground="#c0392b")
        self.stats_label = ttk.Label(self.root, text="", foreground="#2c3e50")

        self._build_ui()
        self._refresh_stats()

    def _build_ui(self) -> None:
        top_bar = ttk.Frame(self.root)
        top_bar.pack(fill=tk.X, padx=10, pady=8)

        ttk.Button(top_bar, text="Validate", command=self.validate_only).pack(side=tk.LEFT, padx=4)
        ttk.Button(top_bar, text="Start", command=self.start_runtime).pack(side=tk.LEFT, padx=4)
        ttk.Button(top_bar, text="Close", command=self.root.destroy).pack(side=tk.LEFT, padx=4)

        self.stats_label.pack(fill=tk.X, padx=12, pady=(0, 6))
        self.validation_label.pack(fill=tk.X, padx=12, pady=(0, 8))

        container = ttk.Frame(self.root)
        container.pack(fill=tk.BOTH, expand=True, padx=10, pady=8)

        canvas = tk.Canvas(container, highlightthickness=0)
        scrollbar = ttk.Scrollbar(container, orient=tk.VERTICAL, command=canvas.yview)
        scroll_frame = ttk.Frame(canvas)

        scroll_frame.bind(
            "<Configure>",
            lambda _e: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        for i, key in enumerate(self.keys):
            value = getattr(config, key)
            ttk.Label(scroll_frame, text=key).grid(row=i, column=0, sticky="w", padx=6, pady=4)

            if isinstance(value, bool):
                var = tk.BooleanVar(value=value)
                cb = ttk.Checkbutton(
                    scroll_frame,
                    variable=var,
                    command=self._on_any_change,
                )
                cb.grid(row=i, column=1, sticky="w", padx=6, pady=4)
                self.vars[key] = var
            else:
                if isinstance(value, Path):
                    initial = str(value)
                elif isinstance(value, str):
                    initial = value
                else:
                    initial = repr(value)
                var = tk.StringVar(value=initial)
                entry = ttk.Entry(scroll_frame, textvariable=var, width=90)
                entry.grid(row=i, column=1, sticky="ew", padx=6, pady=4)
                var.trace_add("write", lambda *_args: self._on_any_change())
                self.vars[key] = var
                self.widgets[key] = entry

        scroll_frame.columnconfigure(1, weight=1)

    def _on_any_change(self) -> None:
        self.validation_label.configure(text="")
        self._refresh_stats()

    def _read_values(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key in self.keys:
            current = getattr(config, key)
            var = self.vars[key]
            if isinstance(var, tk.BooleanVar):
                out[key] = bool(var.get())
            else:
                raw = str(var.get())
                out[key] = parse_value(raw, current)
        return out

    def _apply_dependency_rules(self, values: dict[str, Any]) -> list[str]:
        notes: list[str] = []
        use_video_file = bool(values.get("USE_VIDEO_FILE", False))
        source_fps = float(values.get("VIDEO_TARGET_FPS", 0.0) if use_video_file else values.get("TARGET_VIDEO_FPS", 0.0))
        infer_every = max(1, int(values.get("INFER_EVERY_N_FRAMES", 1)))
        target_infer = float(values.get("TARGET_INFER_FPS", 1.0))
        if source_fps > 0:
            physical_cap = source_fps / infer_every
            if target_infer > physical_cap:
                values["TARGET_INFER_FPS"] = float(physical_cap)
                if isinstance(self.vars.get("TARGET_INFER_FPS"), tk.StringVar):
                    self.vars["TARGET_INFER_FPS"].set(f"{physical_cap:.3f}")
                notes.append(
                    f"TARGET_INFER_FPS auto-clamped to {physical_cap:.2f} (source_fps={source_fps:.2f}, INFER_EVERY_N_FRAMES={infer_every})."
                )
        return notes

    def _refresh_stats(self) -> None:
        try:
            values = self._read_values()
        except Exception:
            self.stats_label.configure(text="Derived stats: waiting for valid input…")
            return

        use_video_file = bool(values.get("USE_VIDEO_FILE", False))
        source_fps = float(values.get("VIDEO_TARGET_FPS", 0.0) if use_video_file else values.get("TARGET_VIDEO_FPS", 0.0))
        infer_every = max(1, int(values.get("INFER_EVERY_N_FRAMES", 1)))
        target_infer = float(values.get("TARGET_INFER_FPS", 0.0))
        frame_buf = max(1, int(values.get("MAX_FRAME_BUFFER", 1)))
        display_buf_target = float(values.get("DISPLAY_BUFFER_MS", 0.0))

        physical_cap = source_fps / infer_every if source_fps > 0 else 0.0
        effective_infer_fps = min(target_infer, physical_cap) if physical_cap > 0 else target_infer
        display_capacity_ms = (frame_buf / source_fps) * 1000.0 if source_fps > 0 else math.inf

        self.stats_label.configure(
            text=(
                f"Derived: source_fps={source_fps:.2f}, physical_infer_cap={physical_cap:.2f}, "
                f"effective_target_infer_fps={effective_infer_fps:.2f}, "
                f"display_buffer_target={display_buf_target:.1f}ms, "
                f"display_buffer_capacity≈{display_capacity_ms:.1f}ms"
            )
        )

    def _apply_to_runtime_config(self, values: dict[str, Any]) -> None:
        for key, value in values.items():
            setattr(config, key, value)

    def validate_only(self) -> None:
        try:
            values = self._read_values()
            notes = self._apply_dependency_rules(values)
            self._apply_to_runtime_config(values)
            rt_dlc_obs.validate_runtime_config()
        except Exception as exc:
            self.validation_label.configure(text=f"Validation error: {exc}")
            return

        msg = "Validation OK."
        if notes:
            msg += " " + " ".join(notes)
        self.validation_label.configure(text=msg, foreground="#1b7f3b")

    def start_runtime(self) -> None:
        try:
            values = self._read_values()
            notes = self._apply_dependency_rules(values)
            self._apply_to_runtime_config(values)
            rt_dlc_obs.validate_runtime_config()
        except Exception as exc:
            messagebox.showerror("Validation error", str(exc))
            return

        if notes:
            messagebox.showinfo("Auto-adjusted dependencies", "\n".join(notes))
        self.root.destroy()
        rt_dlc_obs.main()

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    gui = ConfigGUI()
    gui.run()


if __name__ == "__main__":
    main()
