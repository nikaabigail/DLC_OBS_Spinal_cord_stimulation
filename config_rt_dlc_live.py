from pathlib import Path

# =========================
# Source
# =========================
CAM_INDEX = 2
USE_VIDEO_FILE = True

# Minimal single-stream config for rt_dlc_live.py.
# Pick one of your two side-view videos at a time.
VIDEO_FILE_PATH = r"C:\dlc\videos\4_MER2-230-168U3C(FDE22070174)_20240604_164749.avi"
# Alternative second stream:
ALT_VIDEO_FILE_PATH = r"C:\dlc\videos\4_MER2-230-168U3C(FDE22070175)_20240604_164748.avi"

VIDEO_TARGET_FPS = 90.0
VIDEO_SKIP_IF_BEHIND = False

FRAME_W = 1920
FRAME_H = 1080
TARGET_VIDEO_FPS = 100.0

# =========================
# DLC-Live model
# =========================
# IMPORTANT:
# This must point to an EXPORTED DLC-Live model directory,
# not to snapshot-best-380.pt directly.
MODEL_PATH = r"C:\dlc\project\r_tm_side-og-2024-10-25\exported-models-pytorch\DLC_r_tm_side_resnet_50_iteration-0_shuffle-5\DLC_r_tm_side_resnet_50_iteration-0_shuffle-5_snapshot-best-380.pt"
MODEL_TYPE = "pytorch"

# Optional bodypart names for labels/ordering in overlay.
BODY_PARTS = [
    "nose", "eye_l", "eye_r", "fl_toes_l", "fl_toes_r",
    "hl_toes_l", "hl_ankle_l", "hl_hip_l", "hl_iliac_l",
    "hl_toes_r", "hl_ankle_r", "hl_hip_r", "hl_iliac_r",
    "spine", "tail",
]

# =========================
# ROI / inference size
# =========================
# For your already-cropped 1920x220 side videos, keep USE_ROI=False.
# If you switch to full 1920x1080 source, set USE_ROI=True and use ROI below.
USE_ROI = False
ROI = (0, 430, 1920, 649)

# Working baseline from your recent tests.
INFER_W = 640
INFER_H = 160

# =========================
# Display / logging
# =========================
WINDOW_NAME = "DLC Live"
SHOW_SCALE = 0.8
SHOW_FULL_FRAME = True
CONF_THRESH_DRAW = 0.30

DRAW_NAMES = True
DRAW_CONF = False
LOG_EVERY_N_FRAMES = 30

# Not used directly by rt_dlc_live.py right now, but kept for clarity.
DEVICE = "cuda"
