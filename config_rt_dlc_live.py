import os

# =========================
# Source
# =========================
CAM_INDEX = 2
USE_VIDEO_FILE = True

# Prefer env vars to avoid committing local absolute paths.
VIDEO_FILE_PATH = os.getenv(
    "DLC_LIVE_VIDEO_PATH",
    r"C:\dlc\videos\4_MER2-230-168U3C(FDE22070174)_20240604_164749.avi",
)
ALT_VIDEO_FILE_PATH = os.getenv(
    "DLC_LIVE_ALT_VIDEO_PATH",
    r"C:\dlc\videos\4_MER2-230-168U3C(FDE22070175)_20240604_164748.avi",
)

VIDEO_TARGET_FPS = 90.0
VIDEO_SKIP_IF_BEHIND = False

FRAME_W = 1920
FRAME_H = 1080
TARGET_VIDEO_FPS = 100.0

# =========================
# DLC-Live model
# =========================
# MODEL_PATH can be either:
# - exported model directory, or
# - a specific exported .pt file (current setup).
MODEL_PATH = os.getenv(
    "DLC_LIVE_MODEL_PATH",
    r"C:\dlc\project\r_tm_side-og-2024-10-25\exported-models-pytorch\DLC_r_tm_side_resnet_50_iteration-0_shuffle-5\DLC_r_tm_side_resnet_50_iteration-0_shuffle-5_snapshot-best-380.pt",
)
MODEL_TYPE = "pytorch"

BODY_PARTS = [
    "nose",
    "eye_l",
    "eye_r",
    "fl_toes_l",
    "fl_toes_r",
    "hl_toes_l",
    "hl_ankle_l",
    "hl_hip_l",
    "hl_iliac_l",
    "hl_toes_r",
    "hl_ankle_r",
    "hl_hip_r",
    "hl_iliac_r",
    "spine",
    "tail",
]

# =========================
# ROI / inference size
# =========================
USE_ROI = False
ROI = (0, 430, 1920, 649)

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

# Reserved for future extension.
DEVICE = "cuda"
