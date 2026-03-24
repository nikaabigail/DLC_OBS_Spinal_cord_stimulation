from pathlib import Path

# =========================
# Video / OBS
# =========================
CAM_INDEX = 2
FRAME_W = 1920
FRAME_H = 1080  
TARGET_VIDEO_FPS = 60.0
SHOW_SCALE = 0.8
AUTO_START_ON_MOTION = True
FRAME_DIFF_THRESHOLD = 0.5

SKIP_NEAR_DUPLICATE_FRAMES = True
DUPLICATE_FRAME_THRESHOLD = 0.15  # средняя разница по grayscale для infer-frame

SHOW_FULL_FRAME = True

# Размер кадра, который подаем в DLC
# Для боковой дорожки обычно разумно сначала резать ROI, потом уменьшать.
INFER_W = 960
INFER_H = 220

# ROI на исходном OBS-кадре (если нужно вырезать дорожку с мышью)
# Формат: x1, y1, x2, y2
USE_ROI = False
ROI = (0, 430, 1920, 649)

# =========================
# DLC
# =========================
DLC_CONFIG_PATH = Path(r"C:\dlc\project\r_tm_side-og-2024-10-25\config.yaml")
DLC_SHUFFLE = 5
DLC_SNAPSHOT = Path(
    r"C:\dlc\project\r_tm_side-og-2024-10-25\dlc-models-pytorch\iteration-0"
    r"\r_tm_sideOct25-trainset95shuffle5\train\snapshot-best-380.pt"
)
DLC_PYTORCH_CFG = Path(
    r"C:\dlc\project\r_tm_side-og-2024-10-25\dlc-models-pytorch\iteration-0"
    r"\r_tm_sideOct25-trainset95shuffle5\train\pytorch_config.yaml"
)

DEVICE = "cuda"

# Какие точки реально используем в online-режиме.
# Для боковой съемки лучше сначала брать ОДНУ видимую сторону.
USE_POINTS = [
    "hl_hip_r",
    "hl_ankle_r",
    "hl_toes_r",
    "spine",
    "tail",
    "nose",
]

# Для отрисовки
CONF_THRESH_DRAW = 0.6

# =========================
# Online filtering
# =========================
CONF_THRESH_USE = 0.6
DESPIKE_THRESHOLD_PX = 80.0     # online threshold
MAX_HOLD_FRAMES = 3             # сколько кадров держим последнюю хорошую точку
MEDIAN_WINDOW = 5               # online median window по x/y

# =========================
# Feature extraction
# =========================
COMPUTE_HIND_ANGLE = True
HIND_ANGLE_POINTS = ("hl_hip_r", "hl_ankle_r", "hl_toes_r")

# =========================
# Overlay
# =========================
DRAW_POINTS = True
DRAW_NAMES = True
DRAW_CONF = False
DRAW_HIND_ANGLE = True
DRAW_FPS = True

WINDOW_NAME = "OBS + DLC realtime"