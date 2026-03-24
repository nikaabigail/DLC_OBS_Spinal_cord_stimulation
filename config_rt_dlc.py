from pathlib import Path

# =========================
# Video / OBS
# =========================
CAM_INDEX = 2
FRAME_W = 1920
FRAME_H = 1080  
TARGET_VIDEO_FPS = 100.0
SHOW_SCALE = 0.8
AUTO_START_ON_MOTION = True
FRAME_DIFF_THRESHOLD = 0.5

SKIP_NEAR_DUPLICATE_FRAMES = False
DUPLICATE_FRAME_THRESHOLD = 0.15  # средняя разница по grayscale для infer-frame

SHOW_FULL_FRAME = True
DISPLAY_DELAY_MS = 120          # задержка вывода для более точного совпадения frame->keypoints

# Размер кадра, который подаем в DLC
# Для боковой дорожки обычно разумно сначала резать ROI, потом уменьшать.
INFER_W = 640
INFER_H = 160
INFER_QUEUE_MAXSIZE = 4         # очередь кадров в async инференсе (держим маленькой против лага)
MAX_RESULT_HISTORY = 300        # сколько результатов хранить для точного frame matching

# ROI на исходном OBS-кадре (если нужно вырезать дорожку с мышью)
# Формат: x1, y1, x2, y2
USE_ROI = True
ROI = (0, 430, 1920, 649)
AUTO_DETECT_CONTENT_ROI = True  # авто-вырезка непустой области (убирает черные поля)
CONTENT_BLACK_THRESH = 12       # пиксели темнее порога считаем "черным полем"

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
CONF_THRESH_DRAW = 0.3

# =========================
# Online filtering
# =========================
CONF_THRESH_USE = 0.3
DESPIKE_THRESHOLD_PX = 140.0    # online threshold
MAX_HOLD_FRAMES = 6             # сколько кадров держим последнюю хорошую точку
MEDIAN_WINDOW = 3               # online median window по x/y

# Логи / диагностика
LOG_PATH = Path(r"C:\dlc\DLC_OBS_Spinal_cord_stimulation\rt_dlc_debug.log")
LOG_EVERY_N_FRAMES = 30

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
