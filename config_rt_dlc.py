from pathlib import Path

# =========================
# Video / OBS
# =========================
CAM_INDEX = 2
USE_VIDEO_FILE = True
VIDEO_FILE_PATH = r"C:\dlc\videos\6_MER2-230-168U3C(FDE22070174)_20240604_170308.avi"
USE_DUAL_VIDEO_FILES = False
VIDEO_FILE_PATHS = [
    r"C:\dlc\videos\left.avi",
    r"C:\dlc\videos\right.avi",
]
# Производительность dual-режима (2 потока): ограничиваем инференс на поток,
# иначе 2x full-rate DLC приводит к сильному лагу.
DUAL_TARGET_INFER_FPS_PER_STREAM = 10.0
DUAL_STAGGER_INFER = True  # True: запускать инференс потоков по очереди (left/right)
DUAL_INFER_W = 1280
DUAL_INFER_H = 160
DUAL_ENABLE_EXTRAPOLATION = True
DUAL_EXTRAPOLATE_MAX_MS = 80.0
VIDEO_TARGET_FPS = 60.0
VIDEO_SKIP_IF_BEHIND = False  # False: не ускорять видео за счет дропа кадров
FRAME_W = 1920
FRAME_H = 1080  
TARGET_VIDEO_FPS = 100.0
SHOW_SCALE = 0.8

# Режим выполнения:
# - "visual": как сейчас, показываем окно cv2.imshow
# - "background": не показываем окно, обрабатываем поток в фоне
RUNTIME_MODE = "visual"

# Сохранение результирующего видео (с overlay)
SAVE_OUTPUT_VIDEO = True
OUTPUT_VIDEO_PATH = Path(r"C:\dlc\DLC_OBS_Spinal_cord_stimulation\rt_dlc_output.mp4")
OUTPUT_VIDEO_FPS = 0.0  # 0.0 -> брать FPS из источника (VIDEO_TARGET_FPS)
OUTPUT_VIDEO_CODEC = "mp4v"
# Для стабильной онлайн-оценки угла лучше не ждать движения:
# стартуем инференс сразу, иначе первые полезные кадры теряются.
AUTO_START_ON_MOTION = False
FRAME_DIFF_THRESHOLD = 0.5

SKIP_NEAR_DUPLICATE_FRAMES = False 
DUPLICATE_FRAME_THRESHOLD = 0.15  # средняя разница по grayscale для infer-frame
SUPPRESS_LOW_MOTION = False 
LOW_MOTION_THRESHOLD = 0.20 # средняя разница по grayscale для определения "низкого движения"

SHOW_FULL_FRAME = True
DISPLAY_BUFFER_MS = 20  # меньше буфер = меньше визуальная задержка (для closed-loop)
MAX_FRAME_BUFFER = 8 # макс кол-во кадров в буфере для отображения (на случай, если инференс отстает)
MAX_PRED_BUFFER = 8 # макс кол-во предсказаний в буфере (на случай, если инференс отстает, чтобы не держать слишком много старых предсказаний)
INFER_QUEUE_MAXSIZE = 1 # latest-only: не накапливать очередь старых кадров в инференс

# Размер кадра, который подаем в DLC
# Для боковой дорожки обычно разумно сначала резать ROI, потом уменьшать.
INFER_W = 1920
INFER_H = 220
INFER_EVERY_N_FRAMES = 2        # на входе 60 FPS дает целевой ритм около 30 запусков/с
TARGET_INFER_FPS = 30.0         # согласованный soft-limit для стабильного realtime без лишнего throttle

# ROI на исходном OBS-кадре (если нужно вырезать дорожку с мышью)
# Формат: x1, y1, x2, y2
USE_ROI = True
ROI = (0, 430, 1920, 649)
FORCE_FIXED_ROI = True          # основной режим для closed-loop: стабильная геометрия
AUTO_DETECT_CONTENT_ROI = False # оставлено выключенным: не использовать в основном runtime
CONTENT_BLACK_THRESH = 12       # пиксели темнее порога считаем "черным полем"
CONTENT_MIN_ROW_FILL = 0.08     # минимум заполнения строки не-черными пикселями
CONTENT_MIN_COL_FILL = 0.03     # минимум заполнения столбца не-черными пикселями

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
]
# Кандидаты для авто-определения стороны (по суммарной уверенности триплета).
SIDE_POINT_SETS = {
    "right": ("hl_hip_r", "hl_ankle_r", "hl_toes_r"),
    "left": ("hl_hip_l", "hl_ankle_l", "hl_toes_l"),
}

# Для отрисовки
CONF_THRESH_DRAW = 0.3

# =========================
# Online filtering
# =========================
CONF_THRESH_USE = 0.15
DESPIKE_THRESHOLD_PX = 220.0    # online threshold
MAX_HOLD_FRAMES = 12             # сколько кадров держим последнюю хорошую точку
MEDIAN_WINDOW = 3               # online median window по x/y
ENABLE_PCUTOFF = True
ENABLE_DESPIKE = True
ENABLE_HOLD = True
# После длительной окклюзии разрешаем "перезахват" точки даже при большом скачке
# относительно последней валидной позиции (иначе despike может блокировать возврат точки).
DESPIKE_RESET_GAP_FRAMES = 20

# Логи / диагностика
LOG_PATH = Path(r"C:\dlc\DLC_OBS_Spinal_cord_stimulation\rt_dlc_debug.log")
LOG_EVERY_N_FRAMES = 30
# Диагностика триплета (raw/filt/draw/reason) может быть дорогой при логировании каждого кадра.
# 0 = отключить периодический лог, >0 = писать раз в N кадров.
TRIPLET_LOG_EVERY_N_FRAMES = 30
# Дополнительно писать строку сразу при смене состояния (triplet -> no-triplet и обратно).
TRIPLET_LOG_ON_STATE_CHANGE = False
BENCHMARK_CSV_PATH = Path(r"C:\dlc\DLC_OBS_Spinal_cord_stimulation\rt_dlc_benchmark.csv")
ENABLE_BENCHMARK_LOG_ROW = True

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
DEBUG_OVERLAY = False  # False: рабочий overlay (только точки/угол), True: диагностический текст

# Политика устаревших предсказаний
# "drop" — не рисовать точки, если предсказание слишком старое;
# "show" — рисовать последнее доступное предсказание всегда.
STALE_PRED_POLICY = "drop"
STALE_PRED_MAX_MS = 50.0
OVERLAY_HOLD_MS = 120.0  # короткий latch последнего валидного overlay, чтобы убрать мигание

WINDOW_NAME = "OBS + DLC realtime"
