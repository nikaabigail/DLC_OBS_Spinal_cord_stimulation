# DLC Live Runtime (альтернативный realtime-пайплайн)

`rt_dlc_live.py` — упрощённая альтернатива `rt_dlc_obs.py` для low-latency инференса через `deeplabcut-live`.

## Что это даёт

- меньше задержка (без очередей и сложного throttle-пайплайна),
- проще отладка (один цикл capture → infer → draw),
- отдельный конфиг (`config_rt_dlc_live.py`), не мешает основному realtime-скрипту.

## Как запускать

```bash
conda activate dlc_live_env
python rt_dlc_live.py
```

## Зависимости окружения

Минимум:

- `numpy<2`
- `opencv-python`
- `deeplabcut-live[pytorch]`
- `colorcet`

Пример установки:

```bash
pip install "numpy<2" opencv-python==4.11.0.86 colorcet
pip install deeplabcut-live[pytorch] --no-deps
```

## Настройка

Основные параметры находятся в `config_rt_dlc_live.py`:

- источник кадров: `USE_VIDEO_FILE`, `VIDEO_FILE_PATH`, `CAM_INDEX`;
- модель: `MODEL_PATH`, `MODEL_TYPE`, `BODY_PARTS`;
- геометрия инференса: `USE_ROI`, `ROI`, `INFER_W`, `INFER_H`;
- отрисовка и логирование: `CONF_THRESH_DRAW`, `SHOW_SCALE`, `LOG_EVERY_N_FRAMES`.
- детальная диагностика: `LOG_STAGE_TIMINGS`, `LOG_VISIBLE_BREAKDOWN`, `LOG_FRAME_SYNC`, `LOG_DROP_EVENTS`.
- анти-замедление видеофайла: `VIDEO_SKIP_IF_BEHIND` и `MAX_CATCHUP_DROPS_PER_READ`.

Поддерживается задание путей через переменные окружения:

- `DLC_LIVE_VIDEO_PATH`
- `DLC_LIVE_ALT_VIDEO_PATH`
- `DLC_LIVE_MODEL_PATH`

Это позволяет не коммитить локальные machine-specific пути в Git.

## Текущее ограничение

Скрипт intentionally минималистичный: без online-фильтрации (median/despike/hold) из `rt_dlc_obs.py`.
Если нужна максимальная стабильность точек, используйте `rt_dlc_obs.py`.

Если видео визуально идёт медленнее реального времени, включите:

- `VIDEO_SKIP_IF_BEHIND = True`
- при необходимости увеличьте `MAX_CATCHUP_DROPS_PER_READ`.
