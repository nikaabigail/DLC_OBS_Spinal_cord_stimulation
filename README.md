# DLC OBS Spinal Cord Stimulation Toolkit

Набор утилит для:
- подготовки/валидации датасета DeepLabCut,
- обучения и анализа DLC-модели,
- realtime-инференса по потоку из OBS/камеры.

## Состав репозитория

- `run_dlc.py` — основной CLI для train/eval/analyze/metrics и сервисных операций по проекту DLC.
- `check_dlc_dataset.py` — проверка покрытия размеченного датасета.
- `check_dlc_shuffles.py` — проверка состояния шaфлов/снапшотов обучения.
- `rt_dlc_obs.py` — realtime-инференс DLC + оверлей в OpenCV окне.
- `config_rt_dlc.py` — runtime-конфиг для realtime режима (`rt_dlc_obs.py`).
- `rt_dlc_live.py` — альтернативный low-latency runtime через DLCLive.
- `config_rt_dlc_live.py` — конфиг для `rt_dlc_live.py`.
- `rt_dlc_config_gui.py` — графический runtime-конфигуратор (замена ручного редактирования `config_rt_dlc.py`) с валидацией и учетом зависимостей.
- `check_online_buffering.py` — утилита для многократной проверки/диагностики буферизации (симуляция + сводка benchmark CSV).
- `README_DLC_live.md` — отдельная документация по DLCLive режиму.

## Быстрый старт

### 0) Установка зависимостей

```bash
pip install -r requirements.txt
```

Если используется CUDA, `torch/torchvision` рекомендуется ставить отдельно под вашу CUDA-версию, например:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

Критичные нюансы:
- `numpy` должен быть `<2` (в проекте зафиксирован `numpy==1.26.4`) для совместимости с `deeplabcut==3.0.0rc13`.
- Не используйте `opencv-python-headless`, если нужен `cv2.imshow` в realtime-режиме.

### 1) Проверка датасета

```bash
python check_dlc_dataset.py
```

### 2) Проверка шaфлов/snapshots

```bash
python check_dlc_shuffles.py
```

### 3) Основной DLC workflow

```bash
python run_dlc.py check
python run_dlc.py summary
python run_dlc.py sync
python run_dlc.py dataset
python run_dlc.py train
python run_dlc.py evaluate
python run_dlc.py analyze
python run_dlc.py labeled
python run_dlc.py metrics
```

### 4) Realtime режим (основной)

```bash
python rt_dlc_obs.py
```

### 4.1) GUI-настройка runtime + запуск

```bash
python rt_dlc_config_gui.py
```

В GUI:
- меняются значения параметров из `config_rt_dlc.py`,
- автоматически считается производный cap (например, по `source_fps / INFER_EVERY_N_FRAMES`),
- при `Start` применяются значения, выполняется валидация, окно закрывается и стартует `rt_dlc_obs`.

### 5) Альтернатива: DLCLive runtime

```bash
python rt_dlc_live.py
```

Подробности и настройка: `README_DLC_live.md`.

### 6) Проверка буферизации (многократно)

```bash
python check_online_buffering.py simulate --buffers 0 20 40 80 120 --repeats 10
python check_online_buffering.py summarize-csv --csv C:\\dlc\\DLC_OBS_Spinal_cord_stimulation\\rt_dlc_benchmark.csv
```

Новый realtime-пайплайн работает **без OBS Virtual Camera**: источник кадров выбирается в `config_rt_dlc.py` через `USE_VIDEO_FILE`:
- `USE_VIDEO_FILE=False` → `CameraSource`
- `USE_VIDEO_FILE=True` → `VideoFileSource` (с pacing по `VIDEO_TARGET_FPS`)

Для потока 100 FPS в realtime:
- включен strict latest-frame режим без накопления буферов,
- для video source есть pacing (`VIDEO_TARGET_FPS`) и optional skip when behind (`VIDEO_SKIP_IF_BEHIND`),
- инференс контролируется admission-гейтами (`INFER_EVERY_N_FRAMES`, `TARGET_INFER_FPS`).
- для синхронной визуализации используется display-buffer (`DISPLAY_BUFFER_MS`) с match по `frame_id`.

Тюнинг для скорости/точности смотрите в `config_rt_dlc.py`:
- `INFER_W/INFER_H` (главный рычаг производительности),
- `SKIP_NEAR_DUPLICATE_FRAMES` и `DUPLICATE_FRAME_THRESHOLD`,
- `INFER_EVERY_N_FRAMES` и `TARGET_INFER_FPS`,
- `USE_ROI` и `ROI` (фиксированный рабочий crop),
- `DISPLAY_BUFFER_MS`, `MAX_FRAME_BUFFER`, `MAX_PRED_BUFFER`.
- для online-контроля буфера: `BUFFER_DIAG_EVERY_N_FRAMES`, `BUFFER_DIAG_WARN_MIN_SAMPLES`, `BUFFER_DIAG_RESET_AFTER_LOG`.

Логи realtime пишутся в:
- `C:\dlc\DLC_OBS_Spinal_cord_stimulation\rt_dlc_debug.log`

Каждые `LOG_EVERY_N_FRAMES` выводятся диагностические метрики:
- фактический FPS камеры (`cam_fps`) и DLC (`dlc_fps`),
- `raw_visible` vs `filtered_visible`,
- skip-счетчики по причинам (`skip_duplicate`, `skip_motion`, `skip_n`, `skip_fps`),
- тайминги стадий (`t_capture`, `t_pre`, `t_infer`, `t_post`, `t_draw`, `t_disp`).
- отдельная строка `buffer_diag` с контролем буферизации: `target`, `actual_mean`, `err_mean`, `abs_err_mean`, `%on_target`, `%exact_match`, `%no_pred`, `%stale_drop`, а также длины `infer_q/frame_buf/pred_buf`.
- в overlay рядом с `Hind angle` рисуется текущий статус буфера: `BUF <actual>/<target>ms OK|BAD delta_f=<...>`.

## Что проверено и отполировано

### 1) Убраны неиспользуемые сущности

- Удален `realtime_buses.py` (шины не использовались дальше по конвейеру).
- Из `rt_dlc_obs.py` убраны неиспользуемые `FramePacket/KeypointPacket/LatestBus`.

### 2) Убраны дубли и улучшена поддерживаемость

- В `run_dlc.py` добавлен единый helper `existing_video_paths()` вместо повторяющегося кода.
- Унифицировано получение размеченных `CollectedData_*.csv/.h5` через `find_collected_data_pair()`.

### 3) Исправлен потенциальный баг в сборке `video_sets`

Раньше учитывались только `CollectedData_og.csv/.h5`. Теперь поддерживается любой scorer (`CollectedData_<scorer>.csv/.h5`), если пара файлов существует.

### 4) Добавлены guard-проверки runtime-конфига

В `rt_dlc_obs.py` добавлена валидация критичных параметров:
- `INFER_W/INFER_H > 0`,
- `SHOW_SCALE > 0`,
- `DESPIKE_THRESHOLD_PX > 0`,
- `MAX_HOLD_FRAMES >= 0`,
- `MEDIAN_WINDOW > 0`.

### 5) Добавлена защита от некорректного ROI

Перед crop теперь проверяется, что ROI лежит в границах кадра (`0 <= x1 < x2 <= width`, `0 <= y1 < y2 <= height`).
Это предотвращает скрытые ошибки (`empty frame`, падение в resize/inference).

## Рекомендации по безопасности и качеству

- Абсолютные пути (`C:\dlc\...`) лучше вынести в переменные окружения или отдельный локальный конфиг, не коммитить машинно-зависимые значения.
- Для production стоит добавить логирование в файл (rotating logs), а не только `print`.
- Для realtime лучше добавить watchdog по FPS/latency и graceful fallback на CPU при недоступности CUDA.
- Для оффлайн-скриптов полезно ввести `argparse`, чтобы не редактировать код вручную под каждый запуск.

## Тестирование

Локально проверяйте минимум:

```bash
python -m compileall -q .
```

И дополнительно (если окружение готово):

```bash
python run_dlc.py check
python run_dlc.py summary
```

## Примечание

Скрипты ориентированы на Windows-пути и ваш текущий проект DLC. Перед запуском проверьте пути в `run_dlc.py` и `config_rt_dlc.py`.
