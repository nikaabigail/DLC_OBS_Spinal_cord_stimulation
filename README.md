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
- `config_rt_dlc.py` — runtime-конфиг для realtime режима.

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

### 4) Realtime режим

```bash
python rt_dlc_obs.py
```

Для потока 100 FPS в realtime:
- включен асинхронный конвейер (capture/display и inference развязаны),
- используется маленькая очередь инференса (`INFER_QUEUE_MAXSIZE`) с anti-backlog стратегией (берется самый свежий кадр),
- можно включать небольшую задержку вывода (`DISPLAY_DELAY_MS`), чтобы keypoints визуально совпадали с кадром.

Тюнинг для скорости/точности смотрите в `config_rt_dlc.py`:
- `INFER_W/INFER_H` (главный рычаг производительности),
- `DISPLAY_DELAY_MS`,
- `SKIP_NEAR_DUPLICATE_FRAMES` и `DUPLICATE_FRAME_THRESHOLD`,
- `INFER_QUEUE_MAXSIZE`.

Логи realtime пишутся в:
- `C:\dlc\DLC_OBS_Spinal_cord_stimulation\rt_dlc_debug.log`

Каждые `LOG_EVERY_N_FRAMES` выводятся диагностические метрики:
- фактический FPS камеры (`cam_fps`) и DLC (`dlc_fps`),
- размер очереди инференса (`q`),
- доля кадров с exact/fallback/empty соответствием,
- процент «видимых» точек (`visible`).

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
