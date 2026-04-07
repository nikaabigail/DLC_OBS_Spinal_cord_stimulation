# DLC Live Integration History

Краткая история миграции realtime-инференса на DLCLive.

## Stage 1 — проблемы старого контура (`rt_dlc_obs.py`)

- заметная задержка между кадром и предсказанием;
- высокий `skip_rate`;
- нестабильный реальный infer FPS;
- сильная зависимость от admission-control параметров.

**Вывод:** для low-latency closed-loop сценария требовался более простой inference backend.

## Stage 2 — попытка исправить задержку буферизацией

Добавлялись:

- display/inference буфер,
- метрики синхронизации (`frame_delta`, `pred_age_ms`).

**Результат:** визуально стабильнее, но задержка выросла.

## Stage 3 — переход на DLCLive

Принято решение заменить кастомный inference-контур на `deeplabcut-live`.

Причины:

- более прямой путь кадра к инференсу,
- меньше инфраструктурного кода,
- ниже end-to-end latency.

## Stage 4 — экспорт модели

Подготовлен exported model для DLCLive (`exported-models-pytorch/...`).

## Stage 5 — выделение отдельного окружения

Создано отдельное окружение `dlc_live_env` для изоляции зависимостей от основного DLC 3 окружения.

## Stage 6 — стабилизация зависимостей

Исправлены типичные проблемы:

- отсутствие `colorcet`,
- несовместимость с `numpy>=2`,
- `opencv-python-headless` без `imshow`.

## Stage 7 — минимальный runtime

Собран `rt_dlc_live.py`:

- без очередей и сложной skip-логики,
- inference на каждом кадре,
- простой overlay + базовые метрики FPS.

## Stage 8 — текущее состояние

- infer time: примерно `16–25 ms` (зависит от GPU/размера входа),
- dlc fps: обычно `25–50`,
- latency: низкая,
- открытый вопрос: низкий `visible` в части сессий.

## Next steps

1. проверить соответствие `BODY_PARTS` порядку в exported model;
2. сохранить raw output DLCLive для сравнения с `rt_dlc_obs.py`;
3. вернуть причинно-устойчивую фильтрацию (hold + median);
4. подобрать confidence порог и/или ROI под конкретный сетап.
