🚀 DLC Live Integration — Development History
📍 Context

Цель: заменить кастомный realtime-пайплайн (rt_dlc_obs.py) на более стабильный и низколатентный inference через DeepLabCut Live (DLCLive).

🔹 Stage 1 — Проблемы исходного пайплайна

Выявленные проблемы в rt_dlc_obs.py:

❌ Задержка предсказаний (lag между кадром и точками)
❌ Высокий skip_rate (~70%)
❌ Нестабильный FPS инференса
❌ Сильная зависимость от admission control (skip_n / skip_fps)
❌ Не realtime-архитектура (single loop + throttling)

Результат:
→ Пайплайн не подходит для low-latency BCI / closed-loop задач

🔹 Stage 2 — Введение буферизации

Реализовано:

📦 Буфер кадров (~100 ms)
🎯 Синхронизация display ↔ inference
📊 Метрики:
display_frame_id
pred_frame_id
frame_delta
pred_age_ms

Результат:

✔ Визуально smoother
❌ Но latency вырос (~100–120 ms)
❌ Не решает проблему skip_rate
🔹 Stage 3 — Переход на DLC Live

Принято решение:
→ Использовать DLCLive как backend для inference

Причины:

минимальная задержка
отсутствие skip-пайплайна
оптимизированный inference runner
🔹 Stage 4 — Экспорт модели

Выполнено:

deeplabcut.export_model(...)

Результат:

exported-models-pytorch/
  DLC_r_tm_side_resnet_50_iteration-0_shuffle-5/
    *.pt

✔ Модель готова для DLCLive

🔹 Stage 5 — Новое окружение

Создано отдельное окружение:

dlc_live_env

Причина:

изоляция от DLC 3 (TensorFlow/PyTorch конфликты)
контроль зависимостей
🔹 Stage 6 — Установка DLCLive
pip install deeplabcut-live[pytorch] --no-deps
pip install colorcet

Фиксы:

❗ ModuleNotFoundError → colorcet
❗ numpy incompatibility → откат до <2
🔹 Stage 7 — Проблема OpenCV GUI

Ошибка:

cv2.imshow not implemented

Причина:

установлен opencv-python-headless

Решение:

pip install opencv-python
pip install "numpy<2"
🔹 Stage 8 — Минимальный rt_dlc_live.py

Создан новый пайплайн:

❌ без очередей
❌ без skip logic
❌ без фильтров
✔ прямой inference на каждом кадре
✔ DLCLive backend
🔹 Stage 9 — Текущие метрики
infer_time ≈ 16–25 ms
dlc_fps ≈ 25–50
cam_fps ≈ 25–35
latency ≈ минимальная (без буфера)
🔴 Текущая проблема
visible ≈ 20% (очень низко)

Возможные причины:

❗ mismatch BODY_PARTS
❗ другой scaling входа
❗ отличие output формата DLCLive
❗ слишком высокий CONF_THRESH_DRAW
❗ отсутствие фильтрации (noise)
🎯 Следующие шаги
 проверить порядок bodyparts
 логировать raw pose
 сравнить с rt_dlc_obs output
 добавить фильтр (hold + median)
 оптимизировать confidence threshold
 протестировать ROI снова