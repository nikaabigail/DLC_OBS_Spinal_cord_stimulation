🧠 Real-time DLC Pipeline (Spinal Cord / BCI)
📌 Overview

Проект реализует реалтайм-пайплайн оценки позы (pose estimation) для задач:

gait analysis
spinal cord stimulation
closed-loop neurofeedback
BCI / neuroengineering
🏗 Архитектура
Старый пайплайн
rt_dlc_obs.py
кастомный inference runner
сложный admission control
skip frames
буферизация
высокая задержка
Новый пайплайн (экспериментальный)
rt_dlc_live.py
DLCLive backend
прямой inference
минимальная задержка
упрощённая архитектура
⚙️ Окружения
🔹 Основное (DLC 3)
dlc_win_env
deeplabcut 3.0.0rc13
torch 2.10.0 + CUDA 12.8
numpy 1.26.4
opencv-python
🔹 DLCLive окружение
dlc_live_env
Установленные пакеты:
pip install "numpy<2"
pip install opencv-python==4.11.0.86
pip install deeplabcut-live[pytorch] --no-deps
pip install colorcet
📦 Важные зависимости
пакет	версия
torch	2.10.0+cu128
numpy	1.26.x
opencv-python	4.11.x
deeplabcut-live	1.1.0
🧠 Модель

Путь:

C:\dlc\project\r_tm_side-og-2024-10-25\exported-models-pytorch\

Используется:

*.pt snapshot (best)
▶️ Запуск
conda activate dlc_live_env

python rt_dlc_live.py
📊 Текущие метрики
inference time: ~20 ms
dlc fps: 25–50
latency: минимальная
visible: ⚠ ~20% (нужно исправить)
⚠️ Известные проблемы
1. Низкая точность (visible)

Причины:

несовпадение bodyparts
порог confidence
отсутствие фильтрации
2. Нет фильтрации

В текущей версии отсутствуют:

median filter
despike
hold
3. Нет ROI

Используется full-frame inference

🚧 Roadmap
 debug output DLCLive
 восстановить фильтрацию
 сравнить с baseline (rt_dlc_obs)
 добавить causal smoothing
 интеграция с BCI pipeline
🧪 Использование

Подходит для:

real-time gait detection
neuroscience experiments
closed-loop stimulation
pose-based control systems