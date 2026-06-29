import cv2
import time
import torch
import numpy as np
from ultralytics import YOLO


def run_benchmark(video_path, model_path, use_fp16, stride, max_video_sec=15):
    # Очистка кэша видеопамяти перед каждым тестом
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Загружаем модель
    model = YOLO(model_path)
    model.to(device)

    # "Прогрев" (Warm-up) видеокарты
    dummy_img = torch.zeros((1, 3, 640, 640), device=device, dtype=torch.float16 if use_fp16 else torch.float32)
    _ = model.predict(dummy_img, verbose=False)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Ошибка: не удалось открыть видео {video_path}")
        return

    frame_count = 0
    processed_frames = 0
    total_inference_time = 0.0

    # Запускаем общий таймер
    start_time = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ОГРАНИЧЕНИЕ ПО ВРЕМЕНИ: останавливаемся, если дошли до нужной секунды видео
        if cap.get(cv2.CAP_PROP_POS_MSEC) > (max_video_sec * 1000):
            break

        # Делаем ресайз до 640 по ширине
        h, w = frame.shape[:2]
        if w > 640:
            scale = 640 / w
            frame = cv2.resize(frame, (640, int(h * scale)), interpolation=cv2.INTER_LINEAR)

        # Выполняем инференс для каждого N-го кадра (stride)
        if frame_count % stride == 0:
            inf_start = time.perf_counter()

            _ = model.predict(frame, half=use_fp16, verbose=False, device=device)

            inf_end = time.perf_counter()
            total_inference_time += (inf_end - inf_start)
            processed_frames += 1

        frame_count += 1

    end_time = time.perf_counter()
    cap.release()

    total_time = end_time - start_time
    avg_fps = frame_count / total_time if total_time > 0 else 0
    avg_latency_ms = (total_inference_time / processed_frames) * 1000 if processed_frames > 0 else 0

    max_vram_mb = 0
    if device == 'cuda':
        max_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

    config_name = f"FP{'16' if use_fp16 else '32'}, stride={stride}"
    print(f"| {config_name:<16} | {avg_fps:>7.1f} | {max_vram_mb:>7.1f} | {avg_latency_ms:>7.1f} |")


if __name__ == "__main__":
    MODEL_FILE = r"C:\Users\milyukov\PycharmProjects\gamesGPT\DIPLOMA_PROJECT\weights\best.pt"

    VIDEO_FILES = [
        r"C:\Users\milyukov\Downloads\Telegram Desktop\video_2026-04-20_22-02-08.mp4",
        r"C:\Users\milyukov\Downloads\Telegram Desktop\IMG_4446.MOV"
    ]

    configs = [
        (False, 1),
        (False, 2),
        (True, 1),
        (True, 2),
        (True, 3)
    ]

    print("Запуск быстрого тестирования (15 секунд видео)...")

    for video_file in VIDEO_FILES:
        print(f"\n🎬 Тестируем видео: {video_file.split(chr(92))[-1]}")
        print("-" * 55)
        print(f"| {'Configuration':<16} | {'Avg FPS':>7} | {'VRAM MB':>7} | {'Latency ms':>10} |")
        print("-" * 55)

        for use_fp16, stride in configs:
            run_benchmark(video_file, MODEL_FILE, use_fp16, stride, max_video_sec=15)

        print("-" * 55)

    print("\n✅ Все тесты завершены!")