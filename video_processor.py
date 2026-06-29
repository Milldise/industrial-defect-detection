"""
video_processor.py — YOLO tracking + line-crossing counter.

Key design decisions
--------------------
* ``load_model()`` is decorated with ``@st.cache_resource`` so the weights
  are loaded exactly once per Streamlit server process — avoids VRAM churn.
* ``ProductionTracker`` is a plain Python class; Streamlit stores one instance
  in ``st.session_state`` so the tracker (and its in-memory state) survives
  across Streamlit re-runs.
* Direction of movement is BOTTOM → TOP (y decreasing).  An object is counted
  when its vertical centre crosses from below the counting line to above it in
  a single frame-to-frame transition.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO

from config import (
    CONFIDENCE_THRESHOLD,
    CLASS_COLORS_BGR,
    CLASS_NAMES,
    DEVICE,
    IMG_SIZE,
    LINE_POSITION,
    MODEL_PATH,
)


# ─────────────────────────────────────────────────────────────────────────────
#  Model loading — cached once for the entire Streamlit session
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="⏳ Loading YOLO model onto GPU…")
def load_model(model_path: str = MODEL_PATH) -> YOLO:
    """Load and warm-up the YOLO model.  Called once; cached thereafter."""
    model = YOLO(model_path)
    model.to(DEVICE)
    # Warm-up run to initialise CUDA kernels → lower latency on first real frame
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    model.predict(dummy, imgsz=IMG_SIZE, verbose=False, device=DEVICE)
    return model


# ─────────────────────────────────────────────────────────────────────────────
#  Data containers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CrossingEvent:
    track_id:   int
    class_name: str
    timestamp:  float = field(default_factory=time.time)


@dataclass
class FrameResult:
    annotated:      np.ndarray          # BGR frame ready for display
    new_crossings:  list[CrossingEvent] # Objects that just crossed the line
    fps:            float               # Instantaneous FPS for this frame


# ─────────────────────────────────────────────────────────────────────────────
#  Tracker
# ─────────────────────────────────────────────────────────────────────────────

class ProductionTracker:
    """Maintains per-session counting state across Streamlit re-runs.

    One instance is stored in ``st.session_state['tracker']``; it persists
    between Streamlit widget interactions so counts are never reset by accident.
    """

    def __init__(self, model: YOLO) -> None:
        self.model = model
        self.reset()
        self._fps_ts   = time.perf_counter()
        self._fps_frames = 0
        self._fps_val    = 0.0

    # ── Public API ─────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Clear all in-session counters and tracking memory."""
        self.session_counts: dict[str, int] = {k: 0 for k in CLASS_NAMES}
        self._tracked_ids:   set[int]  = set()   # IDs already counted
        self._prev_y:        dict[int, float] = {}  # last known y-centre per ID

    @property
    def total(self) -> int:
        return sum(self.session_counts.values())

    @property
    def defect_count(self) -> int:
        return self.session_counts["paper_defect"] + self.session_counts["wrap_defect"]

    @property
    def defect_rate_pct(self) -> float:
        return (self.defect_count / self.total * 100) if self.total else 0.0

    # ── Frame processing ───────────────────────────────────────────────────

    def process_frame(
        self,
        frame:     np.ndarray,
        conf:      float = CONFIDENCE_THRESHOLD,
        imgsz:     int   = IMG_SIZE,
    ) -> FrameResult:
        """Run tracking on *frame* and return an annotated copy + crossing events."""

        h, w = frame.shape[:2]
        line_y = int(h * LINE_POSITION)

        # ── Inference ──────────────────────────────────────────────────────
        results = self.model.track(
            frame,
            persist    = True,
            imgsz      = imgsz,
            conf       = conf,
            verbose    = False,
            device     = DEVICE,
        )

        new_crossings: list[CrossingEvent] = []

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids   = results[0].boxes.id.cpu().numpy().astype(int)
            clss  = results[0].boxes.cls.cpu().numpy().astype(int)

            for box, track_id, cls_idx in zip(boxes, ids, clss):
                class_name = self.model.names[cls_idx]
                y_centre   = float((box[1] + box[3]) / 2)

                # Retrieve last known centre (default to current so we don't
                # false-trigger on the very first frame the object appears).
                prev_y = self._prev_y.get(track_id, y_centre)

                # Crossing condition:
                #   • object was BELOW the line last frame  (prev_y >= line_y)
                #   • object is ABOVE the line this frame   (y_centre < line_y)
                #   • it hasn't been counted yet
                if (
                    prev_y   >= line_y
                    and y_centre < line_y
                    and track_id not in self._tracked_ids
                    and class_name in CLASS_NAMES
                ):
                    self._tracked_ids.add(track_id)
                    self.session_counts[class_name] += 1
                    new_crossings.append(CrossingEvent(int(track_id), class_name))

                self._prev_y[track_id] = y_centre

        # ── Build annotated frame ──────────────────────────────────────────
        annotated = results[0].plot()       # ultralytics draws boxes + labels
        annotated = self._draw_overlay(annotated, line_y, w)

        # ── FPS ────────────────────────────────────────────────────────────
        self._fps_frames += 1
        if self._fps_frames >= 10:
            elapsed          = time.perf_counter() - self._fps_ts
            self._fps_val    = self._fps_frames / max(elapsed, 1e-6)
            self._fps_ts     = time.perf_counter()
            self._fps_frames = 0

        return FrameResult(annotated, new_crossings, self._fps_val)

    # ── Private helpers ────────────────────────────────────────────────────

    def _draw_overlay(
            self,
            frame: np.ndarray,
            line_y: int,
            width: int,
    ) -> np.ndarray:
        """Draw an enlarged responsive counting line and HUD statistics block."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        counts = self.session_counts

        # Оставляем твой измененный коэффициент
        scale = width / 900.0

        # Динамическая толщина линий и текста (сделали чуть плотнее)
        thick_1 = max(1, int(1 * scale))
        thick_2 = max(2, int(2 * scale))
        thick_3 = max(2, int(4 * scale))  # Желтая линия станет толще и заметнее

        # ── Counting line ──────────────────────────────────────────────────
        cv2.line(frame, (0, line_y), (width, line_y), (0, 255, 255), thick_3)
        cv2.putText(
            frame, "▲  COUNTING LINE  ▲",
            (width // 2 - int(180 * scale), line_y - int(15 * scale)),
            font, 0.75 * scale, (0, 255, 255), thick_2, cv2.LINE_AA,
        )

        # ── Semi-transparent HUD box ───────────────────────────────────────
        overlay = frame.copy()

        # КРИТИЧЕСКИ УВЕЛИЧИВАЕМ РАЗМЕРЫ ПЛАШКИ
        box_x1 = int(20 * scale)
        box_y1 = int(20 * scale)
        box_x2 = int(520 * scale)  # Было 380 -> коробка стала намного шире
        box_y2 = int(260 * scale)  # Было 185 -> коробка стала намного выше

        cv2.rectangle(overlay, (box_x1, box_y1), (box_x2, box_y2), (10, 10, 10), -1)
        cv2.rectangle(overlay, (box_x1, box_y1), (box_x2, box_y2), (70, 70, 70), thick_2)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # Заголовок (Увеличили базовый масштаб с 0.60 до 0.85)
        cv2.putText(frame, "QUALITY CONTROL  v1.0",
                    (int(35 * scale), int(60 * scale)), font, 0.85 * scale, (190, 190, 190), thick_2, cv2.LINE_AA)

        # Текст счетчиков (Увеличили масштаб букв с 0.72 до 1.05)
        items = [
            ("GOOD", counts["good"], CLASS_COLORS_BGR["good"]),
            ("PAPER DEFECT", counts["paper_defect"], CLASS_COLORS_BGR["paper_defect"]),
            ("WRAP DEFECT", counts["wrap_defect"], CLASS_COLORS_BGR["wrap_defect"]),
        ]
        for i, (label, val, color) in enumerate(items):
            # Раздвинули строки: начальный Y теперь 115 вместо 78, шаг 50 вместо 34
            y = int((115 + i * 50) * scale)
            text = f"{label}: {val}"
            cv2.putText(frame, text,
                        (int(35 * scale), y), font, 1.05 * scale, color, thick_2, cv2.LINE_AA)

        # Индикатор статуса (Кружок в правом верхнем углу кадра тоже пропорционально увеличили)
        dot_color = (50, 205, 50) if self.defect_count == 0 else (40, 40, 210)
        dot_x = width - int(40 * scale)
        dot_y = int(40 * scale)
        cv2.circle(frame, (dot_x, dot_y), int(20 * scale), dot_color, -1, cv2.LINE_AA)

        status_label = "OK" if self.defect_count == 0 else "!!"
        text_x = width - int(52 * scale)
        text_y = int(48 * scale)
        cv2.putText(frame, status_label,
                    (text_x, text_y), font, 0.70 * scale, (255, 255, 255), thick_2, cv2.LINE_AA)

        return frame