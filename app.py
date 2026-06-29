"""
app.py — Industrial Quality Control Dashboard
=============================================
Streamlit front-end for real-time toilet-paper production monitoring.

Run:
    streamlit run app.py
"""
from __future__ import annotations

import os
import queue
import tempfile
import threading
import time
import uuid
from pathlib import Path

import datetime

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from streamlit.delta_generator import DeltaGenerator

from charts import build_defect_trend_chart, build_hourly_chart, build_session_donut
from config import (
    CAMERA_INDEX,
    CHART_REFRESH_FRAMES,
    CLASS_NAMES,
    DEFECT_ALERT_RATE_PCT,
    MODEL_PATH,
)
from database import DatabaseManager
from video_processor import CrossingEvent, ProductionTracker, load_model

# ─────────────────────────────────────────────────────────────────────────────
#  Page configuration  (must be the very first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="QC System · Paper Line",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
#  Global CSS
# ─────────────────────────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
    /* ── Root palette ─────────────────────────────────────────────── */
    :root {
        --c-bg:      #0e1117;
        --c-surface: #161b22;
        --c-border:  #30363d;
        --c-accent:  #00d4ff;
        --c-good:    #00cc44;
        --c-paper:   #ff4444;
        --c-wrap:    #ff8800;
        --c-text:    #e6edf3;
        --c-muted:   #8b949e;
    }

    /* ── Typography ───────────────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600;700&family=Rajdhani:wght@500;700&display=swap');
    html, body, [class*="css"] { font-family: 'JetBrains Mono', monospace; }

    /* ── Header ───────────────────────────────────────────────────── */
    .qc-header {
        background: linear-gradient(135deg, #0d1f2d 0%, #0e1117 60%, #1a1f2e 100%);
        border: 1px solid var(--c-border);
        border-radius: 12px;
        padding: 20px 32px;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        gap: 18px;
    }
    .qc-header h1 {
        font-family: 'Rajdhani', sans-serif;
        font-size: 2.1rem;
        font-weight: 700;
        color: var(--c-accent);
        margin: 0;
        letter-spacing: 2px;
    }
    .qc-header .sub {
        font-size: 0.78rem;
        color: var(--c-muted);
        letter-spacing: 1px;
        margin-top: 2px;
    }

    /* ── Metric badges ────────────────────────────────────────────── */
    .badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    .badge-online  { background:#0d3b1e; color:var(--c-good);  border:1px solid var(--c-good);  }
    .badge-offline { background:#3b0000; color:var(--c-paper); border:1px solid var(--c-paper); }
    .badge-warn    { background:#3b2a00; color:var(--c-wrap);  border:1px solid var(--c-wrap);  }

    /* ── Counter cards ────────────────────────────────────────────── */
    .counter-card {
        background: var(--c-surface);
        border: 1px solid var(--c-border);
        border-radius: 8px;
        padding: 10px 14px;
        margin-bottom: 8px;
        text-align: center;
    }
    .counter-card .label { font-size:0.68rem; color:var(--c-muted); letter-spacing:1px; }
    .counter-card .value { font-size:1.9rem; font-weight:700; }
    .counter-card.good   .value { color: var(--c-good);  }
    .counter-card.paper  .value { color: var(--c-paper); }
    .counter-card.wrap   .value { color: var(--c-wrap);  }

    /* ── Alert box ────────────────────────────────────────────────── */
    .defect-alert {
        background: #2d0b0b;
        border: 1px solid var(--c-paper);
        border-radius: 8px;
        padding: 10px 14px;
        font-size: 0.82rem;
        color: var(--c-paper);
        animation: flash 0.8s infinite alternate;
    }
    @keyframes flash { from { opacity:1; } to { opacity:0.55; } }

    /* ── Video column: constrain max width so vertical videos don't blow up ── */
    [data-testid="stImage"] img {
        max-height: 480px;
        width: auto !important;
        max-width: 100%;
        object-fit: contain;
        border-radius: 6px;
    }

    /* ── Hide Streamlit chrome ────────────────────────────────────── */
    #MainMenu, footer { visibility: hidden; }
    .block-container  { padding-top: 1.5rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
#  Cached singletons
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def get_db() -> DatabaseManager:
    return DatabaseManager()


@st.cache_resource(show_spinner=False)
def get_tracker() -> ProductionTracker:
    model = load_model(MODEL_PATH)
    return ProductionTracker(model)


# ─────────────────────────────────────────────────────────────────────────────
#  Session-state initialisation
# ─────────────────────────────────────────────────────────────────────────────

def _init_state() -> None:
    defaults = {
        "session_id":  str(uuid.uuid4())[:8].upper(),
        "running":     False,
        "frame_count": 0,
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


# ─────────────────────────────────────────────────────────────────────────────
#  Sidebar
# ─────────────────────────────────────────────────────────────────────────────

def render_sidebar(tracker: ProductionTracker, db: DatabaseManager) -> dict:
    """Render the sidebar and return a dict of user settings."""
    with st.sidebar:
        st.markdown("## QC Control Panel")
        st.caption(f"Session `{st.session_state.session_id}`")

        db_badge = (
            '<span class="badge badge-online">● DB ONLINE</span>'
            if db.is_connected
            else '<span class="badge badge-offline">✕ DB OFFLINE</span>'
        )
        st.markdown(db_badge, unsafe_allow_html=True)
        st.divider()

        # ── Video source ──────────────────────────────────────────────────
        st.subheader("📹 Video Source")
        source = st.radio(
            "Input type",
            ["📷 Real-time Camera", "📁 Upload Video File"],
            key="source_radio",
        )

        uploaded = None
        cam_idx  = CAMERA_INDEX

        if source == "📁 Upload Video File":
            uploaded = st.file_uploader(
                "Select MP4 / AVI / MOV / MKV",
                type=["mp4", "avi", "mov", "mkv"],
                key="video_file",
            )
        else:
            cam_idx = st.number_input(
                "Camera index", min_value=0, max_value=10,
                value=CAMERA_INDEX, step=1,
            )

        st.divider()

        # ── Controls ──────────────────────────────────────────────────────
        st.subheader("⚙️ Controls")
        c1, c2 = st.columns(2)
        with c1:
            start = st.button("▶ Start", type="primary", use_container_width=True)
        with c2:
            stop = st.button("■ Stop", use_container_width=True)

        if st.button("↺ Reset Session", use_container_width=True):
            tracker.reset()
            st.session_state.session_id = str(uuid.uuid4())[:8].upper()
            st.session_state.frame_count = 0
            st.rerun()

        st.divider()

        # ── Model settings ────────────────────────────────────────────────
        st.subheader("Detection Settings")
        conf = st.slider("Confidence threshold", 0.10, 0.95, 0.50, 0.05)

        # ── Performance settings ──────────────────────────────────────────
        st.subheader("⚡ Performance")
        infer_every = st.slider(
            "Run YOLO every N frames",
            min_value=1, max_value=6, value=2, step=1,
            help="1 = каждый кадр (медленно), 2-3 = баланс, 4-6 = быстро но реже детекция"
        )
        display_fps = st.slider(
            "Display FPS cap",
            min_value=5, max_value=30, value=15, step=5,
            help="Сколько кадров в секунду показывать в Streamlit (меньше = плавнее)"
        )
        inference_width = st.select_slider(
            "Inference resolution (width)",
            options=[320, 480, 640],
            value=640,
            help="320 = быстро (MOV/4K), 640 = точнее (MP4/низкое разрешение)"
        )

        st.divider()

        # ── DB date-range selector ────────────────────────────────────────
        st.subheader("Database Filter")

        today = datetime.date.today()
        yesterday = today - datetime.timedelta(days=1)

        date_range = st.date_input(
            "Select date range",
            value=(yesterday, today),
            max_value=today,
            key="db_date_filter"
        )

        if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
            selected_dates = date_range
        elif isinstance(date_range, (list, tuple)) and len(date_range) == 1:
            selected_dates = (date_range[0], date_range[0])
        else:
            selected_dates = (today, today)

        if db.is_connected:
            all_counts = db.get_all_time_counts()
            if all_counts:
                st.caption("All-time totals:")
                for cls in CLASS_NAMES:
                    st.caption(f"  {cls}: {all_counts.get(cls, 0)}")

        return {
            "source":          source,
            "uploaded":        uploaded,
            "cam_idx":         cam_idx,
            "conf":            conf,
            "date_range":      selected_dates,
            "start":           start,
            "stop":            stop,
            "infer_every":     infer_every,
            "display_fps":     display_fps,
            "inference_width": inference_width,
        }

# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _encode_jpeg(bgr_frame: np.ndarray, quality: int = 75, max_width: int = 480) -> bytes:
    """Encode BGR frame to JPEG bytes; downscale to max_width for fast display."""
    h, w = bgr_frame.shape[:2]
    if w > max_width:
        scale     = max_width / w
        bgr_frame = cv2.resize(
            bgr_frame, (max_width, int(h * scale)),
            interpolation=cv2.INTER_LINEAR,
        )
    ok, buf = cv2.imencode(".jpg", bgr_frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes() if ok else b""


# ─────────────────────────────────────────────────────────────────────────────
#  Background inference worker
# ─────────────────────────────────────────────────────────────────────────────

def _inference_worker(
    cap:             cv2.VideoCapture,
    tracker:         ProductionTracker,
    db:              DatabaseManager,
    conf:            float,
    session_id:      str,
    infer_every:     int,
    inference_width: int,
    result_queue:    queue.Queue,
    stop_event:      threading.Event,
) -> None:
    """
    Runs in a background daemon thread.
    Reads frames from `cap`, optionally skips inference, and pushes
    (annotated_frame, new_crossings, fps) into result_queue.

    Uses maxsize=2 on the queue — if Streamlit display is slow, old frames
    are simply discarded and only the latest is shown (no frame build-up).
    """
    frame_idx = 0

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            result_queue.put(None)   # Sentinel: video ended
            break

        # ── Resize for inference (GPU memory & speed) ──────────────────
        h, w = frame.shape[:2]
        if w > inference_width:
            scale     = inference_width / w
            frame_inf = cv2.resize(
                frame, (inference_width, int(h * scale)),
                interpolation=cv2.INTER_LINEAR,
            )
        else:
            frame_inf = frame

        # ── YOLO inference (every infer_every frames) ──────────────────
        if frame_idx % infer_every == 0:
            result = tracker.process_frame(frame_inf, conf=conf)

            # Persist to DB on background thread (non-blocking for display)
            for ev in result.new_crossings:
                db.log_detection(ev.class_name, ev.track_id, session_id)

            # Non-blocking put: drop if queue already has a pending frame.
            # This keeps display latency low — we never queue up stale frames.
            if result_queue.full():
                try:
                    result_queue.get_nowait()   # Discard oldest
                except queue.Empty:
                    pass
            try:
                result_queue.put_nowait(result)
            except queue.Full:
                pass

        frame_idx += 1


# ─────────────────────────────────────────────────────────────────────────────
#  Main processing loop  (display-side, runs in Streamlit thread)
# ─────────────────────────────────────────────────────────────────────────────

def run_processing_loop(
    cap:            cv2.VideoCapture,
    tracker:        ProductionTracker,
    db:             DatabaseManager,
    conf:           float,
    date_range:     tuple,
    infer_every:    int,
    display_fps:    int,
    inference_width: int,
    # UI placeholders
    frame_ph:       DeltaGenerator,
    status_ph:      DeltaGenerator,
    fps_ph:         DeltaGenerator,
    alert_ph:       DeltaGenerator,
    donut_ph:       DeltaGenerator,
    chart_ph:       DeltaGenerator,
    table_ph:       DeltaGenerator,
) -> None:
    """
    Display loop — pulls processed frames from the inference thread's queue
    and updates the Streamlit UI at a capped display FPS.

    The inference thread runs independently so the GPU never has to wait for
    Streamlit's re-render cycle.
    """
    # Shared state between threads
    result_queue = queue.Queue(maxsize=2)
    stop_event   = threading.Event()

    # Start inference in background
    worker = threading.Thread(
        target=_inference_worker,
        args=(
            cap, tracker, db, conf,
            st.session_state.session_id,
            infer_every, inference_width,
            result_queue, stop_event,
        ),
        daemon=True,
    )
    worker.start()

    # Initial chart render so the right panel isn't empty
    _refresh_charts(tracker, db, date_range, donut_ph, chart_ph, table_ph)

    display_interval = 1.0 / max(display_fps, 1)
    last_display_ts  = 0.0
    last_chart_ts    = time.perf_counter()
    chart_interval   = 5.0          # Refresh charts every 5 s (not every N frames)

    tmp_alert_clear  = 0
    frame_count      = 0

    try:
        while st.session_state.running:
            # ── Pull latest result from worker ─────────────────────────
            try:
                result = result_queue.get(timeout=0.08)
            except queue.Empty:
                # Worker is still processing — yield briefly and check stop
                time.sleep(0.01)
                continue

            if result is None:
                status_ph.info("✅ Video finished — press Reset to start a new session.")
                st.session_state.running = False
                break

            # ── Rate-limited display update ────────────────────────────
            now = time.perf_counter()
            if now - last_display_ts >= display_interval:
                jpeg_bytes = _encode_jpeg(result.annotated, quality=75, max_width=480)
                frame_ph.image(jpeg_bytes, use_container_width=False)
                fps_ph.metric("⚡ FPS", f"{result.fps:.1f}")
                last_display_ts = now

            # ── Defect alerts ──────────────────────────────────────────
            if result.new_crossings:
                defect_events = [e for e in result.new_crossings if e.class_name != "good"]
                if defect_events:
                    lines = "<br>".join(
                        f"⚠ {e.class_name.replace('_', ' ').upper()} · ID {e.track_id}"
                        for e in defect_events
                    )
                    alert_ph.markdown(
                        f'<div class="defect-alert"><b>DEFECT DETECTED</b><br>{lines}</div>',
                        unsafe_allow_html=True,
                    )
                    tmp_alert_clear = frame_count + 60
            elif frame_count >= tmp_alert_clear and tmp_alert_clear > 0:
                alert_ph.empty()

            # ── Chart refresh (time-based, not frame-based) ────────────
            if now - last_chart_ts >= chart_interval:
                _refresh_charts(tracker, db, date_range, donut_ph, chart_ph, table_ph)
                last_chart_ts = now

            frame_count += 1

    finally:
        # Always stop the background thread cleanly
        stop_event.set()
        worker.join(timeout=3.0)
        cap.release()


# ─────────────────────────────────────────────────────────────────────────────
#  Chart refresh helper
# ─────────────────────────────────────────────────────────────────────────────

def _refresh_charts(tracker, db, date_range, donut_ph, chart_ph, table_ph) -> None:
    rows = db.get_stats_by_date_range(date_range[0], date_range[1])

    chart_ph.plotly_chart(
        build_hourly_chart(rows),
        use_container_width=True,
        key=f"bar_{time.time()}",
    )
    donut_ph.plotly_chart(
        build_session_donut(tracker.session_counts),
        use_container_width=True,
        key=f"donut_{time.time()}",
    )

    events = db.get_recent_events(limit=15)
    if events:
        df = pd.DataFrame(events)[["timestamp", "class_name", "track_id", "session_id"]]
        df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.strftime("%H:%M:%S")
        df.columns = ["Time", "Class", "Track ID", "Session"]
        table_ph.dataframe(df, use_container_width=True, hide_index=True)
    else:
        table_ph.caption("No events logged yet.")


# ─────────────────────────────────────────────────────────────────────────────
#  Entry-point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    _init_state()
    db      = get_db()
    tracker = get_tracker()

    # ── Page header ───────────────────────────────────────────────────────
    st.markdown(
        """
        <div class="qc-header">
          <span style="font-size:2.5rem"></span>
          <div>
            <h1>INDUSTRIAL QUALITY CONTROL SYSTEM</h1>
            <div class="sub">TOILET PAPER PRODUCTION LINE · YOLO26 NANO · RTX 2050</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Sidebar ───────────────────────────────────────────────────────────
    cfg = render_sidebar(tracker, db)

    if cfg["start"]:
        st.session_state.running = True
    if cfg["stop"]:
        st.session_state.running = False

    # ─────────────────────────────────────────────────────────────────────
    #  MAIN LAYOUT: video (left) │ data & table (right)
    # ─────────────────────────────────────────────────────────────────────
    vid_col, data_col = st.columns([2, 3], gap="medium")

    with vid_col:
        st.markdown("#### Live Stream")
        frame_placeholder  = st.empty()
        status_placeholder = st.empty()

        fps_col, alert_col = st.columns([1, 2])
        with fps_col:
            fps_placeholder = st.empty()
        with alert_col:
            alert_placeholder = st.empty()

        st.markdown("**Session distribution**")
        donut_placeholder = st.empty()

    with data_col:
        st.markdown("#### Hourly Production")
        chart_placeholder = st.empty()

        st.markdown("---")
        st.markdown("#### Recent Events (DB)")
        table_placeholder = st.empty()

    # ── Initial static render (when idle) ────────────────────────────────
    if not st.session_state.running:
        frame_placeholder.markdown(
            """
            <div style="
                background:#161b22;
                border:1px dashed #30363d;
                border-radius:10px;
                height: 480px;
                width: 263px;
                display: flex;
                flex-direction: column;
                justify-content: center;
                align-items: center;
                text-align:center;
                color:#8b949e;
                ">
              <div style="font-size:3rem">📷</div>
              <div style="margin-top:12px;font-size:1rem; padding: 0 10px;">
                Select a video source<br>and press <b>▶ Start</b>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        _refresh_charts(
            tracker, db, cfg["date_range"],
            donut_placeholder, chart_placeholder, table_placeholder,
        )
        return

    # ── Validate model file ───────────────────────────────────────────────
    if not Path(MODEL_PATH).exists():
        st.error(
            f"❌ Model weights not found at `{MODEL_PATH}`.\n\n"
            "Set the `MODEL_PATH` environment variable to the correct path.",
            icon="🚨",
        )
        st.session_state.running = False
        return

    # ── Open video source ─────────────────────────────────────────────────
    cap      = None
    tmp_path = None

    try:
        if cfg["source"] == "📁 Upload Video File":
            if cfg["uploaded"] is None:
                status_placeholder.warning("⚠️ Please upload a video file first.")
                st.session_state.running = False
                return
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                tmp.write(cfg["uploaded"].read())
                tmp_path = tmp.name
            cap = cv2.VideoCapture(tmp_path)
            status_placeholder.success("📁 Processing uploaded video…")
        else:
            cap = cv2.VideoCapture(cfg["cam_idx"])
            status_placeholder.success(f"📷 Camera {cfg['cam_idx']} active…")

        if not cap or not cap.isOpened():
            st.error("❌ Could not open the selected video source.")
            st.session_state.running = False
            return

        # Set OpenCV buffer small so we always get fresh frames
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

        run_processing_loop(
            cap, tracker, db,
            conf            = cfg["conf"],
            date_range      = cfg["date_range"],
            infer_every     = cfg["infer_every"],
            display_fps     = cfg["display_fps"],
            inference_width = cfg["inference_width"],
            frame_ph        = frame_placeholder,
            status_ph       = status_placeholder,
            fps_ph          = fps_placeholder,
            alert_ph        = alert_placeholder,
            donut_ph        = donut_placeholder,
            chart_ph        = chart_placeholder,
            table_ph        = table_placeholder,
        )

    finally:
        if cap is not None:
            cap.release()
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


if __name__ == "__main__":
    main()