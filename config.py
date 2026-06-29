"""
config.py — Centralized configuration for the Quality Control System.
All tuneable parameters live here; override via environment variables.
"""
import os
from pathlib import Path

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
MODEL_PATH  = os.getenv("MODEL_PATH", str(BASE_DIR / "weights" / "best.pt"))

# ─── Detection ───────────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = float(os.getenv("CONF_THRESH", 0.5))
IMG_SIZE             = int(os.getenv("IMG_SIZE", 640))
LINE_POSITION        = float(os.getenv("LINE_POS", 0.15))   # 15 % from top
DEVICE               = os.getenv("DEVICE", "cuda")           # "cuda" | "cpu"

# ─── Classes ─────────────────────────────────────────────────────────────────
CLASS_NAMES = ["good", "paper_defect", "wrap_defect"]

CLASS_COLORS_BGR = {          # Used by OpenCV overlays
    "good":         (50,  205,  50),
    "paper_defect": (40,   40, 220),
    "wrap_defect":  (0,   165, 255),
}

CLASS_COLORS_HEX = {          # Used by Plotly / Streamlit
    "good":         "#00cc44",
    "paper_defect": "#ff3333",
    "wrap_defect":  "#ff8800",
}

# ─── Camera ──────────────────────────────────────────────────────────────────
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", 0))

# ─── Database ────────────────────────────────────────────────────────────────
DB_CONFIG = {
    "host":     os.getenv("DB_HOST",     "localhost"),
    "port":     int(os.getenv("DB_PORT", 5432)),
    "dbname":   os.getenv("DB_NAME",     "quality_control"),
    "user":     os.getenv("DB_USER",     "postgres"),
    "password": os.getenv("DB_PASSWORD", "enter-password"),
}

# ─── UI ──────────────────────────────────────────────────────────────────────
CHART_REFRESH_FRAMES  = 30     # Re-draw Plotly chart every N frames
DEFECT_ALERT_RATE_PCT = 5.0    # Show red alert when defect rate exceeds this
