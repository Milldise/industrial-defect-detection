# 🏭 Industrial Quality Control System v1.0
### Toilet Paper Production Line · YOLO26 Nano · RTX 2050

---

## 📁 Project Structure

```
quality_control/
├── app.py               ← Streamlit entry-point
├── config.py            ← All tuneable parameters
├── database.py          ← PostgreSQL / psycopg2 layer
├── video_processor.py   ← YOLO tracking + line-crossing logic
├── charts.py            ← Plotly visualisation helpers
├── requirements.txt
├── .env.example         ← Copy → .env and fill credentials
└── weights/
    └── best.pt          ← Place your YOLO26 Nano weights here
```

---

## 🚀 Installation

### 1 · Python environment (Python 3.10 or 3.11 recommended)

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate
```

### 2 · PyTorch with CUDA 12.x (RTX 2050 → CUDA 12.1)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

> **Note:** choose the correct CUDA version for your driver.
> Check with `nvidia-smi` → look at "CUDA Version" in the top-right corner.
> Other index URLs:
> * CUDA 11.8 → `https://download.pytorch.org/whl/cu118`
> * CPU only  → `https://download.pytorch.org/whl/cpu`

### 3 · All other dependencies

```bash
pip install -r requirements.txt
```

**Single-line copy-paste version:**

```bash
pip install streamlit>=1.35.0 ultralytics>=8.3.0 opencv-python-headless>=4.9 numpy>=1.26.0 pandas>=2.2.0 plotly>=5.22.0 psycopg2-binary>=2.9.9 python-dotenv>=1.0.0 Pillow>=10.3.0
```

---

## 🗄️ PostgreSQL Setup

### Option A — Docker (fastest)

```bash
docker run -d \
  --name qc_postgres \
  -e POSTGRES_DB=quality_control \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=postgres \
  -p 5432:5432 \
  postgres:16-alpine
```

### Option B — Native install (Windows)

1. Download and install from https://www.postgresql.org/download/windows/
2. During setup, set a password for the `postgres` user.
3. After install, open **pgAdmin** or **psql** and run:

```sql
CREATE DATABASE quality_control;
```

The application will create the `production_log` table automatically on first run.

---

## ⚙️ Configuration

```bash
cp .env.example .env
# Edit .env with your DB credentials and model path
```

Key variables:

| Variable       | Default          | Description                          |
|----------------|------------------|--------------------------------------|
| `MODEL_PATH`   | `weights/best.pt`| Path to YOLO26 Nano `.pt` file       |
| `DEVICE`       | `cuda`           | `cuda` or `cpu`                      |
| `CONF_THRESH`  | `0.50`           | Detection confidence threshold       |
| `DB_HOST`      | `localhost`      | PostgreSQL host                      |
| `DB_NAME`      | `quality_control`| Database name                        |
| `DB_PASSWORD`  | `postgres`       | Database password                    |

---

## ▶️ Running the App

```bash
streamlit run app.py
```

Open your browser at **http://localhost:8501**

---

## 🎯 Feature Overview

| Feature                        | Details                                          |
|-------------------------------|--------------------------------------------------|
| **Video sources**             | Real-time camera OR uploaded file (MP4/AVI/MOV) |
| **Model**                     | YOLO26 Nano, NMS-free, `.track(persist=True)`   |
| **Line-crossing logic**       | Bottom→Top movement, 15 % from top of frame     |
| **Classes**                   | `good`, `paper_defect`, `wrap_defect`           |
| **Database**                  | PostgreSQL · unique constraint on track_id       |
| **Charts**                    | Hourly bar, session donut, defect-rate gauge    |
| **Model loading**             | `@st.cache_resource` — loaded once per server   |
| **Performance (RTX 2050)**    | ~200 FPS inference · 4.7 ms/frame               |

---

## 🔧 Troubleshooting

**Model not found**
```
❌ Model weights not found at weights/best.pt
```
→ Set `MODEL_PATH` in `.env` to the absolute path of your `best.pt` file.

**CUDA not available**
```bash
python -c "import torch; print(torch.cuda.is_available())"
# Should print: True
```
→ Re-install PyTorch with the correct CUDA index URL (see Step 2).

**Database connection failed**
The app runs in **offline mode** — counters still work, DB charts show empty.
Check your `.env` credentials and ensure PostgreSQL is running:
```bash
pg_isready -h localhost -p 5432
```

**Camera not opening**
→ Try `CAMERA_INDEX=1` (or 2) in `.env` if index 0 doesn't work.
