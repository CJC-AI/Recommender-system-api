# 🚀 Recommender System API

Production-ready **two-stage recommender system** (Item-Item Collaborative Filtering + Learning-to-Rank) built on the RetailRocket dataset and deployed as a FastAPI microservice on Render.

The system is designed to handle:

* Extreme data sparsity
* Cold-start users
* Long-tail item distributions
* Strict cloud memory constraints (512MB free tier)

**Live API:**
`https://recommender-system-api-6dqc.onrender.com`

---

## 🔗 Quick Access

* **Swagger Docs:**
  [https://recommender-system-api-6dqc.onrender.com/docs](https://recommender-system-api-6dqc.onrender.com/docs)

* **Health Check:**
  [https://recommender-system-api-6dqc.onrender.com/health](https://recommender-system-api-6dqc.onrender.com/health)

---

# 🏗️ System Architecture

This is a **two-stage hybrid recommender system**:

### 1️⃣ Candidate Generation (Retrieval Layer)

* Item–Item Collaborative Filtering (cosine similarity)
* Taxonomy-aware backfill
* Popularity fallback for cold users
* Configurable candidate pool size

### 2️⃣ Learning-to-Rank (Ranking Layer)

* Logistic Regression ranker
* Six engineered features:

  * Similarity score
  * Item popularity
  * Interaction count
  * User history size
  * Category affinity
  * Time since last interaction

---

# 🧠 Memory-Constrained Deployment Strategy

Running a recommender system on a 512MB cloud instance requires careful architectural decisions.

Loading raw interaction data and building similarity matrices at startup exceeds memory limits.

### Solution: Decouple Heavy Computation from Inference

The system follows a two-phase lifecycle:

---

## Phase 1 — Offline Pre-computation

Script: `app/export_artifacts.py`

Executed locally or in CI/CD:

* Loads truncated interaction data (latest 250k events)
* Builds similarity matrix
* Constructs user histories
* Generates category profiles
* Serializes lightweight dictionaries using `joblib` (compressed)

Artifacts are saved to:

```
artifacts/api/
```

These are optimized for fast loading and minimal memory footprint.

---

## Phase 2 — Online Inference (FastAPI)

File: `app/main.py`

At startup:

* Loads pre-computed artifacts
* Loads trained ranker model
* Initializes context once using FastAPI lifespan

No heavy computation runs during inference.

Result:

* Fast boot time
* Low memory usage
* Stable performance on Render Free Tier

---

# 📂 Project Structure

```
RECOMMENDER-SYSTEM-API/
│
├── app/
│   ├── main.py
│   └── export_artifacts.py
│
├── artifacts/
│   ├── api/                   # Pre-computed lightweight dictionaries
│   └── models/
│       └── lr_ranker.joblib
│
├── data/
│   └── processed/
│       └── interactions.csv
│
├── src/                       # Core ML logic
│
└── requirements.txt
```

---

# 💻 Run Locally

### 1️⃣ Generate lightweight artifacts

```bash
python app/export_artifacts.py
```

Ensure this populates `artifacts/api/`.

---

### 2️⃣ Start the API

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

---

### 3️⃣ Open Swagger UI

```
http://127.0.0.1:8000/docs
```

---

# ☁️ Deployment (Render)

Configured for direct GitHub deployment.

### Render Settings

* **Environment:** Python 3
* **Build Command:**

  ```
  pip install -r requirements.txt
  ```
* **Start Command:**

  ```
  uvicorn app.main:app --host 0.0.0.0 --port $PORT
  ```
* **Instance Type:** Free (512MB RAM)

Important:

`artifacts/api/` must be committed so Render can load pre-computed structures.

---

# 📡 API Usage

---

## Health Check

**GET /health**

```bash
curl https://recommender-system-api-6dqc.onrender.com/health
```

**Response**

```json
{
  "status": "healthy"
}
```

---

## Generate Recommendations

**POST /recommend**

### Request Body

```json
{
  "user_id": 35,
  "k": 10
}
```

### Example (cURL)

```bash
curl -X POST \
  https://recommender-system-api-6dqc.onrender.com/recommend \
  -H "Content-Type: application/json" \
  -d '{"user_id": 35, "k": 10}'
```

### Example (Python)

```python
import requests

response = requests.post(
    "https://recommender-system-api-6dqc.onrender.com/recommend",
    json={"user_id": 35, "k": 10}
)

print(response.json())
```

### Response

```json
{
  "user_id": 35,
  "recommendations": [187946, 320130, 69533, 315545],
  "latency_ms": 12.48
}
```

---

# 📈 Engineering Highlights

* Two-stage recommender architecture
* Cold-start aware
* Retrieval bandwidth analysis
* Time-aware negative sampling
* Feature-engineered ranking layer
* Memory-optimized deployment
* Structured logging and latency middleware
* Fully containerizable (Docker-compatible)
* Production-ready FastAPI service

---

# 📌 Project Context

This repository represents the **deployment layer** of a full RetailRocket recommender system pipeline, including:

* Data preprocessing
* Candidate generation
* Ranking model training
* Offline evaluation
* Production inference

---