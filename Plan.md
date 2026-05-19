# Avatar_ML build guide — hybrid Windows + free Colab T4

> **What you're building:** a real-time cloned-voice talking-head system. Upload a voice sample + face video → get a reusable voice profile + avatar profile → type or speak text → see your avatar speak back in your cloned voice over WebRTC in the browser.
>
> **Why this guide differs from a generic "ML stack" plan:** you have no local GPU. So we put the **control plane on your Windows machine** (FastAPI, SQLite, filesystem, WebRTC) and the **data plane on a free Google Colab T4** (OpenVoice V2 + MuseTalk 1.5). The protocol between them stays the same when you later scale to paid GPU — no rewrite required.
>
> **Phase 0 cost:** $0. Phase 0 timeline: 5–6 focused days.

---

## Table of contents

1. [Outcome and architecture](#1-outcome-and-architecture)
2. [Prerequisites](#2-prerequisites)
3. [Repo layout](#3-repo-layout)
4. [Day-by-day build plan](#4-day-by-day-build-plan)
5. [File-by-file specifications](#5-file-by-file-specifications)
6. [Running the system end-to-end](#6-running-the-system-end-to-end)
7. [Verification checklist](#7-verification-checklist)
8. [What's deferred and when to come back](#8-whats-deferred-and-when-to-come-back)
9. [Scaling path to paid infra](#9-scaling-path-to-paid-infra)
10. [Troubleshooting](#10-troubleshooting)
11. [Appendix: why these choices](#11-appendix-why-these-choices)

---

## 1. Outcome and architecture

### What works after Phase 0

- `python scripts/create_voice_profile.py voice.wav` → reusable voice profile on disk.
- `python scripts/create_avatar_profile.py face.mp4` → reusable avatar profile (latents/masks cached).
- `python scripts/generate_video.py --voice <id> --avatar <id> --text "hello"` → `output.mp4` in your cloned voice with a lip-synced face.
- `uvicorn services.api.main:app` boots a FastAPI server. Open `tests/test_page.html` in Chrome, click "Connect", type text, see your avatar speak it over WebRTC. First synced frame target: **1.8–3.0 s**.

### Architecture diagram

```
+----------------------- Windows 11 (control plane, persistent) ---------------------+
|  FastAPI control service          tests/test_page.html  (browser, WebRTC peer)     |
|     |                                  ^                                           |
|     |  SDP offer/answer                |  RTP audio/video                          |
|     v                                  |                                           |
|  aiortc WebRTC peer  <----- asyncio.Queue ----  remote_stream_consumer.py          |
|  SQLite (profiles, sessions)       |                                               |
|  storage/  (voice samples, avatar caches, outputs)                                 |
|                                    |                                               |
|             colab_worker_client.py | (WSS + HTTPS over Cloudflare Tunnel)          |
+------------------------------------+-----------------------------------------------+
                                     |
                                     v
+----------- Google Colab T4 (data plane, ephemeral, ~12 hr sessions) ---------------+
|  notebooks/colab_inference_server.ipynb                                            |
|     FastAPI on :8000 + cloudflared quick-tunnel → *.trycloudflare.com              |
|     POST /voice/extract_embedding   (upload wav → embedding bytes)                 |
|     POST /avatar/preprocess         (upload mp4 → cache.tar.gz)                    |
|     POST /avatar/upload_cache       (Windows re-uploads cache after reconnect)     |
|     WS   /tts/stream                (text → PCM chunks)                            |
|     WS   /lipsync/stream            (PCM + avatar_id → JPEG frames + pts)          |
|     GET  /healthz                                                                  |
|     Models: OpenVoice V2 (~1.5 GB)  +  MuseTalk 1.5 (~5 GB)  fits T4 16 GB         |
+------------------------------------------------------------------------------------+
```

### Four design rules to keep

1. **Colab is stateless.** All durable state lives on Windows. After a Colab restart, Windows re-uploads the avatar cache before generating.
2. **One WebSocket per stream session** carries both video frames and PCM audio with timestamps. Windows owns WebRTC because Colab blocks the inbound UDP that STUN/TURN needs.
3. **The Colab URL is one env var** (`COLAB_INFERENCE_URL` in `.env`). Swapping to RunPod/Modal/k8s later is an env change, not a rewrite.
4. **No Docker, no Redis, no MinIO, no Celery in Phase 0.** SQLite + filesystem + FastAPI `BackgroundTasks` are enough. Add only when you measure a need.

---

## 2. Prerequisites

### Windows machine (your dev box)

Install once:

| Tool | Why | Install command (PowerShell, run as admin) |
|---|---|---|
| Python 3.11 (not 3.12 yet — aiortc wheels lag) | Control plane runtime | `winget install Python.Python.3.11` |
| `uv` (fast Python package manager) | Single-tool venv + lockfile | `winget install astral-sh.uv` |
| FFmpeg | Audio/video muxing | `winget install Gyan.FFmpeg` |
| Git | Source control | `winget install Git.Git` |
| VS Code (optional) | Editor with Python + Jupyter | `winget install Microsoft.VisualStudioCode` |

After install, **restart PowerShell** and verify:

```powershell
python --version    # Python 3.11.x
uv --version
ffmpeg -version
```

### Google account for Colab

- Sign in at <https://colab.research.google.com>.
- Set runtime to **T4 GPU**: `Runtime → Change runtime type → T4 GPU`.
- T4 gives 16 GB VRAM, free tier ~12 hr/day, sessions idle out at ~90 minutes.

### Sample inputs you'll need

Put these in `samples/` once you create the repo:

- `samples/voice.wav` — 10–60 seconds of clean speech (single speaker, 16 kHz+, mono preferred).
- `samples/face.mp4` — 10–60 seconds of a frontal face talking, stable lighting, no occlusion.

**Tip:** record both on your phone, send to yourself, drop into `samples/`. Quality matters more than length here.

---

## 3. Repo layout

Create this structure on Windows. Files marked `(stub)` are empty placeholders for now — they get filled in during the build plan.

```
Avatar_ML/
├── .env                              # COLAB_INFERENCE_URL, etc. NEVER commit.
├── .env.example                      # Committed template.
├── .gitignore                        # storage/, .venv/, .env, samples/*.wav, *.mp4
├── pyproject.toml                    # uv/pip dependencies for Windows side.
├── README.md                         # 30-line "how to run" cheat sheet.
├── Plan.md                           # ← this file
│
├── notebooks/
│   └── colab_inference_server.ipynb  # Single notebook = the entire Colab stack.
│
├── services/
│   ├── __init__.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py                   # FastAPI app, routers wired here.
│   │   ├── settings.py               # Pydantic Settings, loads .env.
│   │   ├── db.py                     # SQLite engine + minimal Profile/Session models.
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   ├── voice_profiles.py
│   │   │   ├── avatar_profiles.py
│   │   │   ├── sessions.py           # POST /api/sessions, GET /api/sessions/{id}
│   │   │   ├── webrtc.py             # POST /webrtc/offer → SDP answer
│   │   │   └── ws_sessions.py        # WS /ws/sessions/{id} control channel
│   │   ├── media/
│   │   │   ├── __init__.py
│   │   │   └── aiortc_track.py       # VideoStreamTrack + AudioStreamTrack from a Queue
│   │   └── remote_stream_consumer.py # WS client pulling frames+PCM from Colab
│   │
│   └── inference_client/
│       ├── __init__.py
│       └── colab_worker_client.py    # The "remote GPU" abstraction.
│
├── scripts/
│   ├── create_voice_profile.py
│   ├── create_avatar_profile.py
│   └── generate_video.py             # Batch MP4 end-to-end test (no streaming).
│
├── storage/                          # Created at runtime, gitignored.
│   ├── voices/<voice_id>/
│   │   ├── profile.json
│   │   ├── source.wav
│   │   └── embedding.npy
│   ├── avatars/<avatar_id>/
│   │   ├── profile.json
│   │   ├── source.mp4
│   │   └── cache.tar.gz              # Round-trips to Colab.
│   └── outputs/<run_id>/
│       └── output.mp4
│
├── samples/                          # User-supplied inputs, gitignored.
│   ├── voice.wav
│   └── face.mp4
│
├── tests/
│   └── test_page.html                # Vanilla JS RTCPeerConnection page.
│
└── data/
    └── avatar_ml.sqlite              # SQLite database, gitignored.
```

### `.env.example`

```dotenv
# Cloudflare Tunnel URL printed by the Colab notebook on startup.
# Update this every time you restart the Colab session.
COLAB_INFERENCE_URL=https://example-quick-tunnel.trycloudflare.com

# Local paths (rarely need to change).
STORAGE_DIR=./storage
DATABASE_URL=sqlite:///./data/avatar_ml.sqlite

# WebRTC ICE config. Public Google STUN is fine for loopback/LAN.
STUN_SERVERS=stun:stun.l.google.com:19302

# Network timeouts (seconds).
COLAB_HTTP_TIMEOUT=120
COLAB_WS_RECV_TIMEOUT=30
COLAB_HEALTHCHECK_INTERVAL=10
```

### `pyproject.toml` (Windows side only — Colab installs its own deps)

```toml
[project]
name = "avatar-ml"
version = "0.1.0"
requires-python = ">=3.11,<3.12"
dependencies = [
    "fastapi>=0.115",
    "uvicorn[standard]>=0.32",
    "pydantic>=2.9",
    "pydantic-settings>=2.6",
    "python-multipart>=0.0.12",   # FastAPI file uploads
    "httpx>=0.27",                # HTTP to Colab
    "websockets>=13",             # WS to Colab
    "aiortc>=1.9",                # WebRTC peer
    "av>=13",                     # PyAV for frame encoding
    "numpy>=1.26",
    "Pillow>=10.4",
    "sqlalchemy>=2.0",
    "aiosqlite>=0.20",
    "python-dotenv>=1.0",
]

[tool.uv]
# Lock to a known-good index.
```

### `.gitignore`

```
.venv/
__pycache__/
*.pyc
.env
storage/
data/
samples/*.wav
samples/*.mp4
samples/*.mov
tests/recorded/
```

### `README.md` (30-line cheat sheet)

```markdown
# Avatar_ML — hybrid Windows + Colab T4

## One-time setup
1. `winget install Python.Python.3.11 astral-sh.uv Gyan.FFmpeg`
2. `uv venv && .venv\Scripts\activate`
3. `uv pip install -e .`
4. `copy .env.example .env`
5. Open `notebooks/colab_inference_server.ipynb` in Colab (T4 runtime). Run all cells. Copy the printed `*.trycloudflare.com` URL into `.env` as `COLAB_INFERENCE_URL`.

## Run end-to-end (batch)
```
python scripts/create_voice_profile.py samples/voice.wav
python scripts/create_avatar_profile.py samples/face.mp4
python scripts/generate_video.py --voice <id> --avatar <id> --text "Hello world"
```

## Run streaming
```
uvicorn services.api.main:app --reload
```
Open `tests/test_page.html` in Chrome → click Connect → type text → watch avatar speak.

## When Colab disconnects
1. Re-run the Colab notebook. Copy the new URL.
2. Update `COLAB_INFERENCE_URL` in `.env`.
3. The FastAPI server reloads automatically (if started with `--reload`). Re-upload avatar cache via `python scripts/create_avatar_profile.py --rehydrate <id>`.
```

---

## 4. Day-by-day build plan

Targets are aggressive but realistic if each day is 3–5 focused hours.

### Day 1 — repo + Colab notebook skeleton

**Goal:** Colab is reachable from Windows over a Cloudflare Tunnel and returns `{"status":"ok"}` from `/healthz`.

Tasks:

1. Create the directory tree in [§3](#3-repo-layout).
2. Initialize git: `git init && git add . && git commit -m "scaffold"`.
3. `uv venv && .venv\Scripts\activate && uv pip install -e .`
4. Build the Colab notebook minimally — just FastAPI + `/healthz` + `cloudflared` tunnel. (Models come on Day 2.) See [§5.1](#51-notebookscolab_inference_serveripynb).
5. Write `services/inference_client/colab_worker_client.py` with **just** `health_check()` and `__init__(base_url)`. See [§5.2](#52-servicesinference_clientcolab_worker_clientpy).
6. Write a 10-line test script: `python -c "from services.inference_client.colab_worker_client import ColabWorkerClient; print(ColabWorkerClient.from_env().health_check())"`. Should print `True`.

**Definition of done:** the health check returns True from PowerShell.

### Day 2 — OpenVoice V2 in Colab, voice profile script on Windows

**Goal:** `python scripts/create_voice_profile.py samples/voice.wav` produces `storage/voices/<id>/embedding.npy` extracted on Colab.

Tasks:

1. Add an OpenVoice V2 install cell to the notebook (pinned commit; see [§5.1](#51-notebookscolab_inference_serveripynb)).
2. Add the `POST /voice/extract_embedding` endpoint — accepts uploaded WAV, returns embedding as `.npy` bytes.
3. Add `extract_voice_embedding(wav_path)` to `colab_worker_client.py`.
4. Write `scripts/create_voice_profile.py` — see [§5.6](#56-scriptscreate_voice_profilepy).
5. Run it end to end. Verify `storage/voices/<id>/embedding.npy` exists and is non-empty.

### Day 3 — MuseTalk in Colab, avatar profile script on Windows

**Goal:** `python scripts/create_avatar_profile.py samples/face.mp4` produces `storage/avatars/<id>/cache.tar.gz` of MuseTalk's preprocessed latents.

Tasks:

1. Add the MuseTalk 1.5 install cell. Pinned commit. Download model weights to `/content/models/musetalk/`.
2. Add `POST /avatar/preprocess` — accepts MP4, runs MuseTalk's face detection + latent extraction, tars the cache dir, returns it.
3. Add `POST /avatar/upload_cache` — accepts a tar.gz and untars into `/content/cache/<avatar_id>/`. Used after Colab restarts.
4. Add `preprocess_avatar(video_path)` and `upload_avatar_cache(avatar_id, tar_path)` to the client.
5. Write `scripts/create_avatar_profile.py` — see [§5.7](#57-scriptscreate_avatar_profilepy). Include a `--rehydrate <id>` mode that uploads an existing cache without recomputing.
6. Run end to end. Cache should be 50–500 MB depending on video length.

### Day 4 — batch MP4 generation (Mode A, no streaming yet)

**Goal:** `python scripts/generate_video.py --voice <id> --avatar <id> --text "Hello world"` produces `storage/outputs/<run_id>/output.mp4` on Windows.

Tasks:

1. Add the WS endpoint `WS /tts/stream` to the Colab notebook. Wraps OpenVoice V2 to emit PCM chunks of ~200 ms.
2. Add the WS endpoint `WS /lipsync/stream` — receives PCM chunks, calls MuseTalk frame by frame, emits `{frame_idx, jpeg_bytes, pts}`.
3. Add `tts_stream(text, voice_embedding)` and `lipsync_stream(pcm_chunks, avatar_id)` async generators to the client.
4. Write `scripts/generate_video.py` — see [§5.8](#58-scriptsgenerate_videopy). It calls Colab, writes raw frames + audio, then uses **FFmpeg on Windows** to mux to MP4.
5. Verify the output plays in VLC with lip-sync intact.

### Day 5 — FastAPI control plane + WebRTC streaming (Mode B)

**Goal:** open `tests/test_page.html` in Chrome, click Connect, type text, watch your avatar speak it.

Tasks:

1. Write `services/api/settings.py`, `services/api/db.py`. Minimal SQLite — see [§5.3](#53-servicesapidbpy).
2. Write the routers in `services/api/routes/`. See [§5.4](#54-servicesapiroutes).
3. Write `services/api/media/aiortc_track.py` — `VideoStreamTrack` and `AudioStreamTrack` that read from `asyncio.Queue` instances. See [§5.10](#510-servicesapimediaaiortc_trackpy).
4. Write `services/api/remote_stream_consumer.py` — opens the Colab WS, decodes frames + PCM, pushes to the queues. See [§5.11](#511-servicesapiremote_stream_consumerpy).
5. Write `tests/test_page.html` — vanilla `RTCPeerConnection` + a text input + a Connect button. See [§5.12](#512-teststest_pagehtml).
6. Smoke test the full path. Latency target: first frame within 3 seconds of typing.

### Day 6 — resilience polish + verification

Tasks:

1. Add `/healthz` polling in `colab_worker_client.py` (every 10 s). On failure → mark sessions degraded, surface a "reconnecting" state over the WS control channel.
2. Add `.env` hot-reload (watch the file mtime; reconfigure the client base URL when it changes). This lets you paste a new Colab URL without restarting FastAPI.
3. Run the [verification checklist](#7-verification-checklist).
4. Commit. Tag `v0.1-phase-0`.

---

## 5. File-by-file specifications

Every file below has a clear purpose, a minimal viable shape, and the **gotchas** specific to this hybrid setup. Implement them in the order from §4.

### 5.1 `notebooks/colab_inference_server.ipynb`

A single notebook with these cells, in order:

**Cell 1 — install pinned versions and weights.**

```python
# In Colab. The pins below are known to work as of late 2025/early 2026.
!nvidia-smi
!pip install --quiet "fastapi>=0.115" "uvicorn[standard]>=0.32" "python-multipart>=0.0.12" "websockets>=13" "numpy<2" "Pillow>=10.4" "soundfile>=0.12" "librosa>=0.10"
# OpenVoice V2 — pin the commit you tested.
!git clone https://github.com/myshell-ai/OpenVoice /content/OpenVoice
%cd /content/OpenVoice
!git checkout v2  # use the v2 branch / known commit
!pip install --quiet -e .
%cd /content
# MuseTalk 1.5 — pin the commit you tested.
!git clone https://github.com/TMElyralab/MuseTalk /content/MuseTalk
%cd /content/MuseTalk
# Follow MuseTalk's install instructions (mmcv + face detector wheels).
# See README for the exact versions for the commit you pin to.
!pip install --quiet -r requirements.txt
%cd /content
# Download weights — replace with the latest official URLs from each repo's README.
!mkdir -p /content/models/openvoice && \
  wget -q -O /content/models/openvoice/checkpoints_v2.zip <OPENVOICE_V2_URL> && \
  unzip -q /content/models/openvoice/checkpoints_v2.zip -d /content/models/openvoice
!mkdir -p /content/models/musetalk && \
  python /content/MuseTalk/download_weights.py --dest /content/models/musetalk
```

**Cell 2 — install and start `cloudflared`, capture the quick-tunnel URL.**

```python
import subprocess, re, time, threading, os

!wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O /usr/local/bin/cloudflared
!chmod +x /usr/local/bin/cloudflared

PORT = 8000
tunnel_url = None

def run_tunnel():
    global tunnel_url
    proc = subprocess.Popen(
        ["cloudflared", "tunnel", "--url", f"http://localhost:{PORT}"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    for line in proc.stdout:
        print(line, end="")
        m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", line)
        if m and not tunnel_url:
            tunnel_url = m.group(0)

threading.Thread(target=run_tunnel, daemon=True).start()
```

**Cell 3 — define the FastAPI app and all endpoints.**

This is the longest cell. It should define:

- `app = FastAPI()`
- A `lazy_load_openvoice()` and `lazy_load_musetalk()` that import + load weights on first call (so the cell finishes quickly even if a model fails — debugging is easier).
- `GET /healthz` → `{"status": "ok", "openvoice_loaded": bool, "musetalk_loaded": bool, "gpu": torch.cuda.is_available()}`.
- `POST /voice/extract_embedding` — see §5.5 protocol.
- `POST /avatar/preprocess` — produces `cache.tar.gz`.
- `POST /avatar/upload_cache` — restores `cache.tar.gz` after a Colab restart.
- `WS /tts/stream` — text in → `{"type": "pcm", "pts": float, "data": base64}` frames out.
- `WS /lipsync/stream` — receives `{"type": "pcm", ...}` and `{"type": "end"}`, emits `{"type": "frame", "frame_idx": int, "pts": float, "jpeg_b64": str}` and a final `{"type": "end"}`.

**Cell 4 — start uvicorn in a thread and print the tunnel URL.**

```python
import uvicorn, threading, time
config = uvicorn.Config(app, host="0.0.0.0", port=PORT, log_level="info")
server = uvicorn.Server(config)
threading.Thread(target=server.run, daemon=True).start()
# Wait for tunnel URL.
for _ in range(60):
    if tunnel_url:
        print(f"\n\n>>> COLAB_INFERENCE_URL={tunnel_url}\n\n")
        break
    time.sleep(1)
```

**Gotchas:**

- Run cells **in order**. Restarting the runtime resets everything; you'll need to re-upload the avatar cache (Windows does this automatically via `--rehydrate`).
- `cloudflared` quick-tunnel URLs are stable for the life of the process. Restarting Colab → new URL → update `.env` on Windows.
- Keep the Colab tab open in your browser. Closing it idles the runtime out fast.
- MuseTalk's torch + mmcv versions are sensitive. Pin them; do not blindly upgrade.

### 5.2 `services/inference_client/colab_worker_client.py`

The Windows-side abstraction over the Colab worker. **Designed to be swapped** for a `RunpodWorkerClient` later without changing callers.

```python
# services/inference_client/colab_worker_client.py
from __future__ import annotations
import asyncio, base64, json, os, time
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator
import httpx
import websockets
from dotenv import load_dotenv

class RemoteWorkerUnavailable(RuntimeError): ...

@dataclass
class ColabWorkerClient:
    base_url: str
    http_timeout: float = 120.0
    ws_recv_timeout: float = 30.0

    @classmethod
    def from_env(cls) -> "ColabWorkerClient":
        load_dotenv()
        url = os.environ.get("COLAB_INFERENCE_URL")
        if not url:
            raise RuntimeError("COLAB_INFERENCE_URL not set in .env")
        return cls(base_url=url.rstrip("/"))

    def _ws_url(self, path: str) -> str:
        return self.base_url.replace("https://", "wss://").replace("http://", "ws://") + path

    async def health_check(self) -> dict:
        async with httpx.AsyncClient(timeout=10) as c:
            try:
                r = await c.get(f"{self.base_url}/healthz")
                r.raise_for_status()
                return r.json()
            except Exception as e:
                raise RemoteWorkerUnavailable(str(e)) from e

    async def extract_voice_embedding(self, wav_path: Path) -> bytes:
        async with httpx.AsyncClient(timeout=self.http_timeout) as c:
            with open(wav_path, "rb") as f:
                r = await c.post(f"{self.base_url}/voice/extract_embedding",
                                 files={"audio": (wav_path.name, f, "audio/wav")})
            r.raise_for_status()
            return r.content  # .npy bytes

    async def preprocess_avatar(self, mp4_path: Path) -> bytes:
        async with httpx.AsyncClient(timeout=self.http_timeout * 5) as c:
            with open(mp4_path, "rb") as f:
                r = await c.post(f"{self.base_url}/avatar/preprocess",
                                 files={"video": (mp4_path.name, f, "video/mp4")})
            r.raise_for_status()
            return r.content  # tar.gz bytes

    async def upload_avatar_cache(self, avatar_id: str, tar_bytes: bytes) -> None:
        async with httpx.AsyncClient(timeout=self.http_timeout) as c:
            r = await c.post(f"{self.base_url}/avatar/upload_cache",
                             params={"avatar_id": avatar_id},
                             content=tar_bytes,
                             headers={"content-type": "application/gzip"})
            r.raise_for_status()

    async def tts_stream(self, text: str, embedding_b64: str) -> AsyncIterator[tuple[float, bytes]]:
        async with websockets.connect(self._ws_url("/tts/stream"),
                                      max_size=None,
                                      ping_interval=20) as ws:
            await ws.send(json.dumps({"text": text, "embedding_b64": embedding_b64}))
            while True:
                msg = await asyncio.wait_for(ws.recv(), timeout=self.ws_recv_timeout)
                payload = json.loads(msg)
                if payload.get("type") == "end":
                    return
                if payload.get("type") == "pcm":
                    yield payload["pts"], base64.b64decode(payload["data"])

    async def lipsync_stream(self, avatar_id: str,
                             pcm_chunks: AsyncIterator[tuple[float, bytes]]
                             ) -> AsyncIterator[tuple[int, float, bytes]]:
        async with websockets.connect(self._ws_url("/lipsync/stream"),
                                      max_size=None,
                                      ping_interval=20) as ws:
            await ws.send(json.dumps({"avatar_id": avatar_id}))

            async def pump():
                async for pts, pcm in pcm_chunks:
                    await ws.send(json.dumps({"type": "pcm", "pts": pts,
                                              "data": base64.b64encode(pcm).decode()}))
                await ws.send(json.dumps({"type": "end"}))

            pump_task = asyncio.create_task(pump())
            try:
                while True:
                    msg = await asyncio.wait_for(ws.recv(), timeout=self.ws_recv_timeout)
                    payload = json.loads(msg)
                    if payload.get("type") == "end":
                        return
                    if payload.get("type") == "frame":
                        yield (payload["frame_idx"], payload["pts"],
                               base64.b64decode(payload["jpeg_b64"]))
            finally:
                pump_task.cancel()
```

**Gotchas:**

- `max_size=None` on `websockets.connect` — JPEG frames exceed the default 1 MB cap.
- `ping_interval=20` keeps Cloudflare from idling the WSS connection (default Cloudflare idle is 100 s).
- `wss://` is required for `*.trycloudflare.com`. The helper above derives it from `COLAB_INFERENCE_URL`.

### 5.3 `services/api/db.py`

Minimal SQLAlchemy. Three tables.

```python
# services/api/db.py
from datetime import datetime
from sqlalchemy import Column, String, DateTime, Text, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from services.api.settings import settings

Base = declarative_base()

class VoiceProfile(Base):
    __tablename__ = "voice_profiles"
    id = Column(String, primary_key=True)
    person_name = Column(String, nullable=False)
    source_path = Column(String, nullable=False)
    embedding_path = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

class AvatarProfile(Base):
    __tablename__ = "avatar_profiles"
    id = Column(String, primary_key=True)
    person_name = Column(String, nullable=False)
    source_path = Column(String, nullable=False)
    cache_path = Column(String, nullable=False)
    fps = Column(String, default="25")
    created_at = Column(DateTime, default=datetime.utcnow)

class SessionRecord(Base):
    __tablename__ = "sessions"
    id = Column(String, primary_key=True)
    voice_profile_id = Column(String, nullable=False)
    avatar_profile_id = Column(String, nullable=False)
    state = Column(String, default="idle")  # idle, streaming, degraded, closed
    created_at = Column(DateTime, default=datetime.utcnow)

engine = create_engine(settings.database_url, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, future=True)

def init_db():
    Base.metadata.create_all(engine)
```

### 5.4 `services/api/routes/`

Each router is ~30–60 lines. Implement them to match the contract in [§5.5](#55-protocol-reference).

- `voice_profiles.py` — `POST /api/voice-profiles` (multipart upload). Stores WAV to `storage/voices/<id>/source.wav`, calls `client.extract_voice_embedding`, saves `embedding.npy`, writes row.
- `avatar_profiles.py` — `POST /api/avatar-profiles` (multipart upload). Stores MP4, calls `client.preprocess_avatar`, saves `cache.tar.gz`, writes row. Also `POST /api/avatar-profiles/{id}/rehydrate` to re-upload after a Colab restart.
- `sessions.py` — `POST /api/sessions` creates a session row, returns `{id}`.
- `webrtc.py` — `POST /webrtc/offer` accepts an SDP offer, creates an aiortc `RTCPeerConnection`, attaches the video+audio tracks from `aiortc_track.py`, returns the SDP answer.
- `ws_sessions.py` — `WS /ws/sessions/{id}` accepts `{"type": "speak", "text": "..."}` and pushes status events. Spawns a background task that drives `remote_stream_consumer.run(session_id, text)`.

### 5.5 Protocol reference

These are the wire formats. Keep them stable — they are the contract that lets you swap backends later.

**Colab `/healthz`** (HTTP GET):

```json
{"status": "ok", "openvoice_loaded": true, "musetalk_loaded": true, "gpu": true, "free_vram_mb": 9200}
```

**Colab `/voice/extract_embedding`** (HTTP POST, multipart `audio`):

- Returns raw `numpy.save`-format bytes. Client `np.load(BytesIO(content))` gives the embedding.

**Colab `/avatar/preprocess`** (HTTP POST, multipart `video`):

- Returns `application/gzip` bytes of a tar containing MuseTalk's per-avatar latents, masks, and idle frames.

**Colab `WS /tts/stream`** (one connection per generation):

- Client → server: `{"text": "...", "embedding_b64": "..."}` (single message).
- Server → client (repeated): `{"type": "pcm", "pts": 0.123, "data": "<base64 16-bit PCM, 24kHz mono>"}`.
- Server → client (final): `{"type": "end"}`.

**Colab `WS /lipsync/stream`** (one connection per generation):

- Client → server (first): `{"avatar_id": "..."}`.
- Client → server (repeated): `{"type": "pcm", "pts": 0.123, "data": "<base64 PCM>"}`.
- Client → server (final): `{"type": "end"}`.
- Server → client (repeated): `{"type": "frame", "frame_idx": 0, "pts": 0.040, "jpeg_b64": "..."}`.
- Server → client (final): `{"type": "end"}`.

**Windows control plane** matches Plan.md's section 6 (don't re-document here — the original API contract is correct and unchanged).

### 5.6 `scripts/create_voice_profile.py`

```python
# scripts/create_voice_profile.py
import argparse, asyncio, hashlib, json, shutil, sys, time, uuid
from pathlib import Path
from services.inference_client.colab_worker_client import ColabWorkerClient

async def main(wav: Path, name: str | None):
    client = ColabWorkerClient.from_env()
    await client.health_check()  # fail fast if Colab is down

    voice_id = f"voice_{uuid.uuid4().hex[:8]}"
    target = Path("storage/voices") / voice_id
    target.mkdir(parents=True, exist_ok=True)
    shutil.copy(wav, target / "source.wav")

    print(f"[1/2] uploading {wav.name} to Colab...", flush=True)
    t0 = time.time()
    npy_bytes = await client.extract_voice_embedding(wav)
    (target / "embedding.npy").write_bytes(npy_bytes)
    print(f"[2/2] embedding {(target/'embedding.npy').stat().st_size} bytes in {time.time()-t0:.1f}s")

    (target / "profile.json").write_text(json.dumps({
        "id": voice_id,
        "person_name": name or wav.stem,
        "source_filename": wav.name,
    }, indent=2))
    print(f"Created voice profile: {voice_id} ({target})")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("wav", type=Path)
    ap.add_argument("--name", default=None)
    a = ap.parse_args()
    asyncio.run(main(a.wav, a.name))
```

### 5.7 `scripts/create_avatar_profile.py`

Includes the rehydrate path. **This is the file that makes Colab disposable.**

```python
# scripts/create_avatar_profile.py
import argparse, asyncio, json, shutil, time, uuid
from pathlib import Path
from services.inference_client.colab_worker_client import ColabWorkerClient

async def create(mp4: Path, name: str | None):
    client = ColabWorkerClient.from_env()
    await client.health_check()

    avatar_id = f"avatar_{uuid.uuid4().hex[:8]}"
    target = Path("storage/avatars") / avatar_id
    target.mkdir(parents=True, exist_ok=True)
    shutil.copy(mp4, target / "source.mp4")

    print(f"[1/3] uploading {mp4.name} for MuseTalk preprocessing (this can take 1-3 min)...", flush=True)
    t0 = time.time()
    tar_bytes = await client.preprocess_avatar(mp4)
    (target / "cache.tar.gz").write_bytes(tar_bytes)
    print(f"[2/3] cache {len(tar_bytes)/1e6:.1f} MB in {time.time()-t0:.1f}s")

    print(f"[3/3] re-uploading cache to anchor avatar_id on Colab...")
    await client.upload_avatar_cache(avatar_id, tar_bytes)

    (target / "profile.json").write_text(json.dumps({
        "id": avatar_id,
        "person_name": name or mp4.stem,
        "source_filename": mp4.name,
    }, indent=2))
    print(f"Created avatar profile: {avatar_id} ({target})")

async def rehydrate(avatar_id: str):
    client = ColabWorkerClient.from_env()
    await client.health_check()
    tar_path = Path("storage/avatars") / avatar_id / "cache.tar.gz"
    if not tar_path.exists():
        raise SystemExit(f"No cache found at {tar_path}")
    print(f"Re-uploading {tar_path} to fresh Colab session...")
    await client.upload_avatar_cache(avatar_id, tar_path.read_bytes())
    print("Done.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mp4", type=Path, nargs="?")
    ap.add_argument("--name")
    ap.add_argument("--rehydrate", help="avatar_id to re-upload to a fresh Colab session")
    a = ap.parse_args()
    if a.rehydrate:
        asyncio.run(rehydrate(a.rehydrate))
    elif a.mp4:
        asyncio.run(create(a.mp4, a.name))
    else:
        ap.error("either mp4 path or --rehydrate <id> required")
```

### 5.8 `scripts/generate_video.py`

Batch end-to-end test that doesn't need the FastAPI server. Useful for validating the pipeline before adding WebRTC.

```python
# scripts/generate_video.py
import argparse, asyncio, base64, json, subprocess, time, uuid
from pathlib import Path
import numpy as np
from services.inference_client.colab_worker_client import ColabWorkerClient

async def stream_tts(client, text, embedding_b64):
    async for pts, pcm in client.tts_stream(text, embedding_b64):
        yield pts, pcm

async def main(voice_id: str, avatar_id: str, text: str):
    client = ColabWorkerClient.from_env()
    await client.health_check()

    voice_dir = Path("storage/voices") / voice_id
    avatar_dir = Path("storage/avatars") / avatar_id
    run_id = f"run_{uuid.uuid4().hex[:8]}"
    out_dir = Path("storage/outputs") / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    embedding_b64 = base64.b64encode((voice_dir / "embedding.npy").read_bytes()).decode()

    # Make sure the avatar cache is on the current Colab session.
    await client.upload_avatar_cache(avatar_id, (avatar_dir / "cache.tar.gz").read_bytes())

    # Collect audio + frames concurrently.
    pcm_buf, frames = [], []
    pcm_queue: asyncio.Queue = asyncio.Queue()

    async def produce_pcm():
        async for pts, pcm in client.tts_stream(text, embedding_b64):
            pcm_buf.append((pts, pcm))
            await pcm_queue.put((pts, pcm))
        await pcm_queue.put(None)

    async def pcm_iter():
        while True:
            item = await pcm_queue.get()
            if item is None:
                return
            yield item

    async def consume_frames():
        async for idx, pts, jpeg in client.lipsync_stream(avatar_id, pcm_iter()):
            frames.append((idx, pts, jpeg))

    t0 = time.time()
    await asyncio.gather(produce_pcm(), consume_frames())
    print(f"Generation took {time.time()-t0:.1f}s. {len(frames)} frames, {sum(len(p) for _,p in pcm_buf)} bytes PCM")

    # Write frames as JPEGs + concatenated PCM, then mux with FFmpeg.
    frames.sort(key=lambda f: f[0])
    for idx, _, jpeg in frames:
        (out_dir / f"frame_{idx:06d}.jpg").write_bytes(jpeg)
    pcm_bytes = b"".join(p for _, p in sorted(pcm_buf, key=lambda x: x[0]))
    (out_dir / "audio.pcm").write_bytes(pcm_bytes)

    # MuseTalk produces 25 FPS. OpenVoice PCM is 24 kHz mono 16-bit.
    out_mp4 = out_dir / "output.mp4"
    subprocess.check_call([
        "ffmpeg", "-y",
        "-framerate", "25", "-i", str(out_dir / "frame_%06d.jpg"),
        "-f", "s16le", "-ar", "24000", "-ac", "1", "-i", str(out_dir / "audio.pcm"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-shortest", str(out_mp4),
    ])
    print(f"Wrote {out_mp4}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", required=True)
    ap.add_argument("--avatar", required=True)
    ap.add_argument("--text", required=True)
    a = ap.parse_args()
    asyncio.run(main(a.voice, a.avatar, a.text))
```

### 5.9 `services/api/settings.py`

```python
# services/api/settings.py
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    colab_inference_url: str = ""
    storage_dir: Path = Path("./storage")
    database_url: str = "sqlite:///./data/avatar_ml.sqlite"
    stun_servers: str = "stun:stun.l.google.com:19302"
    colab_http_timeout: int = 120
    colab_ws_recv_timeout: int = 30
    colab_healthcheck_interval: int = 10

settings = Settings()
```

### 5.10 `services/api/media/aiortc_track.py`

The trick that lets Windows be the WebRTC peer while Colab generates frames.

```python
# services/api/media/aiortc_track.py
import asyncio, fractions, io, time
import av
import numpy as np
from PIL import Image
from aiortc import MediaStreamTrack
from aiortc.mediastreams import AudioFrame, VideoFrame

class QueueVideoTrack(MediaStreamTrack):
    kind = "video"
    def __init__(self, frame_queue: asyncio.Queue):
        super().__init__()
        self._q = frame_queue
        self._pts = 0
        self._time_base = fractions.Fraction(1, 90000)

    async def recv(self):
        jpeg_bytes = await self._q.get()
        img = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")
        frame = VideoFrame.from_image(img)
        frame.pts = self._pts
        frame.time_base = self._time_base
        self._pts += 90000 // 25  # 25 fps
        return frame

class QueueAudioTrack(MediaStreamTrack):
    kind = "audio"
    def __init__(self, pcm_queue: asyncio.Queue, sample_rate: int = 24000):
        super().__init__()
        self._q = pcm_queue
        self._pts = 0
        self._sample_rate = sample_rate
        self._time_base = fractions.Fraction(1, sample_rate)

    async def recv(self):
        pcm_bytes = await self._q.get()
        samples = np.frombuffer(pcm_bytes, dtype=np.int16)
        frame = AudioFrame(format="s16", layout="mono", samples=len(samples))
        frame.planes[0].update(samples.tobytes())
        frame.pts = self._pts
        frame.sample_rate = self._sample_rate
        frame.time_base = self._time_base
        self._pts += len(samples)
        return frame
```

### 5.11 `services/api/remote_stream_consumer.py`

The bridge: pulls frames + PCM from Colab WS, pushes into the two queues that feed the WebRTC tracks.

```python
# services/api/remote_stream_consumer.py
import asyncio
from services.inference_client.colab_worker_client import ColabWorkerClient

async def stream_to_queues(client: ColabWorkerClient,
                           voice_id: str, avatar_id: str, text: str,
                           video_queue: asyncio.Queue, audio_queue: asyncio.Queue,
                           embedding_b64: str):
    pcm_relay: asyncio.Queue = asyncio.Queue()

    async def produce_pcm():
        async for pts, pcm in client.tts_stream(text, embedding_b64):
            await audio_queue.put(pcm)
            await pcm_relay.put((pts, pcm))
        await pcm_relay.put(None)

    async def pcm_iter():
        while True:
            item = await pcm_relay.get()
            if item is None:
                return
            yield item

    async def consume_frames():
        async for _idx, _pts, jpeg in client.lipsync_stream(avatar_id, pcm_iter()):
            await video_queue.put(jpeg)

    await asyncio.gather(produce_pcm(), consume_frames())
```

### 5.12 `tests/test_page.html`

```html
<!doctype html>
<html><head><title>Avatar_ML test</title></head>
<body>
<h2>Avatar_ML test page</h2>
<button id="connect">Connect</button>
<input id="text" style="width:60%" value="Hello, this is my AI avatar speaking in my cloned voice." />
<button id="speak" disabled>Speak</button>
<div id="status"></div>
<video id="video" autoplay playsinline style="width:512px;height:512px;background:#000"></video>
<script>
let pc, ws, sessionId;
document.getElementById('connect').onclick = async () => {
  const r = await fetch('http://localhost:8000/api/sessions', {
    method:'POST', headers:{'content-type':'application/json'},
    body: JSON.stringify({voice_profile_id:'voice_xxx', avatar_profile_id:'avatar_xxx', mode:'script_to_avatar', stream_transport:'webrtc'})
  });
  ({id: sessionId} = await r.json());

  pc = new RTCPeerConnection({iceServers:[{urls:'stun:stun.l.google.com:19302'}]});
  pc.ontrack = (e) => { document.getElementById('video').srcObject = e.streams[0]; };
  pc.addTransceiver('audio', {direction:'recvonly'});
  pc.addTransceiver('video', {direction:'recvonly'});
  const offer = await pc.createOffer();
  await pc.setLocalDescription(offer);
  const ans = await fetch('http://localhost:8000/webrtc/offer', {
    method:'POST', headers:{'content-type':'application/json'},
    body: JSON.stringify({session_id: sessionId, sdp: offer.sdp, type: offer.type})
  }).then(r => r.json());
  await pc.setRemoteDescription(ans);

  ws = new WebSocket(`ws://localhost:8000/ws/sessions/${sessionId}`);
  ws.onmessage = (e) => document.getElementById('status').innerText = e.data;
  document.getElementById('speak').disabled = false;
};
document.getElementById('speak').onclick = () => {
  ws.send(JSON.stringify({type:'speak', text: document.getElementById('text').value}));
};
</script></body></html>
```

Replace `voice_xxx` and `avatar_xxx` with real IDs you got from the scripts.

---

## 6. Running the system end-to-end

```powershell
# In one PowerShell window: activate venv, start the Colab notebook in your browser.
# Copy the *.trycloudflare.com URL it prints into .env.
.venv\Scripts\activate

# Smoke test the connection.
python -c "import asyncio; from services.inference_client.colab_worker_client import ColabWorkerClient; print(asyncio.run(ColabWorkerClient.from_env().health_check()))"

# Build profiles (one-time per voice/avatar).
python scripts/create_voice_profile.py samples/voice.wav --name "demo"
python scripts/create_avatar_profile.py samples/face.mp4 --name "demo"

# Batch test (Mode A — no streaming).
python scripts/generate_video.py --voice voice_xxxxxxxx --avatar avatar_xxxxxxxx --text "Hello"

# Streaming (Mode B — open browser to tests/test_page.html).
uvicorn services.api.main:app --reload
```

When Colab disconnects (you'll know — health check fails):

```powershell
# 1. Re-run all cells in the Colab notebook. Copy the new URL.
# 2. Update COLAB_INFERENCE_URL in .env. uvicorn --reload picks it up.
# 3. Re-upload your avatar cache to the fresh Colab session:
python scripts/create_avatar_profile.py --rehydrate avatar_xxxxxxxx
```

---

## 7. Verification checklist

You're done with Phase 0 when **all five** of these pass. Don't skip any — they're each load-bearing.

- [ ] **VP1.** `python -c "asyncio.run(...).health_check()"` returns `{"status":"ok", "gpu":true, "openvoice_loaded":true, "musetalk_loaded":true}`.
- [ ] **VP2.** `python scripts/create_voice_profile.py samples/voice.wav` creates a non-empty `storage/voices/<id>/embedding.npy`. Total time < 30 s.
- [ ] **VP3.** `python scripts/create_avatar_profile.py samples/face.mp4` creates `storage/avatars/<id>/cache.tar.gz` (50–500 MB). Total time < 5 min on a 30 s video.
- [ ] **VP4.** `python scripts/generate_video.py --voice <id> --avatar <id> --text "Hello world"` produces `storage/outputs/<run_id>/output.mp4`. VLC plays it; audio is your cloned voice; lips match audio within a few frames.
- [ ] **VP5.** `uvicorn services.api.main:app` boots. `tests/test_page.html` connects via WebRTC and shows your avatar speaking text typed into the box. **First synced frame within 3 s** of clicking Speak.

Bonus resilience check (recommended):

- [ ] **VP6.** Restart the Colab runtime (`Runtime → Restart runtime`). Copy the new tunnel URL into `.env`. Run `python scripts/create_avatar_profile.py --rehydrate <id>`, then click Speak again in the test page. Streaming resumes within ~15 s without restarting `uvicorn`.

---

## 8. What's deferred and when to come back

These items appear in the original Plan.md sections 5–14 and 15. **Do not build them now.** Build them in the order below, **only when the verification checklist above is all green.**

| Deferred item | Build when |
|---|---|
| React/Next.js frontend (replacing `test_page.html`) | After VP5 is green. Frontend is a polish layer, not a validator. |
| Consent capture UI + audit logs | Before showing the demo to anyone outside your team. |
| Watermark (visible/invisible) on outputs | Same trigger as consent capture. |
| MP4 export job queue (Redis + RQ) | When >1 concurrent batch user; until then `BackgroundTasks` is fine. |
| ASR + LLM (Mode B full conversational loop) | After streaming TTS+lipsync is rock solid. faster-whisper + Ollama llama.cpp. |
| Idle-loop avatar (between speech chunks) | After interrupt handling becomes a real problem in user testing. |
| Interrupt / barge-in handling | First needed when Mode B users complain about waiting through long answers. |
| Docker Compose for local dev | When you have a teammate joining and they need a one-command setup. Skip the `runtime: nvidia` services — those will live on Colab/RunPod instead. |
| LiveKit SFU | When >3 simultaneous viewers per session, or when serving over the public internet. |
| Postgres + S3/MinIO | When you outgrow SQLite + filesystem in production (not a Phase 0 problem). |
| Ray Serve / Triton | When you need dynamic batching or model ensembles to cut $/request. Phase 5. |
| CosyVoice 2 upgrade (replacing OpenVoice V2) | When you have a dedicated GPU and want higher TTS quality. Same `/tts/stream` contract. |
| LatentSync offline render mode | When customers ask for higher-quality non-realtime output (Mode C). |

---

## 9. Scaling path to paid infra

The Phase 0 abstractions map to production with **env-var-level** swaps, not rewrites.

| Phase 0 (free, Colab) | Production (paid, K8s) | Code/config change |
|---|---|---|
| `COLAB_INFERENCE_URL=*.trycloudflare.com` | `INFERENCE_POOL_URL=https://gpu-pool.internal` (LB in front of N workers) | env only |
| 1 Colab notebook | RunPod / Modal / Lambda Labs / k8s GPU node pool — running the **same FastAPI app** from the notebook | new infra |
| OpenVoice V2 in worker | CosyVoice 2 in worker (same `/tts/stream` API) | model swap inside worker |
| aiortc on Windows | LiveKit SFU + LiveKit room per session | client opens LiveKit room instead of `/webrtc/offer` |
| SQLite | Postgres (`DATABASE_URL` change) | URL only |
| Local filesystem `storage/` | S3 / R2 / MinIO (storage adapter swap) | adapter |
| FastAPI `BackgroundTasks` | Redis + Celery/RQ for batch MP4 | additive |
| Single tunnel URL | Cloudflare/ALB → multiple GPU pods | DNS + LB |
| One session at a time | Session pinning to GPU pod via sticky routing | LB rule |

**Critical insight:** the WebSocket protocol in [§5.5](#55-protocol-reference) is the contract. As long as paid workers honor it, **none of the Windows-side code changes** when you swap backends.

---

## 10. Troubleshooting

**`COLAB_INFERENCE_URL not set` → ** `.env` is missing or `python-dotenv` isn't loading it. Run `python -c "import os; from dotenv import load_dotenv; load_dotenv(); print(os.environ.get('COLAB_INFERENCE_URL'))"`. If `None`, you're in the wrong directory or `.env` is misnamed (must be exactly `.env`).

**WSS connection drops after ~100 s of inactivity.** Cloudflare quick-tunnels have an idle cap. The `ping_interval=20` in [§5.2](#52-servicesinference_clientcolab_worker_clientpy) handles this. If still happening, drop to `ping_interval=15`.

**`WebSocketException: message too big`.** The `max_size=None` flag in the client + `max_size=None` on the server-side `WebSocket` in the notebook. Don't omit it — JPEG frames can hit 200 KB.

**Colab "GPU not available" mid-session.** Free Colab throttles aggressive users. You'll see this if you've been generating for hours. Wait 30 min, switch to a different Google account, or move to Kaggle/Lightning AI free tiers (each is a one-env-var swap).

**`aiortc` fails to install on Windows** with crypto/AV build errors. Use `uv pip install aiortc==1.9.0 av==13.1.0` — those have working Windows wheels for Python 3.11. **Do not use Python 3.12** until aiortc ships 3.12 wheels.

**Lip-sync looks "off" by ~200 ms.** Two common causes: (1) frame timestamps not honored — make sure `pts` from Colab is preserved through the queues; (2) audio queue ahead of video queue — add ~80 ms of silence padding at the start of the audio track to give MuseTalk time to produce the first frame.

**MuseTalk preprocessing OOMs on Colab.** Your input video is too long (>60 s) or too high resolution. MuseTalk operates on a 256×256 face region but pre-extracts the full video. Trim to 30 s and reduce to 720p with FFmpeg before uploading.

**Audio comes out garbled / metallic.** Sample rate mismatch. OpenVoice V2 outputs 24 kHz mono 16-bit PCM. The aiortc `AudioFrame` in [§5.10](#510-servicesapimediaaiortc_trackpy) must declare `sample_rate=24000`. Don't resample on Windows — let WebRTC handle it.

**"It worked yesterday, broken today."** First check: is `COLAB_INFERENCE_URL` still valid? Colab session likely died overnight. Re-run the notebook, update `.env`, run `--rehydrate` on your avatars.

---

## 11. Appendix: why these choices

**Why OpenVoice V2 over CosyVoice 2/3 for Phase 0?** CosyVoice 2 is ~7 GB on T4 and its advertised 150 ms first-token latency is H100-only — on T4 you'll see 400–900 ms. With MuseTalk also resident (~5 GB), you have ~3 GB headroom and one OOM bug ends your session. OpenVoice V2 is ~1.5 GB, MIT-licensed, faster cold-start (which matters every Colab restart), and quality is good enough to validate the product loop. **Plan to upgrade TTS in Phase 2** when you're on dedicated GPU.

**Why MuseTalk 1.5 over LatentSync, Wav2Lip, VideoReTalking?** MuseTalk is the only one of these explicitly positioned for **real-time** inference (30+ FPS on V100) and its license states code/model are commercially usable. LatentSync is higher quality but diffusion-based, so it's batch-only. Wav2Lip's open-source variant is non-commercial. VideoReTalking is Apache-2.0 but slower. MuseTalk's 256×256 face-region limit is the main quality ceiling — accept it for Phase 0, add face restoration/upscaling in Phase 4 if needed.

**Why Cloudflare Tunnel over ngrok?** ngrok free has aggressive rate limits and rotating subdomains. Cloudflare quick-tunnel is unlimited, supports WebSocket, and the URL is stable for the life of the `cloudflared` process. No account required.

**Why aiortc on Windows, not in Colab?** Colab blocks inbound UDP and the STUN/TURN paths that WebRTC's ICE negotiation needs. Running aiortc inside Colab would require a paid TURN relay and still fail on many networks. Putting aiortc on Windows means the browser-to-Windows leg is loopback or LAN (no NAT), which works every time. The Windows ↔ Colab leg is TCP WebSocket, which Cloudflare handles cleanly.

**Why one WebSocket per stream session carrying both frames + PCM?** Two streams (separate WS for audio and video) would force you to do clock synchronization across two independent connections, which gets painful when one stutters. One WS with `pts` on every message keeps the relative ordering trivial.

**Why SQLite + filesystem and not Postgres + S3?** They're fine until ~hundreds of concurrent users and ~50 GB of profile data. Adding them now buys nothing and adds three services to install. Swap them when you measure a problem, not before.

**Why no Docker for Phase 0?** Docker on Windows uses WSL2 anyway, and you explicitly preferred native Windows. The only services that benefit from containers are the GPU ones — and those are on Colab. The Windows side is one Python process under `uv`. Adding compose now would mean teaching everyone to install Docker Desktop for zero benefit.

**Why pin every model version and commit?** ML model repos break their main branches casually. MuseTalk and OpenVoice both have history of `pip install -e .` failing after upstream bumps mmcv or torch. Pinning a known-good commit + the Python versions of mmcv, torch, transformers makes "it worked yesterday" actually mean something.
