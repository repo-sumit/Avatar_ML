"""FastAPI app entrypoint.

Run with:
    uvicorn services.api.main:app --reload

The UI lives at http://localhost:8000/ (serves tests/ui.html via StaticFiles).
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from services.api.db import init_db
from services.api.routes import (
    avatar_profiles,
    generate_video,
    sessions,
    voice_profiles,
    webrtc,
    ws_sessions,
)
from services.api.settings import settings
from services.inference_client.colab_worker_client import (
    ColabWorkerClient,
    RemoteWorkerUnavailable,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("avatar_ml")


app = FastAPI(title="Avatar_ML control plane", version="0.1.0")

# CORS — fine for local demo; tighten when you add a real frontend host.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory state. Both keys must exist before any router uses them.
app.state.sessions = {}    # session_id -> SessionState (services/api/routes/sessions.py)
app.state.jobs = {}        # job_id -> JobState (services/api/routes/generate_video.py)


@app.on_event("startup")
async def on_startup() -> None:
    init_db()
    Path("storage/outputs").mkdir(parents=True, exist_ok=True)
    log.info("DB initialized at %s", settings.database_url)
    log.info("COLAB_INFERENCE_URL=%s", settings.colab_inference_url or "(unset)")
    try:
        client = ColabWorkerClient.from_env()
        health = await client.health_check()
        log.info("Colab health: %s", health)
    except RemoteWorkerUnavailable as exc:
        log.warning("Colab worker not reachable at startup: %s", exc)
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not perform Colab health check: %s", exc)


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok", "service": "avatar-ml-control-plane"}


# API routers.
app.include_router(voice_profiles.router)
app.include_router(avatar_profiles.router)
app.include_router(sessions.router)
app.include_router(webrtc.router)
app.include_router(ws_sessions.router)
app.include_router(generate_video.router)


# Static mounts.
# `/storage/outputs/<run_id>/output.mp4` is consumed directly by the <video>
# tag in the UI. Other storage subdirs are not exposed.
app.mount(
    "/storage/outputs",
    StaticFiles(directory="storage/outputs"),
    name="outputs",
)

# Serve the UI at `/`. `html=True` makes /  fall through to index.html and
# allows ui.html to be reachable at /ui.html. Mounting last so it doesn't
# shadow the API routes above.
app.mount("/", StaticFiles(directory="tests", html=True), name="ui")
