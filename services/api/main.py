"""FastAPI app entrypoint.

Run with:
    uvicorn services.api.main:app --reload
"""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from services.api.db import init_db
from services.api.routes import (
    avatar_profiles,
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

# The browser at tests/test_page.html is served from the filesystem (file://)
# or loopback; permit any origin for the local demo. Tighten this when you add
# a real frontend.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory live state. Mapped by session_id -> SessionState. See
# services/api/routes/sessions.py for the dataclass.
app.state.sessions = {}


@app.on_event("startup")
async def on_startup() -> None:
    init_db()
    log.info("DB initialized at %s", settings.database_url)
    log.info("COLAB_INFERENCE_URL=%s", settings.colab_inference_url or "(unset)")
    # Best-effort connectivity check; we don't block startup on it.
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


app.include_router(voice_profiles.router)
app.include_router(avatar_profiles.router)
app.include_router(sessions.router)
app.include_router(webrtc.router)
app.include_router(ws_sessions.router)
