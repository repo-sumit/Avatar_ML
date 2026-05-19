"""WS /ws/sessions/{id} — control channel.

The browser sends {"type": "speak", "text": "..."} after the WebRTC peer is
established. This route spawns a background task that drives the Colab
inference and pushes frames/PCM into the session's queues, which the WebRTC
tracks then emit to the browser.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from pathlib import Path

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from services.api.db import SessionLocal, VoiceProfile
from services.api.remote_stream_consumer import stream_to_queues
from services.inference_client.colab_worker_client import (
    ColabWorkerClient,
    RemoteWorkerUnavailable,
)


router = APIRouter(tags=["ws"])
log = logging.getLogger(__name__)


@router.websocket("/ws/sessions/{session_id}")
async def session_ws(ws: WebSocket, session_id: str) -> None:
    await ws.accept()
    sessions = ws.app.state.sessions
    sess = sessions.get(session_id)
    if sess is None:
        await ws.send_text(json.dumps({"type": "error", "message": "session not found"}))
        await ws.close()
        return

    await ws.send_text(json.dumps({"type": "status", "stage": "ws_open"}))

    try:
        while True:
            raw = await ws.receive_text()
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                await ws.send_text(json.dumps({"type": "error", "message": "invalid json"}))
                continue

            if payload.get("type") == "speak":
                text = payload.get("text", "")
                if not text:
                    await ws.send_text(json.dumps({"type": "error", "message": "empty text"}))
                    continue

                # If a previous speak is still running, leave it alone — barge-in
                # is a deferred Phase 0+ feature.
                if sess.consumer_task and not sess.consumer_task.done():
                    await ws.send_text(json.dumps({"type": "status", "stage": "busy"}))
                    continue

                # Load the embedding from disk and kick off the streaming task.
                with SessionLocal() as db:
                    vp = db.get(VoiceProfile, sess.voice_profile_id)
                    if vp is None:
                        await ws.send_text(json.dumps({"type": "error", "message": "voice profile missing"}))
                        continue
                    embedding_path = Path(vp.embedding_path)

                if not embedding_path.exists():
                    await ws.send_text(
                        json.dumps({"type": "error", "message": f"embedding file missing: {embedding_path}"})
                    )
                    continue

                embedding_b64 = base64.b64encode(embedding_path.read_bytes()).decode()
                client = ColabWorkerClient.from_env()

                async def drive() -> None:
                    """Background task that streams from Colab into the WebRTC queues."""
                    try:
                        await ws.send_text(json.dumps({"type": "status", "stage": "tts_started"}))
                        sess.state = "streaming"
                        await stream_to_queues(
                            client=client,
                            voice_id=sess.voice_profile_id,
                            avatar_id=sess.avatar_profile_id,
                            text=text,
                            video_queue=sess.video_queue,
                            audio_queue=sess.audio_queue,
                            embedding_b64=embedding_b64,
                        )
                        await ws.send_text(json.dumps({"type": "status", "stage": "video_streaming_done"}))
                        sess.state = "ready"
                    except RemoteWorkerUnavailable as exc:
                        log.warning("Remote worker unavailable: %s", exc)
                        sess.state = "degraded"
                        await ws.send_text(
                            json.dumps({"type": "status", "stage": "degraded", "detail": str(exc)})
                        )
                    except Exception as exc:  # noqa: BLE001 — surface the error to the browser
                        log.exception("speak task failed")
                        await ws.send_text(
                            json.dumps({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
                        )

                sess.consumer_task = asyncio.create_task(drive())

            elif payload.get("type") == "ping":
                await ws.send_text(json.dumps({"type": "pong"}))

            else:
                await ws.send_text(
                    json.dumps({"type": "error", "message": f"unknown message type: {payload.get('type')!r}"})
                )

    except WebSocketDisconnect:
        log.info("WS disconnect for session %s", session_id)
    finally:
        if sess.consumer_task and not sess.consumer_task.done():
            sess.consumer_task.cancel()
