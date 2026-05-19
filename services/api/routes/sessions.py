"""POST /api/sessions, GET /api/sessions/{id}.

A session ties a (voice_profile, avatar_profile) pair to a stream transport
and an in-memory state holder (RTCPeerConnection + queues). DB persistence is
the durable record; the in-memory state lives in app.state.sessions.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from services.api.db import SessionLocal, SessionRecord


router = APIRouter(prefix="/api/sessions", tags=["sessions"])


class CreateSessionRequest(BaseModel):
    voice_profile_id: str
    avatar_profile_id: str
    mode: str = "script_to_avatar"
    stream_transport: str = "webrtc"


@dataclass
class SessionState:
    """In-memory live state for one streaming session.

    The DB row in SessionRecord is the durable view; this dataclass is the
    live view that carries the WebRTC peer and the two media queues.
    """

    session_id: str
    voice_profile_id: str
    avatar_profile_id: str
    state: str = "idle"
    video_queue: "asyncio.Queue[bytes]" = field(default_factory=asyncio.Queue)
    audio_queue: "asyncio.Queue[bytes]" = field(default_factory=asyncio.Queue)
    pc: Any = None  # aiortc.RTCPeerConnection; typed Any to avoid an import cycle
    consumer_task: asyncio.Task | None = None


@router.post("")
async def create_session(req: CreateSessionRequest, request: Request) -> dict:
    session_id = f"session_{uuid.uuid4().hex[:8]}"

    with SessionLocal() as db:
        row = SessionRecord(
            id=session_id,
            voice_profile_id=req.voice_profile_id,
            avatar_profile_id=req.avatar_profile_id,
            state="idle",
        )
        db.add(row)
        db.commit()

    request.app.state.sessions[session_id] = SessionState(
        session_id=session_id,
        voice_profile_id=req.voice_profile_id,
        avatar_profile_id=req.avatar_profile_id,
    )
    return {"id": session_id}


@router.get("/{session_id}")
def get_session(session_id: str, request: Request) -> dict:
    live = request.app.state.sessions.get(session_id)
    with SessionLocal() as db:
        row = db.get(SessionRecord, session_id)
        if row is None:
            raise HTTPException(status_code=404, detail="session not found")
        return {
            "id": row.id,
            "voice_profile_id": row.voice_profile_id,
            "avatar_profile_id": row.avatar_profile_id,
            "state": live.state if live else row.state,
        }
