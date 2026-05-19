"""POST /api/voice-profiles — multipart upload, server-side embedding extraction."""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from services.api.db import SessionLocal, VoiceProfile
from services.inference_client.colab_worker_client import (
    ColabWorkerClient,
    RemoteWorkerUnavailable,
)


router = APIRouter(prefix="/api/voice-profiles", tags=["voice"])


@router.post("")
async def create_voice_profile(
    audio: UploadFile = File(...),
    person_name: str = Form(...),
) -> dict:
    voice_id = f"voice_{uuid.uuid4().hex[:8]}"
    target = Path("storage/voices") / voice_id
    target.mkdir(parents=True, exist_ok=True)

    source_path = target / "source.wav"
    with open(source_path, "wb") as out:
        shutil.copyfileobj(audio.file, out)

    client = ColabWorkerClient.from_env()
    try:
        npy_bytes = await client.extract_voice_embedding(source_path)
    except RemoteWorkerUnavailable as exc:
        raise HTTPException(status_code=503, detail=f"Colab worker unavailable: {exc}")

    embedding_path = target / "embedding.npy"
    embedding_path.write_bytes(npy_bytes)

    with SessionLocal() as session:
        row = VoiceProfile(
            id=voice_id,
            person_name=person_name,
            source_path=str(source_path),
            embedding_path=str(embedding_path),
        )
        session.add(row)
        session.commit()

    return {
        "id": voice_id,
        "person_name": person_name,
        "source_path": str(source_path),
        "embedding_path": str(embedding_path),
    }


@router.get("/{voice_id}")
def get_voice_profile(voice_id: str) -> dict:
    with SessionLocal() as session:
        row = session.get(VoiceProfile, voice_id)
        if row is None:
            raise HTTPException(status_code=404, detail="voice profile not found")
        return {
            "id": row.id,
            "person_name": row.person_name,
            "source_path": row.source_path,
            "embedding_path": row.embedding_path,
        }
