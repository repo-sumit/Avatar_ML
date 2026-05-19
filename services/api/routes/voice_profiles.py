"""Voice profile routes.

Profiles are stored under storage/voices/<id>/ with a profile.json index file.
Filesystem is the source of truth — both this route and the CLI script write
the same layout, so listings work regardless of which created the profile.

The `create_voice_profile_impl` helper is the actual creation logic, factored
out so the unified `/api/profiles` route can also call it.
"""

from __future__ import annotations

import json
import shutil
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from services.api.db import SessionLocal, VoiceProfile
from services.inference_client.colab_worker_client import (
    ColabWorkerClient,
    RemoteWorkerUnavailable,
)


router = APIRouter(prefix="/api/voice-profiles", tags=["voice"])

STORAGE_DIR = Path("storage/voices")


def _scan_profiles() -> list[dict]:
    """Read profile.json from every storage/voices/*/ directory."""
    if not STORAGE_DIR.exists():
        return []
    out = []
    for child in STORAGE_DIR.iterdir():
        pj = child / "profile.json"
        if not pj.exists():
            continue
        try:
            data = json.loads(pj.read_text())
        except Exception:
            continue
        created = data.get("created_at") or datetime.fromtimestamp(
            pj.stat().st_mtime
        ).isoformat()
        out.append({
            "id": data.get("id", child.name),
            "person_name": data.get("person_name", child.name),
            "created_at": created,
        })
    out.sort(key=lambda r: r["created_at"], reverse=True)
    return out


async def create_voice_profile_impl(
    audio: UploadFile,
    person_name: str,
) -> dict:
    """Reusable create-voice-profile body.

    Called by:
      - POST /api/voice-profiles (this file)
      - POST /api/profiles (services/api/routes/profiles.py)

    On success writes:
      storage/voices/<voice_id>/source.wav
      storage/voices/<voice_id>/embedding.npy
      storage/voices/<voice_id>/profile.json
    """
    voice_id = f"voice_{uuid.uuid4().hex[:8]}"
    target = STORAGE_DIR / voice_id
    target.mkdir(parents=True, exist_ok=True)

    source_path = target / "source.wav"
    with open(source_path, "wb") as out:
        shutil.copyfileobj(audio.file, out)

    client = ColabWorkerClient.from_env()
    try:
        npy_bytes = await client.extract_voice_embedding(source_path)
    except RemoteWorkerUnavailable as exc:
        # Roll back the directory so we don't leave a half-created profile.
        shutil.rmtree(target, ignore_errors=True)
        raise HTTPException(status_code=503, detail=f"Colab worker unavailable: {exc}")

    embedding_path = target / "embedding.npy"
    embedding_path.write_bytes(npy_bytes)

    created_at = datetime.utcnow().isoformat()
    profile = {
        "id": voice_id,
        "person_name": person_name,
        "source_filename": audio.filename,
        "created_at": created_at,
    }
    (target / "profile.json").write_text(json.dumps(profile, indent=2))

    with SessionLocal() as session:
        session.merge(VoiceProfile(
            id=voice_id,
            person_name=person_name,
            source_path=str(source_path),
            embedding_path=str(embedding_path),
        ))
        session.commit()

    return {
        "id": voice_id,
        "person_name": person_name,
        "source_path": str(source_path),
        "embedding_path": str(embedding_path),
        "created_at": created_at,
    }


@router.get("")
def list_voice_profiles() -> list[dict]:
    return _scan_profiles()


@router.post("")
async def create_voice_profile(
    audio: UploadFile = File(...),
    person_name: str = Form(...),
) -> dict:
    return await create_voice_profile_impl(audio, person_name)


@router.get("/{voice_id}")
def get_voice_profile(voice_id: str) -> dict:
    target = STORAGE_DIR / voice_id / "profile.json"
    if not target.exists():
        raise HTTPException(status_code=404, detail="voice profile not found")
    return json.loads(target.read_text())
