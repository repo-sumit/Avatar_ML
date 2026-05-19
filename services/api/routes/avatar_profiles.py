"""POST /api/avatar-profiles — multipart upload + MuseTalk preprocessing on Colab.

POST /api/avatar-profiles/{id}/rehydrate — re-upload the local cache after a
Colab restart, no recompute.
"""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from services.api.db import AvatarProfile, SessionLocal
from services.inference_client.colab_worker_client import (
    ColabWorkerClient,
    RemoteWorkerUnavailable,
)


router = APIRouter(prefix="/api/avatar-profiles", tags=["avatar"])


@router.post("")
async def create_avatar_profile(
    video: UploadFile = File(...),
    person_name: str = Form(...),
    fps: str = Form("25"),
) -> dict:
    avatar_id = f"avatar_{uuid.uuid4().hex[:8]}"
    target = Path("storage/avatars") / avatar_id
    target.mkdir(parents=True, exist_ok=True)

    source_path = target / "source.mp4"
    with open(source_path, "wb") as out:
        shutil.copyfileobj(video.file, out)

    client = ColabWorkerClient.from_env()
    try:
        tar_bytes = await client.preprocess_avatar(source_path)
    except RemoteWorkerUnavailable as exc:
        raise HTTPException(status_code=503, detail=f"Colab worker unavailable: {exc}")

    cache_path = target / "cache.tar.gz"
    cache_path.write_bytes(tar_bytes)

    # Anchor the avatar_id on the current Colab session so subsequent lipsync
    # calls know which cache to load.
    try:
        await client.upload_avatar_cache(avatar_id, tar_bytes)
    except RemoteWorkerUnavailable as exc:
        raise HTTPException(status_code=503, detail=f"Colab worker unavailable: {exc}")

    with SessionLocal() as session:
        row = AvatarProfile(
            id=avatar_id,
            person_name=person_name,
            source_path=str(source_path),
            cache_path=str(cache_path),
            fps=fps,
        )
        session.add(row)
        session.commit()

    return {
        "id": avatar_id,
        "person_name": person_name,
        "source_path": str(source_path),
        "cache_path": str(cache_path),
        "fps": fps,
    }


@router.post("/{avatar_id}/rehydrate")
async def rehydrate_avatar(avatar_id: str) -> dict:
    """Re-upload the local cache to a fresh Colab session."""
    with SessionLocal() as session:
        row = session.get(AvatarProfile, avatar_id)
        if row is None:
            raise HTTPException(status_code=404, detail="avatar profile not found")
        cache_path = Path(row.cache_path)

    if not cache_path.exists():
        raise HTTPException(status_code=500, detail=f"cache file missing: {cache_path}")

    client = ColabWorkerClient.from_env()
    try:
        await client.upload_avatar_cache(avatar_id, cache_path.read_bytes())
    except RemoteWorkerUnavailable as exc:
        raise HTTPException(status_code=503, detail=f"Colab worker unavailable: {exc}")

    return {"id": avatar_id, "rehydrated": True}


@router.get("/{avatar_id}")
def get_avatar_profile(avatar_id: str) -> dict:
    with SessionLocal() as session:
        row = session.get(AvatarProfile, avatar_id)
        if row is None:
            raise HTTPException(status_code=404, detail="avatar profile not found")
        return {
            "id": row.id,
            "person_name": row.person_name,
            "source_path": row.source_path,
            "cache_path": row.cache_path,
            "fps": row.fps,
        }
