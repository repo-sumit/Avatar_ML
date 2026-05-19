"""Avatar profile routes.

Profiles are stored under storage/avatars/<id>/ with a profile.json index file.
Filesystem is the source of truth — both this route and the CLI script write
the same layout.
"""

from __future__ import annotations

import json
import shutil
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from services.api.db import AvatarProfile, SessionLocal
from services.inference_client.colab_worker_client import (
    ColabWorkerClient,
    RemoteWorkerUnavailable,
)


router = APIRouter(prefix="/api/avatar-profiles", tags=["avatar"])

STORAGE_DIR = Path("storage/avatars")


def _scan_profiles() -> list[dict]:
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


@router.get("")
def list_avatar_profiles() -> list[dict]:
    return _scan_profiles()


@router.post("")
async def create_avatar_profile(
    video: UploadFile = File(...),
    person_name: str = Form(...),
    fps: str = Form("25"),
) -> dict:
    avatar_id = f"avatar_{uuid.uuid4().hex[:8]}"
    target = STORAGE_DIR / avatar_id
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

    # Anchor avatar_id on the current Colab session so lipsync can find the cache.
    try:
        await client.upload_avatar_cache(avatar_id, tar_bytes)
    except RemoteWorkerUnavailable as exc:
        raise HTTPException(status_code=503, detail=f"Colab worker unavailable: {exc}")

    created_at = datetime.utcnow().isoformat()
    profile = {
        "id": avatar_id,
        "person_name": person_name,
        "source_filename": video.filename,
        "fps": fps,
        "created_at": created_at,
    }
    (target / "profile.json").write_text(json.dumps(profile, indent=2))

    with SessionLocal() as session:
        session.merge(AvatarProfile(
            id=avatar_id,
            person_name=person_name,
            source_path=str(source_path),
            cache_path=str(cache_path),
            fps=fps,
        ))
        session.commit()

    return {
        "id": avatar_id,
        "person_name": person_name,
        "source_path": str(source_path),
        "cache_path": str(cache_path),
        "fps": fps,
        "created_at": created_at,
    }


@router.post("/{avatar_id}/rehydrate")
async def rehydrate_avatar(avatar_id: str) -> dict:
    """Re-upload the local cache to a fresh Colab session."""
    cache_path = STORAGE_DIR / avatar_id / "cache.tar.gz"
    if not cache_path.exists():
        raise HTTPException(status_code=404, detail=f"cache not found at {cache_path}")

    client = ColabWorkerClient.from_env()
    try:
        await client.upload_avatar_cache(avatar_id, cache_path.read_bytes())
    except RemoteWorkerUnavailable as exc:
        raise HTTPException(status_code=503, detail=f"Colab worker unavailable: {exc}")

    return {"id": avatar_id, "rehydrated": True}


@router.get("/{avatar_id}")
def get_avatar_profile(avatar_id: str) -> dict:
    target = STORAGE_DIR / avatar_id / "profile.json"
    if not target.exists():
        raise HTTPException(status_code=404, detail="avatar profile not found")
    return json.loads(target.read_text())
