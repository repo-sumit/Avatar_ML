"""Unified Profile routes.

A "Profile" is one person identified by a name, holding both:
- a voice profile (storage/voices/<voice_id>/)
- an avatar profile (storage/avatars/<avatar_id>/)

Both sub-profiles share the same person_name. The UI sees one entity; the
underlying storage layout is unchanged (so CLI scripts keep working).

GET /api/profiles returns ONLY complete profiles — those where a voice and an
avatar with the same person_name both exist. Orphan voice-only or avatar-only
profiles are hidden.
"""

from __future__ import annotations

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from services.api.routes.avatar_profiles import (
    _scan_profiles as _scan_avatars,
    create_avatar_profile_impl,
)
from services.api.routes.voice_profiles import (
    _scan_profiles as _scan_voices,
    create_voice_profile_impl,
)


router = APIRouter(prefix="/api/profiles", tags=["profiles"])


def _list_unified() -> list[dict]:
    """Group voice + avatar profiles by person_name; return only complete pairs.

    If a name has multiple voices or multiple avatars (user re-uploaded), the
    most recent one wins.
    """
    voices = _scan_voices()    # already sorted newest-first
    avatars = _scan_avatars()

    # Pick the newest voice/avatar per person_name.
    latest_voice: dict[str, dict] = {}
    for v in voices:
        latest_voice.setdefault(v["person_name"], v)
    latest_avatar: dict[str, dict] = {}
    for a in avatars:
        latest_avatar.setdefault(a["person_name"], a)

    out = []
    for name in set(latest_voice) & set(latest_avatar):
        v = latest_voice[name]
        a = latest_avatar[name]
        # "Created" timestamp = the later of the two (when the pair became complete).
        created_at = max(v["created_at"], a["created_at"])
        out.append({
            "name": name,
            "voice_id": v["id"],
            "avatar_id": a["id"],
            "created_at": created_at,
        })
    out.sort(key=lambda r: r["created_at"], reverse=True)
    return out


@router.get("")
def list_profiles() -> list[dict]:
    return _list_unified()


@router.post("")
async def create_profile(
    name: str = Form(...),
    audio: UploadFile = File(...),
    video: UploadFile = File(...),
) -> dict:
    """Create one unified profile from a name + audio + video.

    Internally creates both a voice profile and an avatar profile tagged with
    `person_name=name`. Returns both IDs. The Generate route already takes
    `voice_id + avatar_id`, so the UI can immediately use the new profile.
    """
    name = name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="name is required")

    # Step 1: voice profile (fast — ~5 s after Colab is warm).
    voice = await create_voice_profile_impl(audio, name)

    # Step 2: avatar profile (slow — MuseTalk preprocessing, 1–3 min).
    avatar = await create_avatar_profile_impl(video, name)

    return {
        "name": name,
        "voice_id": voice["id"],
        "avatar_id": avatar["id"],
        "created_at": avatar["created_at"],
    }
