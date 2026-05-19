"""POST /api/generate-video — batch text -> cloned voice + lip-sync -> MP4.

The route returns immediately with a job_id. A background asyncio task drives
the pipeline and updates job state. The UI polls GET /api/generate-video/{id}
for stage + progress + final output URL.
"""

from __future__ import annotations

import asyncio
import logging
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from services.api.db import AvatarProfile, SessionLocal, VoiceProfile
from services.api.pipeline import generate
from services.inference_client.colab_worker_client import (
    ColabWorkerClient,
    RemoteWorkerUnavailable,
)

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/generate-video", tags=["generate"])


class GenerateRequest(BaseModel):
    voice_id: str
    avatar_id: str
    text: str


@dataclass
class JobState:
    job_id: str
    voice_id: str
    avatar_id: str
    text: str
    stage: str = "queued"
    progress: int = 0
    frames_done: int = 0
    frames_total: Optional[int] = None
    error: Optional[str] = None
    output_url: Optional[str] = None
    started_at: float = field(default_factory=time.time)
    task: Optional[asyncio.Task] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "voice_id": self.voice_id,
            "avatar_id": self.avatar_id,
            "stage": self.stage,
            "progress": self.progress,
            "frames_done": self.frames_done,
            "frames_total": self.frames_total,
            "error": self.error,
            "output_url": self.output_url,
            "elapsed_s": round(time.time() - self.started_at, 1),
        }


def _make_progress_cb(state: JobState):
    async def cb(**kw) -> None:
        if "stage" in kw:
            state.stage = kw["stage"]
        if "progress" in kw:
            state.progress = int(kw["progress"])
        if "frames_done" in kw:
            state.frames_done = int(kw["frames_done"])
        if "frames_total" in kw:
            state.frames_total = int(kw["frames_total"])
    return cb


async def _run_job(state: JobState) -> None:
    try:
        client = ColabWorkerClient.from_env()
        # Verify Colab is up before bothering to load profiles.
        await client.health_check()

        result = await generate(
            client=client,
            voice_id=state.voice_id,
            avatar_id=state.avatar_id,
            text=state.text,
            on_progress=_make_progress_cb(state),
        )

        # storage/outputs/<run>/output.mp4 -> /storage/outputs/<run>/output.mp4
        rel = result.out_mp4.relative_to(Path("storage")).as_posix()
        state.output_url = f"/storage/{rel}"
        state.stage = "done"
        state.progress = 100
    except RemoteWorkerUnavailable as exc:
        state.stage = "failed"
        state.error = f"Colab worker unavailable: {exc}"
        log.warning("Job %s failed: %s", state.job_id, exc)
    except FileNotFoundError as exc:
        state.stage = "failed"
        state.error = str(exc)
        log.warning("Job %s failed: %s", state.job_id, exc)
    except Exception as exc:  # noqa: BLE001
        state.stage = "failed"
        # Last 1500 chars of the traceback is enough to diagnose without
        # blowing up the JSON response.
        tb = traceback.format_exc()
        state.error = f"{type(exc).__name__}: {exc}\n\n{tb[-1500:]}"
        log.exception("Job %s failed", state.job_id)


@router.post("")
async def create_job(req: GenerateRequest, request: Request) -> dict[str, str]:
    # Validate that the referenced profiles exist before kicking off a job.
    with SessionLocal() as db:
        if db.get(VoiceProfile, req.voice_id) is None:
            raise HTTPException(status_code=404, detail=f"voice_id {req.voice_id} not found")
        if db.get(AvatarProfile, req.avatar_id) is None:
            raise HTTPException(status_code=404, detail=f"avatar_id {req.avatar_id} not found")
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is required")

    job_id = f"job_{uuid.uuid4().hex[:8]}"
    state = JobState(
        job_id=job_id,
        voice_id=req.voice_id,
        avatar_id=req.avatar_id,
        text=text,
    )
    request.app.state.jobs[job_id] = state
    state.task = asyncio.create_task(_run_job(state))
    return {"job_id": job_id}


@router.get("/{job_id}")
def get_job(job_id: str, request: Request) -> dict[str, Any]:
    state: JobState | None = request.app.state.jobs.get(job_id)
    if state is None:
        raise HTTPException(status_code=404, detail="job not found")
    return state.to_dict()
