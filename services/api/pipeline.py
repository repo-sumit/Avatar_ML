"""Shared text -> cloned-voice -> lip-synced MP4 pipeline.

Used by both:
- scripts/generate_video.py (CLI: text in, MP4 out)
- services/api/routes/generate_video.py (background task with progress callback)

The pipeline:
  1. Upload avatar cache to Colab (idempotent re-anchor).
  2. TTS streaming — collect PCM chunks.
  3. Lip-sync streaming — collect JPEG frames.
  4. Write frames + PCM to disk.
  5. FFmpeg mux to MP4.

`on_progress(stage, **kwargs)` is the only side-channel for UI updates.
"""

from __future__ import annotations

import asyncio
import base64
import math
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable, Callable

from services.inference_client.colab_worker_client import (
    ColabWorkerClient,
    RemoteWorkerUnavailable,
)


ProgressCallback = Callable[..., Awaitable[None] | None]


@dataclass
class GenerateResult:
    run_id: str
    out_dir: Path
    out_mp4: Path
    n_frames: int
    pcm_bytes: int
    elapsed_s: float


async def _maybe_call(cb: ProgressCallback | None, **payload) -> None:
    if cb is None:
        return
    result = cb(**payload)
    if asyncio.iscoroutine(result):
        await result


async def generate(
    *,
    client: ColabWorkerClient,
    voice_id: str,
    avatar_id: str,
    text: str,
    on_progress: ProgressCallback | None = None,
    output_root: Path = Path("storage/outputs"),
    fps: int = 25,
    sample_rate: int = 24000,
) -> GenerateResult:
    """Drive the full pipeline. Returns the MP4 path and stats.

    on_progress(stage=..., progress=0..100, frames_done=..., frames_total=...)
    is awaited on every meaningful state change.
    """
    if not shutil.which("ffmpeg"):
        raise RuntimeError(
            "ffmpeg not found on PATH. Install via `winget install Gyan.FFmpeg`."
        )

    voice_dir = Path("storage/voices") / voice_id
    avatar_dir = Path("storage/avatars") / avatar_id
    embedding_path = voice_dir / "embedding.npy"
    cache_path = avatar_dir / "cache.tar.gz"
    if not embedding_path.exists():
        raise FileNotFoundError(f"Voice embedding not found: {embedding_path}")
    if not cache_path.exists():
        raise FileNotFoundError(f"Avatar cache not found: {cache_path}")

    run_id = f"run_{uuid.uuid4().hex[:8]}"
    out_dir = output_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    await _maybe_call(on_progress, stage="queued", progress=0)

    # Re-anchor the avatar cache on the current Colab session. Cheap if already
    # there; lifesaver if Colab restarted between profile creation and now.
    await _maybe_call(on_progress, stage="uploading_cache", progress=2)
    await client.upload_avatar_cache(avatar_id, cache_path.read_bytes())

    embedding_b64 = base64.b64encode(embedding_path.read_bytes()).decode()

    # ----- Stage: TTS (0 -> 33%) -----
    await _maybe_call(on_progress, stage="tts", progress=5)
    pcm_buf: list[tuple[float, bytes]] = []
    pcm_relay: asyncio.Queue = asyncio.Queue()

    async def produce_pcm() -> None:
        async for pts, pcm in client.tts_stream(text, embedding_b64):
            pcm_buf.append((pts, pcm))
            await pcm_relay.put((pts, pcm))
        await pcm_relay.put(None)

    async def pcm_iter():
        while True:
            item = await pcm_relay.get()
            if item is None:
                return
            yield item

    # ----- Stage: lipsync (33 -> 90%) -----
    frames: list[tuple[int, float, bytes]] = []
    expected_frames_ref = {"n": 0}

    async def consume_frames() -> None:
        async for idx, pts, jpeg in client.lipsync_stream(avatar_id, pcm_iter()):
            frames.append((idx, pts, jpeg))
            # We don't know total frames until TTS finishes, but once we do we
            # can estimate and broadcast progress.
            total = expected_frames_ref["n"]
            if total:
                pct = 33 + min(57, int(57 * len(frames) / total))
                await _maybe_call(
                    on_progress,
                    stage="lipsync",
                    progress=pct,
                    frames_done=len(frames),
                    frames_total=total,
                )

    # Run TTS + lipsync concurrently. Once TTS finishes we know the total audio
    # duration and can estimate frame count.
    async def announce_lipsync_start() -> None:
        # Wait until TTS has produced at least one chunk, then flip to lipsync stage.
        while not pcm_buf:
            await asyncio.sleep(0.05)
        await _maybe_call(on_progress, stage="lipsync", progress=33, frames_done=0)

    tts_task = asyncio.create_task(produce_pcm())
    lip_task = asyncio.create_task(consume_frames())
    announce_task = asyncio.create_task(announce_lipsync_start())

    try:
        await tts_task
    except RemoteWorkerUnavailable:
        lip_task.cancel()
        announce_task.cancel()
        raise

    # TTS done — we now know the total audio length, derive expected frames.
    total_pcm = sum(len(p) for _, p in pcm_buf)
    audio_seconds = total_pcm / 2 / sample_rate
    expected_frames_ref["n"] = max(1, math.ceil(audio_seconds * fps))

    # Wait for lipsync to finish consuming.
    await lip_task
    if not announce_task.done():
        announce_task.cancel()

    # ----- Stage: muxing (90 -> 95%) -----
    await _maybe_call(on_progress, stage="muxing", progress=90,
                      frames_done=len(frames), frames_total=expected_frames_ref["n"])

    frames.sort(key=lambda f: f[0])
    for idx, _pts, jpeg in frames:
        (out_dir / f"frame_{idx:06d}.jpg").write_bytes(jpeg)
    pcm_bytes = b"".join(p for _pts, p in sorted(pcm_buf, key=lambda x: x[0]))
    (out_dir / "audio.pcm").write_bytes(pcm_bytes)

    out_mp4 = out_dir / "output.mp4"
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", str(out_dir / "frame_%06d.jpg"),
        "-f", "s16le",
        "-ar", str(sample_rate),
        "-ac", "1",
        "-i", str(out_dir / "audio.pcm"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-shortest",
        str(out_mp4),
    ]
    # FFmpeg can be noisy on stdout; capture and only surface on failure.
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed (exit {proc.returncode}):\n{proc.stderr[-2000:]}"
        )

    elapsed = time.time() - t0
    await _maybe_call(on_progress, stage="done", progress=100,
                      frames_done=len(frames), frames_total=expected_frames_ref["n"])

    return GenerateResult(
        run_id=run_id,
        out_dir=out_dir,
        out_mp4=out_mp4,
        n_frames=len(frames),
        pcm_bytes=total_pcm,
        elapsed_s=elapsed,
    )
