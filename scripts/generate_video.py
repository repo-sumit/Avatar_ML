"""End-to-end batch test: text -> cloned voice TTS -> MuseTalk lip-sync -> MP4.

This script does not need the FastAPI server. It exists to validate the full
inference pipeline before adding WebRTC.

Usage:
    python scripts/generate_video.py --voice voice_xxxxxxxx \
        --avatar avatar_xxxxxxxx --text "Hello world"
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services.inference_client.colab_worker_client import (  # noqa: E402
    ColabWorkerClient,
    RemoteWorkerUnavailable,
)


async def main(voice_id: str, avatar_id: str, text: str) -> None:
    if not shutil.which("ffmpeg"):
        raise SystemExit("ffmpeg not found on PATH. Install via `winget install Gyan.FFmpeg`.")

    client = ColabWorkerClient.from_env()
    try:
        await client.health_check()
    except RemoteWorkerUnavailable as exc:
        raise SystemExit(f"Cannot reach Colab worker: {exc}")

    voice_dir = Path("storage/voices") / voice_id
    avatar_dir = Path("storage/avatars") / avatar_id
    if not (voice_dir / "embedding.npy").exists():
        raise SystemExit(f"Voice embedding not found: {voice_dir / 'embedding.npy'}")
    if not (avatar_dir / "cache.tar.gz").exists():
        raise SystemExit(f"Avatar cache not found: {avatar_dir / 'cache.tar.gz'}")

    run_id = f"run_{uuid.uuid4().hex[:8]}"
    out_dir = Path("storage/outputs") / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {out_dir}", flush=True)

    embedding_b64 = base64.b64encode((voice_dir / "embedding.npy").read_bytes()).decode()

    # Always re-anchor the avatar cache before generating; it's cheap if the
    # cache is already present on Colab, and a lifesaver if it isn't.
    print("Ensuring avatar cache is on the current Colab session...", flush=True)
    await client.upload_avatar_cache(avatar_id, (avatar_dir / "cache.tar.gz").read_bytes())

    # Streaming pipeline:
    #   produce_pcm:   TTS -> pcm_buf (for muxing) + pcm_relay (for lipsync)
    #   consume_frames: lipsync_stream consumes pcm_relay -> frames buffer
    pcm_buf: list[tuple[float, bytes]] = []
    frames: list[tuple[int, float, bytes]] = []
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

    async def consume_frames() -> None:
        async for idx, pts, jpeg in client.lipsync_stream(avatar_id, pcm_iter()):
            frames.append((idx, pts, jpeg))

    print("Generating...", flush=True)
    t0 = time.time()
    await asyncio.gather(produce_pcm(), consume_frames())
    elapsed = time.time() - t0
    total_pcm = sum(len(p) for _, p in pcm_buf)
    print(
        f"Generation done in {elapsed:.1f}s. "
        f"{len(frames)} frames, {total_pcm} bytes PCM (~{total_pcm / 48000:.1f}s audio).",
        flush=True,
    )

    # Write frames in order.
    frames.sort(key=lambda f: f[0])
    for idx, _pts, jpeg in frames:
        (out_dir / f"frame_{idx:06d}.jpg").write_bytes(jpeg)

    # Concatenate PCM in pts order.
    pcm_bytes = b"".join(p for _pts, p in sorted(pcm_buf, key=lambda x: x[0]))
    (out_dir / "audio.pcm").write_bytes(pcm_bytes)

    # Mux with FFmpeg on Windows.
    out_mp4 = out_dir / "output.mp4"
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        "25",
        "-i",
        str(out_dir / "frame_%06d.jpg"),
        "-f",
        "s16le",
        "-ar",
        "24000",
        "-ac",
        "1",
        "-i",
        str(out_dir / "audio.pcm"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-shortest",
        str(out_mp4),
    ]
    print("Running ffmpeg:", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)
    print(f"Wrote {out_mp4}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--voice", required=True, help="Voice profile id.")
    parser.add_argument("--avatar", required=True, help="Avatar profile id.")
    parser.add_argument("--text", required=True, help="Text to synthesize.")
    args = parser.parse_args()
    asyncio.run(main(args.voice, args.avatar, args.text))
