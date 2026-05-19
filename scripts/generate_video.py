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
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services.api.pipeline import generate  # noqa: E402
from services.inference_client.colab_worker_client import (  # noqa: E402
    ColabWorkerClient,
    RemoteWorkerUnavailable,
)


async def main(voice_id: str, avatar_id: str, text: str) -> None:
    client = ColabWorkerClient.from_env()
    try:
        await client.health_check()
    except RemoteWorkerUnavailable as exc:
        raise SystemExit(f"Cannot reach Colab worker: {exc}")

    async def on_progress(**kw):
        # Single-line progress for the CLI.
        bits = [f"{kw.get('stage', '?'):>14s}", f"{kw.get('progress', 0):>3d}%"]
        if kw.get("frames_total"):
            bits.append(f"{kw.get('frames_done', 0)}/{kw['frames_total']} frames")
        print("  " + "  ".join(bits), flush=True)

    print(f"Generating video for voice={voice_id} avatar={avatar_id} ...")
    try:
        result = await generate(
            client=client,
            voice_id=voice_id,
            avatar_id=avatar_id,
            text=text,
            on_progress=on_progress,
        )
    except FileNotFoundError as exc:
        raise SystemExit(str(exc))
    except RemoteWorkerUnavailable as exc:
        raise SystemExit(f"Colab worker error: {exc}")

    print(
        f"Done in {result.elapsed_s:.1f}s. "
        f"{result.n_frames} frames, {result.pcm_bytes} bytes PCM."
    )
    print(f"Wrote {result.out_mp4}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--voice", required=True, help="Voice profile id.")
    parser.add_argument("--avatar", required=True, help="Avatar profile id.")
    parser.add_argument("--text", required=True, help="Text to synthesize.")
    args = parser.parse_args()
    asyncio.run(main(args.voice, args.avatar, args.text))
