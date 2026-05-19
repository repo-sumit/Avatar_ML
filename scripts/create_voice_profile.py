"""Create a reusable voice profile by extracting an OpenVoice V2 speaker
embedding on Colab and persisting it locally.

Usage:
    python scripts/create_voice_profile.py samples/voice.wav [--name "demo"]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
import time
import uuid
from pathlib import Path

# Allow running as `python scripts/create_voice_profile.py` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services.inference_client.colab_worker_client import (  # noqa: E402
    ColabWorkerClient,
    RemoteWorkerUnavailable,
)


async def main(wav: Path, name: str | None) -> None:
    if not wav.exists():
        raise SystemExit(f"Input file not found: {wav}")

    client = ColabWorkerClient.from_env()
    # Fail fast if Colab is down — saves a confusing upload timeout later.
    try:
        health = await client.health_check()
        print(f"Colab health: {health}", flush=True)
    except RemoteWorkerUnavailable as exc:
        raise SystemExit(f"Cannot reach Colab worker: {exc}")

    voice_id = f"voice_{uuid.uuid4().hex[:8]}"
    target = Path("storage/voices") / voice_id
    target.mkdir(parents=True, exist_ok=True)
    shutil.copy(wav, target / "source.wav")

    print(f"[1/2] uploading {wav.name} to Colab for embedding extraction...", flush=True)
    t0 = time.time()
    npy_bytes = await client.extract_voice_embedding(wav)
    embedding_path = target / "embedding.npy"
    embedding_path.write_bytes(npy_bytes)
    print(
        f"[2/2] embedding ({embedding_path.stat().st_size} bytes) "
        f"saved in {time.time() - t0:.1f}s",
        flush=True,
    )

    (target / "profile.json").write_text(
        json.dumps(
            {
                "id": voice_id,
                "person_name": name or wav.stem,
                "source_filename": wav.name,
            },
            indent=2,
        )
    )
    print(f"Created voice profile: {voice_id} ({target})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wav", type=Path, help="Path to a WAV file (10-60s clean speech).")
    parser.add_argument("--name", default=None, help="Human-readable name (defaults to filename stem).")
    args = parser.parse_args()
    asyncio.run(main(args.wav, args.name))
