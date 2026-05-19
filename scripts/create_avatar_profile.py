"""Create a reusable avatar profile by running MuseTalk preprocessing on Colab,
downloading the cache tarball, and anchoring the avatar_id on the current Colab
session.

The --rehydrate path re-uploads an existing local cache to a freshly-started
Colab session. This is what makes Colab disposable: durable state lives on
Windows; Colab is replaced any time it disconnects.

Usage:
    python scripts/create_avatar_profile.py samples/face.mp4 [--name "demo"]
    python scripts/create_avatar_profile.py --rehydrate avatar_xxxxxxxx
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services.inference_client.colab_worker_client import (  # noqa: E402
    ColabWorkerClient,
    RemoteWorkerUnavailable,
)


async def create(mp4: Path, name: str | None) -> None:
    if not mp4.exists():
        raise SystemExit(f"Input file not found: {mp4}")

    client = ColabWorkerClient.from_env()
    try:
        health = await client.health_check()
        print(f"Colab health: {health}", flush=True)
    except RemoteWorkerUnavailable as exc:
        raise SystemExit(f"Cannot reach Colab worker: {exc}")

    avatar_id = f"avatar_{uuid.uuid4().hex[:8]}"
    target = Path("storage/avatars") / avatar_id
    target.mkdir(parents=True, exist_ok=True)
    shutil.copy(mp4, target / "source.mp4")

    print(
        f"[1/3] uploading {mp4.name} for MuseTalk preprocessing "
        f"(face detection + latent extraction, can take 1-3 min)...",
        flush=True,
    )
    t0 = time.time()
    tar_bytes = await client.preprocess_avatar(mp4)
    cache_path = target / "cache.tar.gz"
    cache_path.write_bytes(tar_bytes)
    print(
        f"[2/3] cache ({len(tar_bytes) / 1e6:.1f} MB) saved in {time.time() - t0:.1f}s",
        flush=True,
    )

    print("[3/3] anchoring avatar_id on the current Colab session...", flush=True)
    await client.upload_avatar_cache(avatar_id, tar_bytes)

    (target / "profile.json").write_text(
        json.dumps(
            {
                "id": avatar_id,
                "person_name": name or mp4.stem,
                "source_filename": mp4.name,
            },
            indent=2,
        )
    )
    print(f"Created avatar profile: {avatar_id} ({target})")


async def rehydrate(avatar_id: str) -> None:
    client = ColabWorkerClient.from_env()
    try:
        await client.health_check()
    except RemoteWorkerUnavailable as exc:
        raise SystemExit(f"Cannot reach Colab worker: {exc}")

    tar_path = Path("storage/avatars") / avatar_id / "cache.tar.gz"
    if not tar_path.exists():
        raise SystemExit(f"No cache found at {tar_path}")

    print(
        f"Re-uploading {tar_path} ({tar_path.stat().st_size / 1e6:.1f} MB) "
        f"to the fresh Colab session...",
        flush=True,
    )
    t0 = time.time()
    await client.upload_avatar_cache(avatar_id, tar_path.read_bytes())
    print(f"Done in {time.time() - t0:.1f}s.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mp4", type=Path, nargs="?", help="Path to a face MP4 (10-60s frontal).")
    parser.add_argument("--name", default=None, help="Human-readable name.")
    parser.add_argument(
        "--rehydrate",
        help="Avatar id to re-upload to a fresh Colab session (no recompute).",
    )
    args = parser.parse_args()

    if args.rehydrate:
        asyncio.run(rehydrate(args.rehydrate))
    elif args.mp4:
        asyncio.run(create(args.mp4, args.name))
    else:
        parser.error("either mp4 path or --rehydrate <id> required")
