"""Windows-side abstraction over the remote inference worker (Colab today,
RunPod/Modal/k8s tomorrow).

The whole point of this module: callers depend on the *interface* (methods on
ColabWorkerClient), not on Colab specifically. To swap backends later, write a
RunpodWorkerClient with the same surface and change one import.

Protocol mirrored from Plan.md §5.5. Do not change message shapes without also
updating the Colab notebook.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator

import httpx
import websockets
from dotenv import load_dotenv


class RemoteWorkerUnavailable(RuntimeError):
    """Raised when the remote worker is unreachable, returns 5xx, or times out.

    Callers should catch this and decide whether to surface a "reconnecting"
    state, retry with backoff, or bail out.
    """


@dataclass
class ColabWorkerClient:
    """Thin async client over the Colab FastAPI inference server.

    Constructed via :meth:`from_env` so the rest of the codebase never reads
    environment variables directly.
    """

    base_url: str
    http_timeout: float = 120.0
    ws_recv_timeout: float = 30.0

    @classmethod
    def from_env(cls) -> "ColabWorkerClient":
        load_dotenv()
        url = os.environ.get("COLAB_INFERENCE_URL", "").strip()
        if not url:
            raise RuntimeError(
                "COLAB_INFERENCE_URL not set in .env. "
                "Run the Colab notebook and paste the printed tunnel URL into .env."
            )
        http_timeout = float(os.environ.get("COLAB_HTTP_TIMEOUT", "120"))
        ws_recv_timeout = float(os.environ.get("COLAB_WS_RECV_TIMEOUT", "30"))
        return cls(
            base_url=url.rstrip("/"),
            http_timeout=http_timeout,
            ws_recv_timeout=ws_recv_timeout,
        )

    def _ws_url(self, path: str) -> str:
        # Cloudflare quick-tunnel URLs are always HTTPS, so WSS is required.
        return (
            self.base_url.replace("https://", "wss://").replace("http://", "ws://")
            + path
        )

    async def health_check(self) -> dict:
        """GET /healthz. Raises RemoteWorkerUnavailable if Colab is unreachable."""
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(f"{self.base_url}/healthz")
                response.raise_for_status()
                return response.json()
        except Exception as exc:
            raise RemoteWorkerUnavailable(
                f"Colab worker unreachable at {self.base_url}: {exc}"
            ) from exc

    @staticmethod
    def _raise_with_body(response: httpx.Response) -> None:
        """Like response.raise_for_status() but surfaces the response body.

        Without this, callers see a generic '500 Internal Server Error' with no
        clue what actually broke on the Colab side. The Colab notebook returns
        FastAPI HTTPException(detail=...) bodies — we want to see them.
        """
        if response.is_success:
            return
        body_preview = response.text[:1000] if response.text else "<empty body>"
        raise RemoteWorkerUnavailable(
            f"HTTP {response.status_code} from {response.request.url}: {body_preview}"
        )

    async def extract_voice_embedding(self, wav_path: Path) -> bytes:
        """POST /voice/extract_embedding. Returns raw numpy.save bytes."""
        async with httpx.AsyncClient(timeout=self.http_timeout) as client:
            with open(wav_path, "rb") as f:
                response = await client.post(
                    f"{self.base_url}/voice/extract_embedding",
                    files={"audio": (wav_path.name, f, "audio/wav")},
                )
            self._raise_with_body(response)
            return response.content

    async def preprocess_avatar(self, mp4_path: Path) -> bytes:
        """POST /avatar/preprocess. Returns tar.gz bytes of MuseTalk's cache."""
        # MuseTalk preprocessing can take a few minutes on a long video.
        async with httpx.AsyncClient(timeout=self.http_timeout * 5) as client:
            with open(mp4_path, "rb") as f:
                response = await client.post(
                    f"{self.base_url}/avatar/preprocess",
                    files={"video": (mp4_path.name, f, "video/mp4")},
                )
            self._raise_with_body(response)
            return response.content

    async def upload_avatar_cache(self, avatar_id: str, tar_bytes: bytes) -> None:
        """POST /avatar/upload_cache. Used after a Colab restart to restore state."""
        async with httpx.AsyncClient(timeout=self.http_timeout) as client:
            response = await client.post(
                f"{self.base_url}/avatar/upload_cache",
                params={"avatar_id": avatar_id},
                content=tar_bytes,
                headers={"content-type": "application/gzip"},
            )
            self._raise_with_body(response)

    async def tts_stream(
        self, text: str, embedding_b64: str
    ) -> AsyncIterator[tuple[float, bytes]]:
        """WS /tts/stream. Yields (pts_seconds, pcm_bytes) tuples.

        pcm_bytes are 16-bit mono 24 kHz PCM. See Plan.md §5.5.
        """
        async with websockets.connect(
            self._ws_url("/tts/stream"),
            max_size=None,        # JPEG frames + PCM exceed default 1 MB cap
            ping_interval=20,     # Cloudflare idles WSS at 100 s; 20 s keeps it alive
        ) as ws:
            await ws.send(json.dumps({"text": text, "embedding_b64": embedding_b64}))
            while True:
                msg = await asyncio.wait_for(ws.recv(), timeout=self.ws_recv_timeout)
                payload = json.loads(msg)
                msg_type = payload.get("type")
                if msg_type == "end":
                    return
                if msg_type == "pcm":
                    yield payload["pts"], base64.b64decode(payload["data"])
                elif msg_type == "error":
                    raise RemoteWorkerUnavailable(
                        f"Colab /tts/stream error: {payload.get('message')}"
                    )

    async def lipsync_stream(
        self,
        avatar_id: str,
        pcm_chunks: AsyncIterator[tuple[float, bytes]],
    ) -> AsyncIterator[tuple[int, float, bytes]]:
        """WS /lipsync/stream. Pushes PCM chunks, yields (frame_idx, pts, jpeg_bytes)."""
        async with websockets.connect(
            self._ws_url("/lipsync/stream"),
            max_size=None,
            ping_interval=20,
        ) as ws:
            await ws.send(json.dumps({"avatar_id": avatar_id}))

            async def pump_pcm() -> None:
                async for pts, pcm in pcm_chunks:
                    await ws.send(
                        json.dumps(
                            {
                                "type": "pcm",
                                "pts": pts,
                                "data": base64.b64encode(pcm).decode(),
                            }
                        )
                    )
                await ws.send(json.dumps({"type": "end"}))

            pump_task = asyncio.create_task(pump_pcm())
            try:
                while True:
                    msg = await asyncio.wait_for(
                        ws.recv(), timeout=self.ws_recv_timeout
                    )
                    payload = json.loads(msg)
                    msg_type = payload.get("type")
                    if msg_type == "end":
                        return
                    if msg_type == "frame":
                        yield (
                            payload["frame_idx"],
                            payload["pts"],
                            base64.b64decode(payload["jpeg_b64"]),
                        )
                    elif msg_type == "error":
                        raise RemoteWorkerUnavailable(
                            f"Colab /lipsync/stream error: {payload.get('message')}"
                        )
            finally:
                pump_task.cancel()
                try:
                    await pump_task
                except (asyncio.CancelledError, Exception):
                    pass
