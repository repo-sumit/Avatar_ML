"""Bridge: pull frames + PCM from the Colab WebSocket, push into the two
asyncio Queues that feed the WebRTC tracks.

This is the meat of streaming Mode B. It runs as a background task spawned by
the WS /ws/sessions/{id} route when the browser sends a `speak` message.
"""

from __future__ import annotations

import asyncio

from services.inference_client.colab_worker_client import ColabWorkerClient


async def stream_to_queues(
    client: ColabWorkerClient,
    voice_id: str,
    avatar_id: str,
    text: str,
    video_queue: "asyncio.Queue[bytes]",
    audio_queue: "asyncio.Queue[bytes]",
    embedding_b64: str,
) -> None:
    """Drive both /tts/stream and /lipsync/stream end-to-end.

    voice_id is accepted for future logging/metrics but is not used directly
    — the embedding (already extracted on the voice profile creation step) is
    what the TTS endpoint needs.
    """
    del voice_id  # Currently unused; kept for symmetry / future logging.

    # We need to feed PCM into two places: the audio_queue (for the WebRTC
    # audio track) and the lipsync_stream (for MuseTalk on Colab). A small
    # relay queue duplicates the stream.
    pcm_relay: asyncio.Queue = asyncio.Queue()

    async def produce_pcm() -> None:
        async for pts, pcm in client.tts_stream(text, embedding_b64):
            await audio_queue.put(pcm)
            await pcm_relay.put((pts, pcm))
        await pcm_relay.put(None)

    async def pcm_iter():
        while True:
            item = await pcm_relay.get()
            if item is None:
                return
            yield item

    async def consume_frames() -> None:
        async for _idx, _pts, jpeg in client.lipsync_stream(avatar_id, pcm_iter()):
            await video_queue.put(jpeg)

    await asyncio.gather(produce_pcm(), consume_frames())
