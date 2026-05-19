"""Queue-backed WebRTC tracks.

Colab generates frames + PCM and pushes them over WebSocket to Windows. The
two queues in this module bridge that consumer to aiortc's RTCPeerConnection.

Windows owns the WebRTC peer because Colab blocks the inbound UDP that
STUN/TURN ICE negotiation needs. By terminating WebRTC on Windows, the
browser-to-server leg becomes loopback or LAN — clean ICE every time.
"""

from __future__ import annotations

import asyncio
import fractions
import io

import numpy as np
from aiortc import MediaStreamTrack
from av import AudioFrame, VideoFrame
from PIL import Image


VIDEO_FPS = 25
VIDEO_CLOCK_RATE = 90000
AUDIO_SAMPLE_RATE = 24000  # OpenVoice V2 PCM is 24 kHz mono 16-bit.


class QueueVideoTrack(MediaStreamTrack):
    """Pulls JPEG-encoded frames out of an asyncio.Queue and yields VideoFrames.

    The recv() call blocks while the queue is empty; aiortc handles this fine —
    the peer just doesn't see new frames until something is queued. That is the
    correct behavior when no Speak request is in flight.
    """

    kind = "video"

    def __init__(self, frame_queue: "asyncio.Queue[bytes]") -> None:
        super().__init__()
        self._queue = frame_queue
        self._pts = 0
        self._time_base = fractions.Fraction(1, VIDEO_CLOCK_RATE)
        self._frame_interval = VIDEO_CLOCK_RATE // VIDEO_FPS

    async def recv(self) -> VideoFrame:
        jpeg_bytes = await self._queue.get()
        image = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")
        frame = VideoFrame.from_image(image)
        frame.pts = self._pts
        frame.time_base = self._time_base
        self._pts += self._frame_interval
        return frame


class QueueAudioTrack(MediaStreamTrack):
    """Pulls 16-bit mono PCM chunks out of an asyncio.Queue and yields AudioFrames.

    Critical: declare sample_rate=24000 here. The browser-side WebRTC stack
    resamples to its native rate, but if we lie about the source rate the
    audio comes out garbled / metallic.
    """

    kind = "audio"

    def __init__(
        self,
        pcm_queue: "asyncio.Queue[bytes]",
        sample_rate: int = AUDIO_SAMPLE_RATE,
    ) -> None:
        super().__init__()
        self._queue = pcm_queue
        self._pts = 0
        self._sample_rate = sample_rate
        self._time_base = fractions.Fraction(1, sample_rate)

    async def recv(self) -> AudioFrame:
        pcm_bytes = await self._queue.get()
        samples = np.frombuffer(pcm_bytes, dtype=np.int16)

        frame = AudioFrame(format="s16", layout="mono", samples=len(samples))
        frame.planes[0].update(samples.tobytes())
        frame.pts = self._pts
        frame.sample_rate = self._sample_rate
        frame.time_base = self._time_base
        self._pts += len(samples)
        return frame
