"""POST /webrtc/offer — accept browser SDP, attach queue-backed tracks, answer.

The browser is the WebRTC offerer. Windows is the answerer. The two queue
tracks remain attached for the lifetime of the session; the WS /ws/sessions
route pushes frames + PCM into them when the browser sends `speak`.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from aiortc import RTCPeerConnection, RTCSessionDescription

from services.api.media.aiortc_track import QueueAudioTrack, QueueVideoTrack


router = APIRouter(prefix="/webrtc", tags=["webrtc"])


class OfferRequest(BaseModel):
    session_id: str
    sdp: str
    type: str


@router.post("/offer")
async def webrtc_offer(req: OfferRequest, request: Request) -> dict:
    sessions = request.app.state.sessions
    sess = sessions.get(req.session_id)
    if sess is None:
        raise HTTPException(status_code=404, detail="session not found; create via POST /api/sessions first")

    if sess.pc is not None:
        # Re-offer support is out of scope for Phase 0. Close the old pc.
        await sess.pc.close()
        sess.pc = None

    pc = RTCPeerConnection()
    sess.pc = pc

    @pc.on("connectionstatechange")
    async def on_connection_state_change() -> None:
        # Keep the live SessionState in sync with the WebRTC peer state.
        if pc.connectionState in ("failed", "closed", "disconnected"):
            sess.state = "closed"
        elif pc.connectionState == "connected":
            sess.state = sess.state if sess.state != "idle" else "ready"

    video_track = QueueVideoTrack(sess.video_queue)
    audio_track = QueueAudioTrack(sess.audio_queue)
    pc.addTrack(video_track)
    pc.addTrack(audio_track)

    offer = RTCSessionDescription(sdp=req.sdp, type=req.type)
    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
