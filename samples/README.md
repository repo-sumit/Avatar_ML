# samples/

Drop your input media here. These files are **gitignored** — do not commit them.

- `voice.wav` — 10–60 seconds of clean speech (single speaker, 16 kHz or higher, mono preferred). Used to clone the voice.
- `face.mp4` — 10–60 seconds of a frontal face talking, stable lighting, minimal head motion, no occlusion. Used to drive lip-sync.

**Tip:** record both on your phone, send them to yourself, drop them in here as `voice.wav` and `face.mp4`. Quality (clean audio, frontal face) matters more than length.

If your audio is in a different format, convert it first:

```powershell
ffmpeg -i voice.m4a -ar 16000 -ac 1 voice.wav
```

If your video is too long or high-resolution, trim and downscale:

```powershell
ffmpeg -i face_raw.mp4 -t 30 -vf "scale=-2:720" -c:v libx264 -c:a aac face.mp4
```
