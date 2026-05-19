# Avatar_ML — hybrid Windows + Colab T4

Real-time cloned-voice talking-head system. Windows runs the control plane (FastAPI + SQLite + WebRTC). Free Colab T4 runs the inference data plane (OpenVoice V2 + MuseTalk 1.5) exposed via Cloudflare Tunnel. See [Plan.md](Plan.md) for the full design.

## One-time setup (Windows, PowerShell)

You need Python 3.11, FFmpeg, and Git on PATH. If you don't have them yet:

```powershell
winget install Python.Python.3.11 Gyan.FFmpeg Git.Git
```

Then create a virtual environment and install dependencies (no `uv` required — Python 3.11's built-in `venv` works):

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
copy .env.example .env
```

If PowerShell blocks `Activate.ps1` with an execution-policy error, run once:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

If you'd rather skip activation entirely, use `.\.venv\Scripts\python.exe` and `.\.venv\Scripts\pip.exe` directly — every script in this repo works that way too.

Then open `notebooks/colab_inference_server.ipynb` in Google Colab (Runtime → Change runtime type → T4 GPU). Run all cells. Copy the printed `*.trycloudflare.com` URL into `.env` as `COLAB_INFERENCE_URL`.

Drop your input media:
- `samples/voice.wav` — 10–60s clean speech (single speaker).
- `samples/face.mp4` — 10–60s frontal face, stable lighting.

## Smoke test the Colab connection

```powershell
python -c "import asyncio; from services.inference_client.colab_worker_client import ColabWorkerClient; print(asyncio.run(ColabWorkerClient.from_env().health_check()))"
```

## Run the UI (recommended)

```powershell
.\.venv\Scripts\python.exe -m uvicorn services.api.main:app --reload
```

Open <http://localhost:8000/> in Chrome. The UI has two cards:

1. **Profile** — one entity per person, holding a voice sample + face video together.
   - **Use existing**: pick a previously-created profile from the dropdown.
   - **Create new**: enter a name, choose a `.wav` (voice) and `.mp4` (face), click *Create profile*. The server extracts the voice embedding (~5–60 s) and runs MuseTalk preprocessing on the video (~1–3 min). On success the new profile appears in the dropdown.
2. **Generate** — type text, click *Generate video*, watch the progress bar advance through `tts → lipsync → muxing → done`. The video player + Download button appear when finished.

A "Profile" is the simplest unit you can iterate on: create once per person, then run unlimited Generate calls against the same profile with different text.

## Run end-to-end via CLI (alternative)

```powershell
.\.venv\Scripts\python.exe scripts\create_voice_profile.py samples\voice.wav --name "demo"
.\.venv\Scripts\python.exe scripts\create_avatar_profile.py samples\face_opt.mp4 --name "demo"
.\.venv\Scripts\python.exe scripts\generate_video.py --voice voice_xxxxxxxx --avatar avatar_xxxxxxxx --text "Hello world"
```

Output lands at `storage\outputs\<run_id>\output.mp4`. The CLI and the UI call the same `services/api/pipeline.generate()` under the hood.

## WebRTC streaming preview (Mode B)

Same `uvicorn` server, open `http://localhost:8000/test_page.html`. Edit `VOICE_PROFILE_ID` / `AVATAR_PROFILE_ID` at the top of `tests/test_page.html` with your IDs. Click **Connect**, type text, click **Speak**.

## When Colab disconnects

1. Re-run all cells in the Colab notebook. Copy the new `*.trycloudflare.com` URL.
2. Update `COLAB_INFERENCE_URL` in `.env` (uvicorn `--reload` picks it up automatically).
3. Re-upload your avatar cache to the fresh Colab session:

```powershell
python scripts/create_avatar_profile.py --rehydrate avatar_xxxxxxxx
```

## Layout

See [Plan.md §3](Plan.md) for the full repo layout. Highlights:

- `services/api/` — FastAPI control plane.
- `services/inference_client/colab_worker_client.py` — the **only** code that talks to Colab. Swap this for `RunpodWorkerClient` later without changing callers.
- `scripts/` — CLI tools for the batch flow.
- `notebooks/colab_inference_server.ipynb` — the entire Colab side.
- `storage/` — voice/avatar profiles and outputs (gitignored, created at runtime).
