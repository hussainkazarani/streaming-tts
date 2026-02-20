import os
import sys
import warnings

# Suppress harmless PyTorch/torchaudio warnings before any other imports
warnings.filterwarnings("ignore", category=UserWarning)

# Add backend/ to path so sibling packages (streaming, tts_engine, voice_cloning) are importable.
# server.py lives in backend/web_api/, so one level up is backend/, two levels up is project root.
_backend_dir  = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
_project_root = os.path.abspath(os.path.join(_backend_dir, '..'))
sys.path.insert(0, _backend_dir)

import threading
import asyncio
import logging
import psutil
import platform
import torch
from fastapi import Form, UploadFile, File
from voice_cloning.manager import Segment, load_reference_audio
from streaming.worker import get_engine
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse

from streaming.utils import split_text
from streaming.worker import req_queue, VOICES, flush_queues, init_worker_thread

# Initialize module logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# performed 1 optimization (priority process on admin privileges)
# Elevate process priority to ensure the web server isn't starved by PyTorch
try:
    p = psutil.Process(os.getpid())
    p.nice(psutil.HIGH_PRIORITY_CLASS if platform.system() == "Windows" else -10)
except Exception as e:
    logger.warning(f"Could not elevate process priority: {e}")

app = FastAPI(title="Stream TTS API")

# Global state to prevent multiple simultaneous users from crashing the GPU VRAM
current_active_user = None
abort_event = threading.Event()

@app.on_event("startup")
async def startup_event():
    """Starts the PyTorch worker thread when the FastAPI server boots up."""
    logger.info("FastAPI server starting up...")
    main_loop = asyncio.get_running_loop()
    init_worker_thread(main_loop)

@app.get("/")
async def get_index():
    """Serves the main HTML interface."""
    return FileResponse(os.path.join(_project_root, "frontend", "index.html"))

@app.get("/favicon.ico")
async def favicon():
    return FileResponse(os.path.join(_project_root, "assets", "favicon.ico"))

@app.get("/client.js")
async def get_client_js():
    """Serves the frontend JavaScript logic."""
    return FileResponse(os.path.join(_project_root, "frontend","client.js"), media_type="application/javascript")

@app.get("/style.css")
async def get_css():
    """Serves the frontend stylesheet."""
    return FileResponse(os.path.join(_project_root, "frontend", "style.css"), media_type="text/css")

@app.get("/api/voices")
async def list_voices():
    """Returns the pre-loaded voices for the frontend UI."""
    return JSONResponse([{"name": name, "speaker_id": seg.speaker} for name, seg in VOICES.items()])

@app.post("/api/voices/upload")
async def upload_voice(
    name: str = Form(...),
    transcript: str = Form(...),
    file: UploadFile = File(...),
):
    if not file.filename.endswith('.wav'):
        return JSONResponse({"error": "Only .wav files accepted"}, status_code=400)

    if name in VOICES:
        return JSONResponse({"error": f"Voice '{name}' already exists"}, status_code=400)

    engine = get_engine()
    if engine is None:
        return JSONResponse({"error": "Engine not ready yet"}, status_code=503)

    import tempfile
    contents = await file.read()
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        audio = load_reference_audio(tmp_path)
        seg   = Segment(text=transcript, speaker=len(VOICES), audio=audio)

        with torch.no_grad():
            audio_gpu    = audio.to(engine.device)
            audio_tokens = engine.audio_tokenizer.encode(audio_gpu.unsqueeze(0).unsqueeze(0))[0]
            audio_tokens = audio_tokens[:engine._num_codebooks, :]
            eos          = torch.zeros(audio_tokens.size(0), 1, device=engine.device)
            audio_tokens = torch.cat([audio_tokens, eos], dim=1)
            T            = audio_tokens.size(1)
            width        = engine._num_codebooks + 1
            frame        = torch.zeros(T, width, dtype=torch.long, device=engine.device)
            mask         = torch.zeros(T, width, dtype=torch.bool, device=engine.device)
            frame[:, :engine._num_codebooks] = audio_tokens.transpose(0, 1)
            mask[:, :engine._num_codebooks]  = True
            seg.audio_tokens = (frame.unsqueeze(0), mask.unsqueeze(0))

        VOICES.update({name: seg})
        logger.info(f"Voice '{name}' uploaded and added to memory.")
        return JSONResponse({"success": True, "name": name, "speaker_id": seg.speaker})

    except Exception as e:
        logger.error(f"Failed to upload voice '{name}'", exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)

    finally:
        os.unlink(tmp_path)

@app.websocket("/api/{voice_name}")
async def voice_websocket(websocket: WebSocket, voice_name: str):
    """
    Handles real-time WebSocket connections.
    Receives text, pushes to the AI worker queue, and streams back audio bytes.
    """
    global current_active_user

    # 1. Validate requested voice
    if voice_name not in VOICES:
        logger.warning(f"Connection rejected: Voice '{voice_name}' not found.")
        return await websocket.close(code=1008)

    # 2. Enforce single-user lock
    if current_active_user is not None:
        logger.warning("Connection rejected: Server is currently busy.")
        return await websocket.close(code=1013)

    await websocket.accept()
    current_active_user = websocket
    voice_segment = VOICES[voice_name]
    speaker_id    = voice_segment.speaker

    abort_event.clear()
    flush_queues()
    logger.info(f"WebSocket connected. Streaming voice: {voice_name}")

    try:
        while True:
            # Wait for incoming text from the browser
            data   = await websocket.receive_text()
            chunks = split_text(data)

            if not chunks:
                continue

            # Create a dedicated async queue for this specific streaming request
            async_q: asyncio.Queue = asyncio.Queue()

            # Pass the job to the PyTorch background thread
            req_queue.put((chunks, speaker_id, voice_segment, abort_event, async_q))

            first_packet = False

            while True:
                # Wait for generated audio chunks from the worker thread
                chunk_idx, item = await async_q.get()

                if abort_event.is_set() or item == "DONE":
                    break

                if item == "EOS":
                    # One chunk finished — continue to next
                    continue

                if not first_packet:
                    logger.debug("First audio packet routed through WebSocket.")
                    first_packet = True

                # Stream the raw audio bytes directly to the browser
                await websocket.send_bytes(item)

            if not abort_event.is_set():
                await websocket.send_text("END_OF_AUDIO")

    except WebSocketDisconnect:
        logger.info("Client disconnected normally.")
    except Exception:
        logger.error("WebSocket encountered an error", exc_info=True)
    finally:
        # Trigger abort to stop the PyTorch generator immediately
        abort_event.set()
        flush_queues()
        current_active_user = None
        logger.info("Connection closed. GPU lock released.")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        ws_ping_interval=None,
        ws_ping_timeout=None,
    )