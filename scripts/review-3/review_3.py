# review-3.py
# Based on streamer-7 architecture (async queue per request, shared mimi context,
# abort_event, main loop capture, aggressive warmup, all GPU optimizations)
# No voice cloning — single speaker, simple /ws endpoint like streamer-4.
# Text splitting by word count (not paragraphs).
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
# REMOVED: CUDA_LAUNCH_BLOCKING=1 serializes every GPU op — kills throughput
# REMOVED: PYTORCH_DISABLE_CUDA_GRAPHS=1 — disabled at compile time instead
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import sys
import threading
import queue
import time
import asyncio
from fastapi.responses import FileResponse
import psutil
import platform
import math
import numpy as np
import torch
import logging
import warnings
import torch._inductor.config
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download

# Warnings/Paths
logging.getLogger("torch.fx.experimental.symbolic_shapes").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)
current_script_path = os.path.abspath(__file__)
streaming_dir = os.path.dirname(current_script_path)
backend_dir = os.path.dirname(streaming_dir)
project_root = os.path.dirname(backend_dir)
sys.path.append(project_root)

try:
    from backend.tts_engine.models import Model, ModelArgs
    from moshi.models import loaders
    from tokenizers.processors import TemplateProcessing
except ImportError:
    print("❌ Critical Error: Could not import backend.tts_engine.models or moshi.models.")
    print("   Make sure you are running this from the project root.")
    exit(1)

# ==============================================================================
# TUNABLE CONSTANTS
# ==============================================================================
MAX_MS             = 60000  # Max audio duration per chunk (model context caps ~163s)
FIRST_CHUNK_FRAMES = 20     # Frames before first audio is sent (low-latency burst)
TEMPERATURE        = 0.8    # Voice expressiveness
TOPK               = 50     # Sampling diversity
CHUNK_SIZE         = 20     # DO NOT CHANGE — mimi streaming contract
SPEAKER_ID         = 0      # Fixed speaker (no voice cloning)
MAX_WORDS_PER_CHUNK = 400   # Word-count limit per text chunk sent to the model

# ==============================================================================
# BLOCK 1: GLOBAL ENVIRONMENT OPTIMIZATIONS
# ==============================================================================

# 1. Enable TensorFloat-32 (Huge speedup on RTX 30xx/40xx/A100)
torch.backends.cuda.matmul.allow_tf32 = True
if hasattr(torch.backends.cudnn, 'allow_tf32'):
    torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

# 2. Memory Fraction
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.95)

# 3. Limit CPU threads to prevent fighting with the WebSocket event loop
torch.set_num_threads(3)

# 4. Boost Process Priority
try:
    p = psutil.Process(os.getpid())
    if platform.system() == "Windows":
        p.nice(psutil.HIGH_PRIORITY_CLASS)
    else:
        p.nice(-10)
    print("🚀 System: Process Priority set to HIGH")
except Exception:
    print("⚠️ System: Could not set high priority (Run as Admin for boost)")

# Global request queue and per-request async response queues
req_queue = queue.Queue()
current_active_user = None

# Signal to stop generation mid-stream on disconnect
abort_event = threading.Event()

# Global event loop reference (set at startup) so the worker thread can post to it
_main_loop: asyncio.AbstractEventLoop = None

app = FastAPI()

@app.on_event("startup")
async def _capture_loop():
    """Capture the running event loop so the worker thread can post chunks back."""
    global _main_loop
    _main_loop = asyncio.get_running_loop()

def flush_queues():
    """Drain the request queue to prevent stale audio leaking into the next request."""
    while not req_queue.empty():
        try:
            req_queue.get_nowait()
        except queue.Empty:
            break

# ==============================================================================
# BLOCK 2: HELPER CLASS
# ==============================================================================
class Segment:
    """Lightweight container for a text+speaker turn (no audio/cloning needed here)."""
    def __init__(self, text, speaker, audio=None, sample_rate=24000):
        self.text        = text
        self.speaker     = speaker
        self.audio       = audio
        self.sample_rate = sample_rate
        self.audio_tokens = None  # Unused without voice cloning

# ==============================================================================
# BLOCK 3: THE ENGINE
# ==============================================================================
class StreamTTS:
    def __init__(self, model_path="sesame/csm-1b", device="cuda"):
        print(f"--- Initializing Engine on {device} ---")
        self.device       = device
        self.sample_rate  = 24000
        self._text_token_cache = {}

        self._load_model(model_path)
        self._run_aggressive_warmup()

    # ------------------------------------------------------------------
    # TEXT TOKENIZER
    # ------------------------------------------------------------------
    def _tokenize_text(self, text: str, speaker_id: int):
        cache_key = f"{speaker_id}:{text}"
        if cache_key in self._text_token_cache:
            return self._text_token_cache[cache_key]

        text_tokens = self.text_tokenizer.encode(f"[{speaker_id}]{text}")

        width = self._num_codebooks + 1
        frame = torch.zeros(len(text_tokens), width).long()
        mask  = torch.zeros(len(text_tokens), width).bool()
        frame[:, -1] = torch.tensor(text_tokens)
        mask[:, -1]  = True

        final_frame = frame.unsqueeze(0).to(self.device)
        final_mask  = mask.unsqueeze(0).to(self.device)

        result = (final_frame, final_mask)
        self._text_token_cache[cache_key] = result
        return result

    # ------------------------------------------------------------------
    # PROMPT BUILDER (text-only, no voice cloning context)
    # ------------------------------------------------------------------
    def _build_prompt(self, text: str, speaker_id: int):
        txt_tok, txt_msk = self._tokenize_text(text, speaker_id)

        tokens = txt_tok.squeeze(0)
        masks  = txt_msk.squeeze(0)

        # Hard cap at model context window
        if tokens.size(0) > 2048:
            tokens = tokens[-2048:]
            masks  = masks[-2048:]

        curr_tokens = tokens.unsqueeze(0)
        curr_mask   = masks.unsqueeze(0)
        curr_pos    = torch.arange(0, tokens.size(0)).unsqueeze(0).to(self.device)

        return curr_tokens, curr_mask, curr_pos

    # ------------------------------------------------------------------
    # DECODE HELPER — always pads to CHUNK_SIZE for consistent mimi shape
    # ------------------------------------------------------------------
    def _decode_buffer(self, buffer: list, real_count: int):
        padded = buffer.copy()
        if real_count < CHUNK_SIZE:
            padded += [torch.zeros_like(buffer[0])] * (CHUNK_SIZE - real_count)
        codes = torch.stack(padded).permute(1, 2, 0)
        wav   = self.audio_tokenizer.decode(codes).squeeze(0).squeeze(0)
        if real_count < CHUNK_SIZE:
            real_samples = int(real_count * self.sample_rate * 0.08)
            wav = wav[:real_samples]
        return wav

    # ------------------------------------------------------------------
    # STREAM GENERATOR — one text chunk inside an active mimi context
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def generate_stream(self, text: str, speaker_id: int, max_ms: int = MAX_MS):
        """Generate audio for one text chunk. Must be called inside audio_tokenizer.streaming()."""
        self.model.reset_caches()

        curr_tokens, curr_mask, curr_pos = self._build_prompt(text, speaker_id)

        buffer      = []
        first_sent  = False
        frame_count = 0
        gen_start   = time.time()

        zeros_1_1      = torch.zeros(1, 1).long().to(self.device)
        zeros_mask_1_1 = torch.zeros(1, 1).bool().to(self.device)

        for _ in range(int(max_ms / 80)):
            if abort_event.is_set():
                print("🛑 Generation aborted by user disconnect.")
                return

            frame_start = time.time()
            next_token  = self.model.generate_frame(
                curr_tokens, curr_mask, curr_pos, TEMPERATURE, TOPK
            )
            frame_ms = (time.time() - frame_start) * 1000

            if torch.all(next_token == 0):
                break  # EOS

            frame_count += 1
            buffer.append(next_token)

            threshold = FIRST_CHUNK_FRAMES if not first_sent else CHUNK_SIZE

            if len(buffer) >= threshold:
                decode_start = time.time()
                wav          = self._decode_buffer(buffer, len(buffer))
                decode_ms    = (time.time() - decode_start) * 1000
                total_ms     = (time.time() - gen_start) * 1000
                print(f"   🎵 Frame {frame_count:3d} | gen={frame_ms:.0f}ms decode={decode_ms:.0f}ms | total={total_ms:.0f}ms | {'FIRST' if not first_sent else 'chunk'}")
                yield wav.cpu().numpy().tobytes()
                buffer     = []
                first_sent = True

            # Autoregressive step
            curr_tokens = torch.cat([next_token, zeros_1_1], dim=1).unsqueeze(1)
            curr_mask   = torch.cat([torch.ones_like(next_token).bool(), zeros_mask_1_1], dim=1).unsqueeze(1)
            curr_pos    = curr_pos[:, -1:] + 1

        # Flush remaining frames
        if buffer:
            wav = self._decode_buffer(buffer, len(buffer))
            yield wav.cpu().numpy().tobytes()

    # ------------------------------------------------------------------
    # REQUEST HANDLER — all chunks share ONE mimi streaming context
    # This eliminates the ~600ms mimi reinitialization between chunks.
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def generate_request(self, chunks: list, speaker_id: int):
        """
        Yield (chunk_index, audio_bytes) for each chunk, then (chunk_index, 'EOS').
        All chunks share a single mimi streaming session for seamless audio.
        """
        with self.audio_tokenizer.streaming(1):
            for i, text in enumerate(chunks):
                label = f"'{text[:40]}...'" if len(text) > 40 else f"'{text}'"
                print(f"   📝 Chunk {i+1}/{len(chunks)}: {label}")
                for audio_bytes in self.generate_stream(text, speaker_id):
                    yield (i, audio_bytes)
                yield (i, "EOS")

    # ------------------------------------------------------------------
    # WARMUP — same aggressive routine as streamer-7
    # ------------------------------------------------------------------
    def _run_aggressive_warmup(self):
        warmup_text = "Hello, this is a comprehensive warmup text."
        speaker_id  = 0

        print("🔥 Starting maximum-intensity warmup sequence...")

        # A. Position Embeddings
        if hasattr(self.model, 'backbone') and hasattr(self.model.backbone, 'positional_embedding'):
            with torch.inference_mode():
                positions = torch.arange(0, 2048).to(self.device)
                _ = self.model.backbone.positional_embedding(positions)

        # B. Memory Allocation Optimization
        if torch.cuda.is_available():
            print("   - Optimizing GPU memory allocation...")
            try:
                reserved = []
                for size_mb in [128, 256, 512, 256, 128, 64]:
                    size        = int(size_mb * 1024 * 1024 / 4)
                    tensor_size = int(math.sqrt(size))
                    reserved.append(torch.ones((tensor_size, tensor_size), device=self.device, dtype=torch.float32))
                torch.cuda.synchronize()
                del reserved
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            except Exception as e:
                print(f"   - Memory pre-allocation skipped: {e}")

        # C. Compile Triggers
        print("   - Forcing compilation of kernels...")
        with torch.inference_mode():
            dummy_tokens = torch.ones(1, 1, 33).long().to(self.device)
            dummy_mask   = torch.ones(1, 1, 33).bool().to(self.device)
            dummy_pos    = torch.zeros(1, 1).long().to(self.device)
            for temp in [0.6, 0.8]:
                _ = self.model.generate_frame(dummy_tokens, dummy_mask, dummy_pos, temp, 40)

        # D. Functional Warmup
        print("   - Running functional warmup generation...")
        try:
            for _ in self.generate_stream(warmup_text, speaker_id, max_ms=2000):
                pass
        except Exception as e:
            print(f"   - Functional warmup warning: {e}")

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        print("✅ Warmup Complete. Model is HOT and READY.")

    # ------------------------------------------------------------------
    # MODEL LOADER
    # ------------------------------------------------------------------
    def _load_model(self, model_path):
        print("   - Loading Model weights...")
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        print(f"   dtype: {dtype}")

        config = ModelArgs(
            backbone_flavor="llama-1B",
            decoder_flavor="llama-100M",
            text_vocab_size=128256,
            audio_vocab_size=2051,
            audio_num_codebooks=32,
        )

        self.model = Model.from_pretrained(model_path, config=config)
        self.model.to(self.device, dtype=dtype)

        print("   - Loading Tokenizers...")
        tokenizer_name = "meta-llama/Llama-3.2-1B"
        self.text_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        bos = self.text_tokenizer.bos_token
        eos = self.text_tokenizer.eos_token
        self.text_tokenizer._tokenizer.post_processor = TemplateProcessing(
            single=f"{bos}:0 $A:0 {eos}:0",
            pair=f"{bos}:0 $A:0 {eos}:0 {bos}:1 $B:1 {eos}:1",
            special_tokens=[
                (f"{bos}", self.text_tokenizer.bos_token_id),
                (f"{eos}", self.text_tokenizer.eos_token_id),
            ],
        )

        mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
        self.audio_tokenizer = loaders.get_mimi(mimi_weight, device=self.device)
        self.audio_tokenizer.set_num_codebooks(32)
        self._num_codebooks = 32

        # Disable CUDA Graphs (compile-time setting)
        torch._inductor.config.triton.cudagraphs = False
        torch._inductor.config.fx_graph_cache    = False

        print("   - Compiling Model (this will take ~1 min)...")
        self.model.backbone = torch.compile(self.model.backbone, mode='reduce-overhead', fullgraph=True, backend='inductor')
        self.model.decoder  = torch.compile(self.model.decoder,  mode='reduce-overhead', fullgraph=True, backend='inductor')

        self.model.setup_caches(1)

# ==============================================================================
# BLOCK 4: WORKER THREAD
# ==============================================================================
def model_worker():
    print("🤖 Worker: Starting Engine...")
    try:
        engine = StreamTTS()
        print("🤖 Worker: Engine Ready and Listening.")
    except Exception as e:
        print(f"❌ Worker Failed to Start: {e}")
        return

    while True:
        try:
            req = req_queue.get()
            if req is None:
                break  # Shutdown signal

            chunks, speaker_id, async_q = req

            for (chunk_idx, item) in engine.generate_request(chunks, speaker_id):
                _main_loop.call_soon_threadsafe(async_q.put_nowait, (chunk_idx, item))

            # Final sentinel — all chunks done
            _main_loop.call_soon_threadsafe(async_q.put_nowait, (-1, "DONE"))

        except Exception as e:
            print(f"⚠️ Worker Error: {e}")
            import traceback; traceback.print_exc()
            _main_loop.call_soon_threadsafe(async_q.put_nowait, (-1, "DONE"))

worker_thread = threading.Thread(target=model_worker, daemon=True)
worker_thread.start()

# ==============================================================================
# BLOCK 5: TEXT SPLITTING (by word count)
# ==============================================================================
def split_text(text: str, max_words: int = MAX_WORDS_PER_CHUNK) -> list:
    """
    Splits text purely by word count. The entire input is treated as a flat
    word stream — no paragraph or newline awareness. Chunks up to max_words words.
    """
    words = text.split()
    return [
        " ".join(words[i:i + max_words])
        for i in range(0, len(words), max_words)
    ]

# ==============================================================================
# BLOCK 6: FASTAPI SERVER
# ==============================================================================

# Serve static files relative to this script's location
FILES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))

@app.get("/")
async def get_index():
    html_path = os.path.join(FILES_DIR, "index.html")
    if not os.path.exists(html_path):
        return f"Error: Could not find index.html at {html_path}"
    return FileResponse(html_path)

@app.get("/client.js")
async def get_js():
    js_path = os.path.join(FILES_DIR, "client.js")
    if not os.path.exists(js_path):
        return f"Error: Could not find client.js at {js_path}"
    return FileResponse(js_path, media_type="application/javascript")

@app.get("/style.css")
async def get_css():
    css_path = os.path.join(FILES_DIR, "style.css")
    if not os.path.exists(css_path):
        return f"Error: Could not find style.css at {css_path}"
    return FileResponse(css_path, media_type="text/css")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global current_active_user

    if current_active_user is not None:
        print("⛔ Connection Rejected: Server Busy")
        await websocket.close(code=1013)
        return

    await websocket.accept()
    current_active_user = websocket
    abort_event.clear()
    flush_queues()
    print("✅ User Connected")

    try:
        while True:
            data = await websocket.receive_text()
            print(f"📩 Input: {data[:60]}{'...' if len(data) > 60 else ''}")

            chunks = split_text(data)
            print(f"   Split into {len(chunks)} chunk(s)")

            if not chunks:
                continue

            async_q: asyncio.Queue = asyncio.Queue()
            req_queue.put((chunks, SPEAKER_ID, async_q))

            start_time   = time.time()
            first_packet = False

            while True:
                chunk_idx, item = await async_q.get()

                if abort_event.is_set():
                    break

                if item == "DONE":
                    break
                elif item == "EOS":
                    # One chunk finished — continue to next
                    continue
                else:
                    # Audio bytes
                    if not first_packet:
                        lat = (time.time() - start_time) * 1000
                        print(f"⚡ First audio latency: {lat:.0f}ms")
                        first_packet = True
                    await websocket.send_bytes(item)

            if not abort_event.is_set():
                await websocket.send_text("END_OF_AUDIO")
                print("✅ Done.")

    except WebSocketDisconnect:
        print("User Disconnected")
    except Exception as e:
        print(f"Socket Error: {e}")
    finally:
        abort_event.set()
        flush_queues()
        current_active_user = None
        print("🔓 Lock Released")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, ws_ping_interval=None, ws_ping_timeout=None)