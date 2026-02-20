import os

# System-level optimizations must be set before importing torch
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import time
import math
import logging
import torch
import torch._inductor.config
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download, login

from streaming.config import (
    HF_TOKEN, MODEL_PATH, DEVICE, SAMPLE_RATE,
    TEMPERATURE, TOPK, CHUNK_SIZE, FIRST_CHUNK_FRAMES, MAX_MS
)
from tts_engine.models import Model, ModelArgs
from moshi.models import loaders
from tokenizers.processors import TemplateProcessing

# Suppress excessive inductor/fx logging from PyTorch internals
logging.getLogger("torch.fx.experimental.symbolic_shapes").setLevel(logging.ERROR)

# Initialize the module logger
logger = logging.getLogger(__name__)


class StreamTTS:
    """
    Handles model initialization, compilation, and real-time inference for the TTS engine.
    Maintains token caches and yields raw audio bytes during streaming generation.
    """
    def __init__(self):
        if HF_TOKEN:
            logger.info("Authenticating with Hugging Face...")
            login(token=HF_TOKEN)
        else:
            logger.warning("No HF_TOKEN found in environment variables.")

        logger.info("Initializing StreamTTS Engine...")
        self.device = DEVICE
        self.sample_rate = SAMPLE_RATE
        self._text_token_cache = {}

        self._configure_hardware()
        self._load_model()
        self._run_aggressive_warmup()
        logger.info("StreamTTS Engine is fully initialized and ready.")

    # 3 optimizations performed (tensorfloat-32, memory fraction, cpu threads)
    def _configure_hardware(self):
        """
        Configures hardware-specific optimizations for PyTorch.
        Enables TensorFloat-32, sets VRAM memory fractions, and caps CPU threads.
        """
        logger.debug("Applying PyTorch hardware optimizations...")
        torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends.cudnn, 'allow_tf32'):
            torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(0.95)

        # Cap to 3 threads to prevent CPU fighting with the WebSocket event loop
        torch.set_num_threads(3)
        logger.debug("CPU threads restricted to 3.")

    # 2 optimizations performed (compile, KV caching)
    def _load_model(self):
        """
        Loads the neural network weights, initializes text/audio tokenizers,
        and triggers torch.compile for the backbone and decoder.
        """
        logger.info(f"Loading TTS model weights from '{MODEL_PATH}'...")
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

        config = ModelArgs(
            backbone_flavor="llama-1B",
            decoder_flavor="llama-100M",
            text_vocab_size=128256,
            audio_vocab_size=2051,
            audio_num_codebooks=32,
        )

        self.model = Model.from_pretrained(MODEL_PATH, config=config)
        self.model.to(self.device, dtype=dtype)

        logger.info("Initializing tokenizers...")
        self.text_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
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

        logger.info("Compiling PyTorch models (this may take ~1 minute)...")
        # REMOVED: CUDA_LAUNCH_BLOCKING=1 serializes every GPU op — kills throughput
        # REMOVED: PYTORCH_DISABLE_CUDA_GRAPHS=1 — disabled at compile time instead
        torch._inductor.config.triton.cudagraphs = False
        torch._inductor.config.fx_graph_cache = False

        self.model.backbone = torch.compile(self.model.backbone, mode='reduce-overhead', fullgraph=True, backend='inductor')
        self.model.decoder  = torch.compile(self.model.decoder,  mode='reduce-overhead', fullgraph=True, backend='inductor')
        self.model.setup_caches(1)
        logger.info("Model compilation complete.")

    def _tokenize_text(self, text: str, speaker_id: int):
        """Encodes text input into tensor formats, utilizing a cache for repeated strings."""
        cache_key = f"{speaker_id}:{text}"
        if cache_key in self._text_token_cache:
            return self._text_token_cache[cache_key]

        text_tokens = self.text_tokenizer.encode(f"[{speaker_id}]{text}")

        width = self._num_codebooks + 1
        frame = torch.zeros(len(text_tokens), width).long()
        mask  = torch.zeros(len(text_tokens), width).bool()

        frame[:, -1] = torch.tensor(text_tokens)
        mask[:, -1]  = True

        result = (frame.unsqueeze(0).to(self.device), mask.unsqueeze(0).to(self.device))
        self._text_token_cache[cache_key] = result
        return result

    def _tokenize_audio(self, audio: torch.Tensor):
        """Encodes raw reference audio into mimi codebook tensors."""
        audio = audio.to(self.device)
        audio_tokens = self.audio_tokenizer.encode(audio.unsqueeze(0).unsqueeze(0))[0]
        audio_tokens = audio_tokens[:self._num_codebooks, :]

        eos = torch.zeros(audio_tokens.size(0), 1, device=self.device)
        audio_tokens = torch.cat([audio_tokens, eos], dim=1)

        T     = audio_tokens.size(1)
        width = self._num_codebooks + 1
        frame = torch.zeros(T, width).long().to(self.device)
        mask  = torch.zeros(T, width).bool().to(self.device)
        frame[:, :self._num_codebooks] = audio_tokens.transpose(0, 1)
        mask[:, :self._num_codebooks]  = True

        return frame.unsqueeze(0), mask.unsqueeze(0)

    def _build_prompt(self, text: str, speaker_id: int, context_segments: list):
        """Constructs the combined tensor payload from text and voice cloning context."""
        all_tokens, all_masks = [], []

        for seg in context_segments:
            txt_tok, txt_msk = self._tokenize_text(seg.text, seg.speaker)
            all_tokens.append(txt_tok.squeeze(0))
            all_masks.append(txt_msk.squeeze(0))

            if seg.audio_tokens is not None:
                # Use pre-tokenized audio — safe to call inside streaming() context
                aud_tok, aud_msk = seg.audio_tokens
                all_tokens.append(aud_tok.squeeze(0))
                all_masks.append(aud_msk.squeeze(0))
            elif seg.audio is not None:
                # Fallback: tokenize on the fly (will crash inside streaming context!)
                aud_tok, aud_msk = self._tokenize_audio(seg.audio)
                all_tokens.append(aud_tok.squeeze(0))
                all_masks.append(aud_msk.squeeze(0))

        txt_tok, txt_msk = self._tokenize_text(text, speaker_id)
        all_tokens.append(txt_tok.squeeze(0))
        all_masks.append(txt_msk.squeeze(0))

        tokens = torch.cat(all_tokens, dim=0)
        masks  = torch.cat(all_masks,  dim=0)

        # Context window truncation
        if tokens.size(0) > 2048:
            logger.warning("Context exceeded 2048 tokens; truncating older context.")
            tokens = tokens[-2048:]
            masks  = masks[-2048:]

        curr_pos = torch.arange(0, tokens.size(0)).unsqueeze(0).to(self.device)
        return tokens.unsqueeze(0), masks.unsqueeze(0), curr_pos

    def _decode_buffer(self, buffer: list, real_count: int):
        """Pads the buffer to a consistent chunk size for decoding, then trims excess audio."""
        padded = buffer.copy()
        if real_count < CHUNK_SIZE:
            padded += [torch.zeros_like(buffer[0])] * (CHUNK_SIZE - real_count)

        codes = torch.stack(padded).permute(1, 2, 0)
        wav   = self.audio_tokenizer.decode(codes).squeeze(0).squeeze(0)

        if real_count < CHUNK_SIZE:
            real_samples = int(real_count * self.sample_rate * 0.08)
            wav = wav[:real_samples]

        return wav

    @torch.inference_mode()
    def generate_stream(self, text: str, speaker_id: int, abort_event, context_segments=None, max_ms=MAX_MS):
        """
        Core generator. Yields raw audio byte chunks for a single text segment.
        Must be called inside an active audio_tokenizer.streaming() context.
        Gracefully handles early termination via the abort_event.
        """
        self.model.reset_caches()
        curr_tokens, curr_mask, curr_pos = self._build_prompt(text, speaker_id, context_segments or [])

        buffer      = []
        first_sent  = False
        frame_count = 0
        zeros_1_1      = torch.zeros(1, 1).long().to(self.device)
        zeros_mask_1_1 = torch.zeros(1, 1).bool().to(self.device)

        gen_start_time = time.time()
        logger.debug(f"Starting audio generation for speaker {speaker_id}...")

        for _ in range(int(max_ms / 80)):
            if abort_event.is_set():
                logger.info("Generation aborted by client disconnect.")
                return

            # Time the neural network generation
            frame_start = time.time()
            next_token  = self.model.generate_frame(curr_tokens, curr_mask, curr_pos, TEMPERATURE, TOPK)
            frame_ms    = (time.time() - frame_start) * 1000

            if torch.all(next_token == 0):
                break  # End of Sequence

            frame_count += 1
            buffer.append(next_token)
            threshold = FIRST_CHUNK_FRAMES if not first_sent else CHUNK_SIZE

            if len(buffer) >= threshold:
                # Time the mimi decoder
                decode_start = time.time()
                wav          = self._decode_buffer(buffer, len(buffer))
                decode_ms    = (time.time() - decode_start) * 1000
                total_ms     = (time.time() - gen_start_time) * 1000
                chunk_type   = 'FIRST' if not first_sent else 'chunk'

                # Log the exact latency breakdown for this specific chunk
                logger.debug(
                    f"Frame {frame_count:3d} | gen={frame_ms:.0f}ms decode={decode_ms:.0f}ms | "
                    f"total={total_ms:.0f}ms | {chunk_type}"
                )

                yield wav.cpu().numpy().tobytes()
                buffer     = []
                first_sent = True

            # Autoregressive step
            curr_tokens = torch.cat([next_token, zeros_1_1], dim=1).unsqueeze(1)
            curr_mask   = torch.cat([torch.ones_like(next_token).bool(), zeros_mask_1_1], dim=1).unsqueeze(1)
            curr_pos    = curr_pos[:, -1:] + 1

        # Flush any remaining frames that didn't fill a full chunk
        if buffer:
            wav = self._decode_buffer(buffer, len(buffer))
            yield wav.cpu().numpy().tobytes()

        logger.debug("Audio generation completed successfully.")

    @torch.inference_mode()
    def generate_request(self, chunks: list, speaker_id: int, voice_segment, abort_event):
        """
        Processes multiple text chunks sequentially under a single mimi streaming context.
        This eliminates the ~600ms mimi reinitialization between chunks.
        Yields (chunk_index, audio_bytes) tuples, then (chunk_index, 'EOS') per chunk.
        """
        logger.info(f"Processing request with {len(chunks)} chunk(s) for speaker {speaker_id}.")
        with self.audio_tokenizer.streaming(1):
            for i, text in enumerate(chunks):
                label = f"'{text[:40]}...'" if len(text) > 40 else f"'{text}'"
                logger.debug(f"Chunk {i+1}/{len(chunks)}: {label}")
                for audio_bytes in self.generate_stream(text, speaker_id, abort_event, context_segments=[voice_segment]):
                    yield (i, audio_bytes)
                yield (i, "EOS")

    def _run_aggressive_warmup(self):
        """
        Forces compilation of all inductor graphs before any user connects,
        preventing first-request latency spikes.
        """
        logger.info("Executing aggressive model warmup sequence...")

        # A. Position Embeddings
        if hasattr(self.model, 'backbone') and hasattr(self.model.backbone, 'positional_embedding'):
            with torch.inference_mode():
                positions = torch.arange(0, 2048).to(self.device)
                _ = self.model.backbone.positional_embedding(positions)

        # B. Memory Allocation Optimization
        if torch.cuda.is_available():
            try:
                reserved_memory = []
                for size_mb in [128, 256, 512, 256, 128, 64]:
                    size        = int(size_mb * 1024 * 1024 / 4)
                    tensor_size = int(math.sqrt(size))
                    reserved_memory.append(
                        torch.ones((tensor_size, tensor_size), device=self.device, dtype=torch.float32)
                    )
                torch.cuda.synchronize()
                del reserved_memory
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            except Exception as e:
                logger.warning(f"Memory pre-allocation skipped: {e}")

        # C. Compile Triggers — dummy generate_frame calls to wake up the inductor kernels
        with torch.inference_mode():
            dummy_tokens = torch.ones(1, 1, 33).long().to(self.device)
            dummy_mask   = torch.ones(1, 1, 33).bool().to(self.device)
            dummy_pos    = torch.zeros(1, 1).long().to(self.device)
            for temp in [0.6, 0.8]:
                _ = self.model.generate_frame(dummy_tokens, dummy_mask, dummy_pos, temp, 40)

        # D. Functional Warmup — run a real generate_stream pass to compile the full streaming path
        logger.debug("Running functional warmup generation...")
        try:
            dummy_abort = __import__('threading').Event()  # never set — warmup runs to completion
            for _ in self.generate_stream("Hello, this is a warmup.", 0, dummy_abort, max_ms=2000):
                pass
        except Exception as e:
            logger.warning(f"Functional warmup warning: {e}")

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        logger.info("Warmup complete. Model is hot and ready.")