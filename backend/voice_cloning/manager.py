import os
import logging
import torch
import torchaudio
from streaming.config import VOICE_DEFINITIONS, SAMPLE_RATE

# Initialize the module-specific logger
logger = logging.getLogger(__name__)

class Segment:
    """
    Data structure representing a distinct voice cloning segment.
    Holds the text transcript, speaker ID, raw audio tensor, and pre-tokenized tensors.
    """
    def __init__(self, text: str, speaker: int, audio: torch.Tensor = None, sample_rate: int = SAMPLE_RATE):
        self.text = text
        self.speaker = speaker
        self.audio = audio
        self.sample_rate = sample_rate
        self.audio_tokens = None

def get_voice_path(filename: str) -> str:
    """Safely resolves the absolute path to a voice file relative to this script."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_dir, "voices", filename)

def load_reference_audio(path: str, sample_rate: int = SAMPLE_RATE) -> torch.Tensor:
    """
    Loads a .wav file, converts stereo to mono if necessary, 
    and resamples the audio to match the engine's target sample rate.
    """
    if not os.path.exists(path):
        logger.error(f"Reference audio not found: {path}")
        raise FileNotFoundError(f"Reference audio not found: {path}")
    
    try:
        wav, sr = torchaudio.load(path)
        # Downmix stereo to mono by averaging channels
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        wav = wav.squeeze(0)
        
        return torchaudio.functional.resample(wav, orig_freq=sr, new_freq=sample_rate)
    except Exception as e:
        logger.error(f"Failed to load or resample audio at {path}", exc_info=True)
        raise

def load_voices(engine) -> dict:
    """
    Iterates through VOICE_DEFINITIONS, loads the reference audio, 
    and pre-computes the tokenizer tensors to prevent latency on first request.
    """
    logger.info("Initializing voice cloning references...")
    voices = {}
    
    for d in VOICE_DEFINITIONS:
        try:
            path = get_voice_path(d["filename"])
            audio = load_reference_audio(path, SAMPLE_RATE)
            seg = Segment(text=d["text"], speaker=d["speaker_id"], audio=audio)
            
            with torch.no_grad():
                audio_gpu = audio.to(engine.device)
                audio_tokens = engine.audio_tokenizer.encode(audio_gpu.unsqueeze(0).unsqueeze(0))[0]
                audio_tokens = audio_tokens[:engine._num_codebooks, :]
                
                eos = torch.zeros(audio_tokens.size(0), 1, device=engine.device)
                audio_tokens = torch.cat([audio_tokens, eos], dim=1)
                
                T = audio_tokens.size(1)
                width = engine._num_codebooks + 1
                frame = torch.zeros(T, width, dtype=torch.long, device=engine.device)
                mask = torch.zeros(T, width, dtype=torch.bool, device=engine.device)
                
                frame[:, :engine._num_codebooks] = audio_tokens.transpose(0, 1)
                mask[:, :engine._num_codebooks] = True
                
                seg.audio_tokens = (frame.unsqueeze(0), mask.unsqueeze(0))
            
            voices[d["name"]] = seg
            logger.debug(f"Successfully loaded and tokenized voice: {d['name']}")
            
        except Exception:
            # exc_info=True grabs the full traceback, which is crucial for debugging file I/O
            logger.error(f"Failed to load voice profile '{d['name']}'", exc_info=True)
            
    logger.info(f"Completed loading {len(voices)} voice profiles.")
    return voices