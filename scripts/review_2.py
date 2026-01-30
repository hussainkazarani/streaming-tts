# it runs tts to save a file to system
import sys
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
# 1. Get the path to the current file (scripts/run_tts_file.py)
current_script_path = os.path.abspath(__file__)
# 2. Get the parent directory (scripts/)
script_dir = os.path.dirname(current_script_path)
# 3. Get the project root (streaming_tts/)
project_root = os.path.dirname(script_dir)
# 4. Add the project root to Python's search path
sys.path.append(project_root)

import torch
import torchaudio
from huggingface_hub import hf_hub_download
from dataclasses import dataclass
from typing import List, Tuple, Union, List
# models
from backend.tts_engine.models import Model
from moshi.models import loaders
from tokenizers.processors import TemplateProcessing
from transformers import AutoTokenizer

# --- Data Structures ---
@dataclass
class Segment:
    text: str
    speaker: int
    audio: torch.Tensor = None 

# --- Main Class ---
class FileTTS:
    def __init__(self, device: str = None):
        """
        Step 1: Setup & Loading
        This runs once when you create the class. It loads the heavy models
        so you don't have to reload them for every sentence.
        """
        print("--- Step 1: Initializing TTSFile ---")
        
        # 1. Setup Device
        os.environ["NO_TORCH_COMPILE"] = "1"
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # 2. Load Text Tokenizer (Llama)
        print("Loading Text Tokenizer...")
        tokenizer_name = "meta-llama/Llama-3.2-1B"
        self.text_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Configure special tokens for dialogue (BOS/EOS)
        bos = self.text_tokenizer.bos_token
        eos = self.text_tokenizer.eos_token
        self.text_tokenizer._tokenizer.post_processor = TemplateProcessing(
            single=f"{bos}:0 $A:0 {eos}:0",
            pair=f"{bos}:0 $A:0 {eos}:0 {bos}:1 $B:1 {eos}:1",
            special_tokens=[
                (f"{bos}", self.text_tokenizer.bos_token_id), 
                (f"{eos}", self.text_tokenizer.eos_token_id)
            ],
        )

        # 3. Load Audio Tokenizer (Mimi)
        print("Loading Audio Tokenizer (Mimi)...")
        mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
        self.audio_tokenizer = loaders.get_mimi(mimi_weight, device=self.device)
        self.audio_tokenizer.set_num_codebooks(32)
        self.sample_rate = self.audio_tokenizer.sample_rate

        # 4. Load Main Generation Model (CSM 1B)
        print("Loading CSM 1B Model...")
        self.model = Model.from_pretrained("sesame/csm-1b")
        self.model.to(device=self.device, dtype=torch.bfloat16)
        self.model.setup_caches(1)
        print("--- Models Loaded Successfully ---")

    def get_voice_path(self, filename):
        return os.path.join(project_root, "backend", "voice_cloning", "voices", filename)

    def load_reference_audio(self, path: str) -> torch.Tensor:
        """Helper: Loads a WAV file and resamples it to match the model."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Reference audio not found: {path}")
            
        wav, sr = torchaudio.load(path)
        # Convert to mono if stereo
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        wav = wav.squeeze(0)
        return torchaudio.functional.resample(wav, orig_freq=sr, new_freq=self.sample_rate)

    def _tokenize_text(self, text: str, speaker_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Internal: Converts text string into model-ready tensors."""
        text_tokens = self.text_tokenizer.encode(f"[{speaker_id}]{text}")
        
        # Create frame of width 33 (32 audio placeholders + 1 text)
        frame = torch.zeros(len(text_tokens), 33).long()
        mask = torch.zeros(len(text_tokens), 33).bool()
        
        # Fill last column with text
        frame[:, -1] = torch.tensor(text_tokens)
        mask[:, -1] = True
        return frame.to(self.device), mask.to(self.device)

    def _tokenize_audio(self, audio_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Internal: Converts audio waveform into discrete codes using Mimi."""
        audio_tensor = audio_tensor.to(self.device)
        with torch.no_grad():
            # Encode: (1, 1, Codebooks, Time) -> (Codebooks, Time)
            codes = self.audio_tokenizer.encode(audio_tensor.unsqueeze(0).unsqueeze(0))[0]
        
        # Add EOS column
        eos_col = torch.zeros(codes.size(0), 1).to(self.device)
        codes = torch.cat([codes, eos_col], dim=1)
        
        # Create frame
        seq_len = codes.size(1)
        frame = torch.zeros(seq_len, 33).long().to(self.device)
        mask = torch.zeros(seq_len, 33).bool().to(self.device)
        
        # Fill first 32 columns with audio codes
        frame[:, :-1] = codes.transpose(0, 1)
        mask[:, :-1] = True
        return frame, mask

    def _prepare_context(self, history: List[Segment]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Internal: Stitches all previous segments into one big context tensor."""
        all_tokens, all_masks = [], []
        
        for seg in history:
            # Tokenize text part
            t_tok, t_mask = self._tokenize_text(seg.text, seg.speaker)
            all_tokens.append(t_tok)
            all_masks.append(t_mask)
            
            # Tokenize audio part (if exists)
            if seg.audio is not None:
                a_tok, a_mask = self._tokenize_audio(seg.audio)
                all_tokens.append(a_tok)
                all_masks.append(a_mask)
                
        return torch.cat(all_tokens, dim=0), torch.cat(all_masks, dim=0)

    def generate(self, text: str, history: List[Segment], speaker_id: int = 0) -> Segment:
        """
        Step 2: Generation
        Takes the history and new text, runs the model, and returns a new Segment with audio.
        """
        print(f"Generating audio for: '{text}'")
        self.model.reset_caches()

        # A. Prepare Inputs
        context_tokens, context_mask = self._prepare_context(history)
        target_tokens, target_mask = self._tokenize_text(text, speaker_id)
        
        # Combine history + current text prompt
        input_tokens = torch.cat([context_tokens, target_tokens], dim=0)
        input_mask = torch.cat([context_mask, target_mask], dim=0)

        # B. Setup Loop (CHANGE)
        # max_audio_length_ms = 10_000
        max_new_tokens = int(10 * 1000 / 80) # ~10 seconds max
        curr_tokens = input_tokens.unsqueeze(0)
        curr_mask = input_mask.unsqueeze(0)
        curr_pos = torch.arange(0, input_tokens.size(0)).unsqueeze(0).long().to(self.device)
        
        new_samples = []

        # C. Autoregressive Loop
        with torch.inference_mode():
            for _ in range(max_new_tokens):
                # Predict next token
                next_token = self.model.generate_frame(
                    curr_tokens, curr_mask, curr_pos, temperature=0.8, topk=40
                )
                
                # Check for silence/EOS
                if torch.all(next_token == 0):
                    break
                
                new_samples.append(next_token)
                
                # Prepare next input (Audio code + blank text placeholder)
                next_input = torch.cat([next_token, torch.zeros(1, 1).long().to(self.device)], dim=1).unsqueeze(1)
                next_mask_val = torch.cat([torch.ones_like(next_token).bool(), torch.zeros(1, 1).bool().to(self.device)], dim=1).unsqueeze(1)
                
                curr_tokens = next_input
                curr_mask = next_mask_val
                curr_pos = curr_pos[:, -1:] + 1

        # D. Decode Codes to Audio
        if not new_samples:
            return Segment(text=text, speaker=speaker_id, audio=torch.zeros(1))

        audio_codes = torch.stack(new_samples).permute(1, 2, 0)
        with torch.no_grad():
            final_audio = self.audio_tokenizer.decode(audio_codes).squeeze(0).squeeze(0)
            
        return Segment(text=text, speaker=speaker_id, audio=final_audio.cpu())

    def save_raw_audio(self, audio_data: Union[torch.Tensor, List[torch.Tensor]], filename: str):
        """
        Step 3: Save
        Takes a single tensor OR a list of tensors, joins them, and saves immediately.
        No silence insertion.
        """
        if isinstance(audio_data, list):
            # Filter empty tensors just in case
            valid = [a for a in audio_data if a is not None and a.numel() > 1]
            if not valid:
                print("No audio to save.")
                return
            final_tensor = torch.cat(valid, dim=0)
        else:
            final_tensor = audio_data

        if final_tensor.dim() == 1:
            final_tensor = final_tensor.unsqueeze(0) # Add channel dim

        torchaudio.save(filename, final_tensor.cpu(), self.sample_rate)
        print(f"--- Raw audio saved to {filename} ---")

    def save_conversation(self, segments: List[Segment], filename: str, silence_duration: float = 0.5):
        """
        Stitches segments together with silence gaps and saves to WAV.
        """
        valid_audio = [s.audio for s in segments if s.audio is not None and s.audio.numel() > 1]
        
        if not valid_audio:
            print("No audio generated to save.")
            return

        silence = torch.zeros(int(self.sample_rate * silence_duration))
        final_sequence = []
        
        for i, audio in enumerate(valid_audio):
            final_sequence.append(audio)
            if i < len(valid_audio) - 1: # Don't add silence after the last one
                final_sequence.append(silence)
                
        full_audio = torch.cat(final_sequence, dim=0)
        torchaudio.save(filename, full_audio.unsqueeze(0), self.sample_rate)
        print(f"--- Saved output to {filename} ---")

# --- Execution Block ---
if __name__ == "__main__":
    # 1. Instantiate the Class (Loads models)
    tts = FileTTS()
    history = []
    
    # 2. Define Reference Audio (Voice Cloning Source)
    # --- SETUP MULTI-SPEAKER HISTORY ---
    # To use multiple speakers, we should ideally load a reference file for each ID.
    # If you only load Speaker 0, Speaker 1 might sound generic or same as 0.

    # Load Reference for Speaker 0
    speaker0_path = tts.get_voice_path("ryan.wav")
    if os.path.exists(speaker0_path):
        print(f"Loading Speaker 0 Reference from {speaker0_path}...")
        ref_0 = tts.load_reference_audio(speaker0_path)
        history.append(Segment(text="The sun was setting slowly, casting long shadows across the empty field.", speaker=0, audio=ref_0))
    
    # Load Reference for Speaker 1 (Optional: Use a different file!)
    speaker1_path = tts.get_voice_path("sesame.wav")
    if os.path.exists(speaker1_path):
        print(f"Loading Speaker 1 Reference from {speaker1_path}...")
        ref_1 = tts.load_reference_audio(speaker1_path)
        history.append(Segment(text="Ardent in the prosecution of heresy, Cyril auspiciously opened his reign by oppressing the Novatians, the most innocent and harmless of the sectaries.", speaker=1, audio=ref_1))
    
    # If no files exist, we start empty (Warning: Quality will be low)
    if not history:
        print("Warning: No reference audio found. Generation will be unconditioned.")

    # --- DEFINE DIALOGUE ---
    # Now we pass a list of DICTIONARIES, so each sentence has a specific speaker ID
    dialogue_script = [
        {"text": "I need you to put that on my desk immediately.", "speaker": 0},
        {"text": "The weather is surprisingly nice today.", "speaker": 1},
        {"text": "Why did you do that?", "speaker": 0},
        {"text": "I really don't know.", "speaker": 1},
    ]

    generated_segments = []
    raw_audio_list = [] # For the simple save function

    # 5. Run Generation Loop
    print("--- Starting Generation ---")
        
    for turn in dialogue_script:
        # 1. Generate
        # We extract speaker ID from the dict and pass it to generate()
        new_seg = tts.generate(turn["text"], history, speaker_id=turn["speaker"])
        
        # 2. Update History & Lists
        history.append(new_seg)
        generated_segments.append(new_seg)
        
        # Collect raw audio tensor for the simple save test
        if new_seg.audio is not None:
            raw_audio_list.append(new_seg.audio)

    # --- SAVE OUTPUTS ---
    # Method A: Save the nice conversation with silence gaps
    tts.save_conversation(generated_segments, "final_conversation.wav", silence_duration=0.5)
    
    # Method B: Save the raw audio (just concatenated, no silence)
    # tts.save_raw_audio(raw_audio_list, "final_raw_output.wav")