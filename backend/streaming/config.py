import os
from dotenv import load_dotenv

load_dotenv()

# Environment and Model Setup
HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_PATH = os.getenv("MODEL_PATH", "sesame/csm-1b")
DEVICE = os.getenv("DEVICE", "cuda")

# Generation Parameters
SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", "24000"))
MAX_MS = int(os.getenv("MAX_MS", "60000"))
FIRST_CHUNK_FRAMES = int(os.getenv("FIRST_CHUNK_FRAMES", "20"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.8"))
TOPK = int(os.getenv("TOPK", "50"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "20"))

# Voice Cloning Definitions
VOICE_DEFINITIONS = [
    {
        "name": "alice",
        "filename": "alice.wav",
        "speaker_id": 0,
        "text": "Its name and its situation were the same as those of the old place of carrying out the terrible sentence inflicted on accused persons who stood mute.",
    },
    {
        "name": "chandrika",
        "filename": "chandrika.wav",
        "speaker_id": 1,
        "text": "Humans are naturally curious. From the earliest days of civilization, people have looked at the stars, the oceans, and the forests and wondered what lies beyond. Curiosity drives innovation, asking questions that lead to discoveries, inventions, and new ways of thinking. Every child exploring a garden or a rock is practicing the same impulse that has led to great scientific achievements. Learning begins with observation, and observation begins with wonder.",
    },
    {
        "name": "ryan",
        "filename": "ryan.wav",
        "speaker_id": 2,
        "text": "The sun was setting slowly, casting long shadows across the empty field.",
    },
    {
        "name": "sesame",
        "filename": "sesame.wav",
        "speaker_id": 3,
        "text": "Ardent in the prosecution of heresy, Cyril auspiciously opened his reign by oppressing the Novatians, the most innocent and harmless of the sectaries.",
    },
]