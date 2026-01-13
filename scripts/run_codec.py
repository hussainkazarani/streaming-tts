import torch
from huggingface_hub import hf_hub_download
from moshi.models import loaders

print(loaders.DEFAULT_REPO,loaders.MIMI_NAME)
def test_codec():
    mimi_weight = hf_hub_download(loaders.DEFAULT_REPO,loaders.MIMI_NAME)
    mimi = loaders.get_mimi(mimi_weight)
    audio = torch.zeros(1,1,mimi.sample_rate)
    tokens = mimi.encode(audio)
    print(f"Encoded Tokens: {tokens}")
    decoded_audio = mimi.decode(tokens)
    print(f"Decoded Audio Shape: {decoded_audio.shape}")

test_codec()