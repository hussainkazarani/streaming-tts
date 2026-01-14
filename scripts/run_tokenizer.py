from transformers import AutoTokenizer

def test_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    text = "Hello, how are you?"
    tokens = tokenizer.encode(text)
    # manual EOS
    tokens.append(tokenizer.eos_token_id)
    decoded_text = tokenizer.decode(tokens)

    print(f"Original Text: {text}")
    print(f"Encoded Tokens: {tokens}")
    print(f"Decoded Text: {decoded_text}")


test_tokenizer()