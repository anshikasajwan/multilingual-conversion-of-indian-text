import torch
from huggingface_hub import login
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit import IndicProcessor
import os

HF_TOKEN = os.getenv("HF_TOKEN")

# HF Token authentication

login(token=HF_TOKEN)

MODEL_NAME = "ai4bharat/indictrans2-indic-indic-dist-320M"

print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME, 
    trust_remote_code=True,
    token=HF_TOKEN
)

model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME, 
    trust_remote_code=True,
    dtype=torch.float32,
    token=HF_TOKEN
)

ip = IndicProcessor(inference=True)

# Select hardware acceleration if available
device = "mps" if torch.backends.mps.is_available() else "cpu"
model.to(device)
print(f"Model successfully loaded on: {device}")

def translate_sentence(text, src_lang="hin_Deva", tgt_lang="tam_Taml"):
    batch = ip.preprocess_batch([text], src_lang=src_lang, tgt_lang=tgt_lang)
    
    inputs = tokenizer(
        batch,
        padding="longest",
        truncation=True,
        max_length=256,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        generated_tokens = model.generate(
            **inputs,
            use_cache=False,        # <--- Fixed: Set to False for IndicTrans2 compatibility
            min_length=0,
            max_length=256,
            num_beams=4,
            num_return_sequences=1,
        )

    with tokenizer.as_target_tokenizer():
        decoded_tokens = tokenizer.batch_decode(
            generated_tokens.detach().cpu().tolist(),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

    translated = ip.postprocess_batch(decoded_tokens, lang=tgt_lang)
    return translated[0]

if __name__ == "__main__":
    sample_text = "नमस्ते, आप कैसे हैं?"
    print("\n--- Translation Test ---")
    print("Source (Hindi):", sample_text)
    
    tamil_output = translate_sentence(sample_text, src_lang="hin_Deva", tgt_lang="tam_Taml")
    print("Target (Tamil):", tamil_output)

    bengali_output = translate_sentence(sample_text, src_lang="hin_Deva", tgt_lang="ben_Beng")
    print("Target (Bengali):", bengali_output)