import torch
from huggingface_hub import login
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit import IndicProcessor

class IndicTranslator:
    def __init__(self, model_name="ai4bharat/indictrans2-indic-indic-dist-320M", hf_token=None):
        if hf_token:
            login(token=hf_token)
            
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            trust_remote_code=True,
            token=hf_token
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, 
            trust_remote_code=True,
            dtype=torch.float32,
            token=hf_token
        ).to(self.device)
        
        self.ip = IndicProcessor(inference=True)

    def translate(self, text: str, src_lang: str = "hin_Deva", tgt_lang: str = "tam_Taml") -> str:
        batch = self.ip.preprocess_batch([text], src_lang=src_lang, tgt_lang=tgt_lang)
        
        inputs = self.tokenizer(
            batch,
            padding="longest",
            truncation=True,
            max_length=256,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            generated_tokens = self.model.generate(
                **inputs,
                use_cache=False,
                min_length=0,
                max_length=256,
                num_beams=4,
                num_return_sequences=1,
            )

        decoded_tokens = self.tokenizer.batch_decode(
            generated_tokens.detach().cpu().tolist(),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        translated = self.ip.postprocess_batch(decoded_tokens, lang=tgt_lang)
        return translated[0]