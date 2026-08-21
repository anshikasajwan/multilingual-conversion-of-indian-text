import time
import sacrebleu
from modules.translation_module import IndicTranslator

# Initialize the translation module
print("Loading model for evaluation...")
translator = IndicTranslator()

# Small test set: (Source Hindi, Ground Truth Target Tamil)
# In practice, you can add 20–100 sentence pairs from benchmarks like IN22 / FLORES-200.
test_dataset = [
    {
        "src": "नमस्ते, आप कैसे हैं?",
        "ref": "வணக்கம், நீங்கள் எப்படி இருக்கிறீர்கள்?"
    },
    {
        "src": "आज का मौसम बहुत अच्छा है।",
        "ref": "இன்றைய வானிலை மிகவும் நன்றாக உள்ளது."
    },
    {
        "src": "कृपया मुझे पानी दीजिए।",
        "ref": "தயவுசெய்து எனக்கு தண்ணீர் கொடுங்கள்."
    },
    {
        "src": "हम एक शोध पत्र लिख रहे हैं।",
        "ref": "நாங்கள் ஒரு ஆய்வுக் கட்டுரையை எழுதுகிறோம்."
    }
]

hypotheses = []  # Model predictions
references = []  # Ground truth answers
latencies = []

print("\nRunning translation evaluation (Hindi -> Tamil)...")
for item in test_dataset:
    src_text = item["src"]
    ref_text = item["ref"]
    
    start_time = time.time()
    predicted_translation = translator.translate(src_text, src_lang="hin_Deva", tgt_lang="tam_Taml")
    latency_ms = (time.time() - start_time) * 1000
    
    hypotheses.append(predicted_translation)
    references.append(ref_text)
    latencies.append(latency_ms)
    
    print(f"\n[Source]    : {src_text}")
    print(f"[Reference] : {ref_text}")
    print(f"[Model Pred]: {predicted_translation}")
    print(f"[Latency]   : {latency_ms:.2f} ms")

# --- Metric Calculations ---
# 1. SacreBLEU
bleu = sacrebleu.corpus_bleu(hypotheses, [references])

# 2. chrF++ (word_order=2 enables chrF++)
chrf_plus = sacrebleu.corpus_chrf(hypotheses, [references], word_order=2)

avg_latency = sum(latencies) / len(latencies)

# --- Final Benchmark Report for Research Paper ---
print("\n================ EVALUATION SUMMARY ================")
print(f"Total Test Samples: {len(test_dataset)}")
print(f"BLEU Score        : {bleu.score:.2f}")
print(f"chrF++ Score      : {chrf_plus.score:.2f}")
print(f"Avg Latency       : {avg_latency:.2f} ms / sentence")
print("====================================================")