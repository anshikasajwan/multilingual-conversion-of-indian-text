import torch
import soundfile as sf
from langdetect import detect
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer


# =========================================================
# MODEL
# =========================================================

MODEL_NAME = "ai4bharat/indic-parler-tts"

print("Loading Indic Parler-TTS...")

device = "cuda:0" if torch.cuda.is_available() else "cpu"

print("Device:", device)

model = ParlerTTSForConditionalGeneration.from_pretrained(
    MODEL_NAME
).to(device)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

description_tokenizer = AutoTokenizer.from_pretrained(
    model.config.text_encoder._name_or_path
)

print("Model loaded successfully!")
print()


# =========================================================
# LANGUAGE NAMES
# =========================================================

LANGUAGES = {
    "as": "Assamese",
    "bn": "Bengali",
    "brx": "Bodo",
    "doi": "Dogri",
    "en": "English",
    "gu": "Gujarati",
    "hi": "Hindi",
    "kn": "Kannada",
    "kok": "Konkani",
    "mai": "Maithili",
    "ml": "Malayalam",
    "mni": "Manipuri",
    "mr": "Marathi",
    "ne": "Nepali",
    "or": "Odia",
    "pa": "Punjabi",
    "sa": "Sanskrit",
    "sat": "Santali",
    "sd": "Sindhi",
    "ta": "Tamil",
    "te": "Telugu",
    "ur": "Urdu",
}


# =========================================================
# LANGUAGE DETECTION
# =========================================================

def detect_language(text):

    try:
        lang = detect(text)

        return lang

    except Exception:
        return None


# =========================================================
# SPEAKER DESCRIPTION
# =========================================================

def get_description(language):

    descriptions = {

        "hi":
        "Rohit speaks in a clear, natural and moderately paced male voice. "
        "The recording is very high quality with no background noise.",

        "bn":
        "Arjun speaks in a clear, natural and moderately paced male voice. "
        "The recording is very high quality with no background noise.",

        "gu":
        "Yash speaks in a clear, natural and moderately paced male voice. "
        "The recording is very high quality with no background noise.",

        "kn":
        "Suresh speaks in a clear, natural and moderately paced male voice. "
        "The recording is very high quality with no background noise.",

        "ml":
        "Harish speaks in a clear, natural and moderately paced male voice. "
        "The recording is very high quality with no background noise.",

        "mr":
        "Sanjay speaks in a clear, natural and moderately paced male voice. "
        "The recording is very high quality with no background noise.",

        "ta":
        "Jaya speaks in a clear, natural and moderately paced female voice. "
        "The recording is very high quality with no background noise.",

        "te":
        "Prakash speaks in a clear, natural and moderately paced male voice. "
        "The recording is very high quality with no background noise.",

        "pa":
        "Divjot speaks in a clear, natural and moderately paced voice. "
        "The recording is very high quality with no background noise.",

        "as":
        "Amit speaks in a clear, natural and moderately paced male voice. "
        "The recording is very high quality with no background noise.",

        "or":
        "Manas speaks in a clear, natural and moderately paced male voice. "
        "The recording is very high quality with no background noise.",

        "sa":
        "Aryan speaks in a clear, natural and moderately paced male voice. "
        "The recording is very high quality with no background noise.",

        "ne":
        "Amrita speaks in a clear, natural and moderately paced female voice. "
        "The recording is very high quality with no background noise.",

        "mni":
        "Laishram speaks in a clear, natural and moderately paced voice. "
        "The recording is very high quality with no background noise.",

        "brx":
        "Bikram speaks in a clear, natural and moderately paced male voice. "
        "The recording is very high quality with no background noise.",

        "doi":
        "Karan speaks in a clear, natural and moderately paced male voice. "
        "The recording is very high quality with no background noise.",

        "en":
        "Thoma speaks in a clear, natural and moderately paced male voice. "
        "The recording is very high quality with no background noise.",
    }

    return descriptions.get(
        language,
        "A clear, natural and moderately paced speaker delivers the speech. "
        "The recording is very high quality with no background noise."
    )


# =========================================================
# GENERATE SPEECH
# =========================================================

def generate_speech(text, number):

    language = detect_language(text)

    print()
    print("Detected language code:", language)

    if language in LANGUAGES:
        print("Detected language:", LANGUAGES[language])
    else:
        print("Language detected:", language)

    # -----------------------------------------------------
    # Check whether language is supported
    # -----------------------------------------------------

    if language not in LANGUAGES:

        print()
        print("This language is not in the supported language list.")
        print("Please enter text in a supported language.")
        return

    # -----------------------------------------------------
    # Prepare description
    # -----------------------------------------------------

    description = get_description(language)

    description_input = description_tokenizer(
        description,
        return_tensors="pt"
    ).to(device)

    prompt_input = tokenizer(
        text,
        return_tensors="pt"
    ).to(device)

    # -----------------------------------------------------
    # Generate
    # -----------------------------------------------------

    print("Generating speech...")

    with torch.no_grad():

        generation = model.generate(
            input_ids=description_input.input_ids,
            attention_mask=description_input.attention_mask,
            prompt_input_ids=prompt_input.input_ids,
            prompt_attention_mask=prompt_input.attention_mask
        )

    # -----------------------------------------------------
    # Convert to audio
    # -----------------------------------------------------

    audio = generation.cpu().numpy().squeeze()

    filename = f"output_{number}.wav"

    sf.write(
        filename,
        audio,
        model.config.sampling_rate
    )

    print()
    print("Speech generated successfully!")
    print("Audio file:", filename)
    print("Sample rate:", model.config.sampling_rate)
    print()


# =========================================================
# TERMINAL LOOP
# =========================================================

print("==============================================")
print("       INDIC PARLER-TTS TERMINAL")
print("==============================================")
print()
print("Type any supported Indian-language text.")
print("Type 'exit' to close the program.")
print()

count = 1

while True:

    text = input("Enter text: ")

    if text.lower() == "exit":
        print("Program closed.")
        break

    if not text.strip():
        print("Please enter some text.")
        continue

    generate_speech(text, count)

    count += 1