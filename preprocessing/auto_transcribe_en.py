import torch
import json
from pathlib import Path
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torchaudio

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = WhisperProcessor.from_pretrained("openai/whisper-small")
model = WhisperForConditionalGeneration.from_pretrained(
    "openai/whisper-small"
).to(device)

AUDIO_DIR = Path("data/raw/en/clips")
OUT_PATH = "data/manifests/train_en.jsonl"

with open(OUT_PATH, "w", encoding="utf-8") as out:
    for audio_path in AUDIO_DIR.glob("*.mp3"):
        waveform, sr = torchaudio.load(audio_path)
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)

        inputs = processor(waveform.squeeze(), sampling_rate=16000, return_tensors="pt").to(device)

        with torch.no_grad():
            pred_ids = model.generate(**inputs)

        text = processor.decode(pred_ids[0], skip_special_tokens=True)

        out.write(json.dumps({
            "audio": str(audio_path),
            "text": text,
            "lang": "en"
        }) + "\n")

print("✅ English pseudo-label manifest created")
