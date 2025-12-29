import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration

device = "cuda"
model = WhisperForConditionalGeneration.from_pretrained(
    "./checkpoints"
).to(device).half()

processor = WhisperProcessor.from_pretrained("openai/whisper-small")

def transcribe(audio):
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt").to(device)
    with torch.no_grad():
        pred = model.generate(**inputs)
    return processor.decode(pred[0], skip_special_tokens=True)
