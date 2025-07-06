import torch
import whisper

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def load_model():
    device = get_device()
    model = whisper.load_model("base").to(device)
    return model, device 