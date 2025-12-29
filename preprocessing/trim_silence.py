import torchaudio
from torchaudio.transforms import Vad
from pathlib import Path

vad = Vad(sample_rate=16000)

RAW = Path("data/raw")
OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)

for wav in RAW.rglob("*.wav"):
    waveform, sr = torchaudio.load(wav)
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)

    trimmed = vad(waveform)

    out_path = OUT / wav.relative_to(RAW)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(out_path, trimmed, 16000)
