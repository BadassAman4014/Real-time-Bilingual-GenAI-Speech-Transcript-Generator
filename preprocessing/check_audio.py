import torchaudio
from pathlib import Path

base = Path("data/raw/en/clips")

for wav in base.rglob("*.wav"):
    waveform, sample_rate = torchaudio.load(str(wav))
    if sample_rate != 16000:
        print("❌", wav, sample_rate)
