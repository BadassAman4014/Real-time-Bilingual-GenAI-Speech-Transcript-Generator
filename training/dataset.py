import json
from torch.utils.data import Dataset
import torchaudio

class SpeechDataset(Dataset):
    def __init__(self, manifest_path):
        """
        manifest_path: path to train.json (jsonl) with fields:
            - audio: path to audio file
            - text: transcript
            - lang: optional language code (hi/en)
        """
        self.manifest_path = manifest_path
        self.data = []

        # Load JSONL manifest
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                self.data.append(item)

    def __len__(self):
        return len(self.data)

    def load_audio(self, path):
        waveform, sr = torchaudio.load(path)
        if sr != 16000:
            waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
        return waveform.squeeze(0)

    def __getitem__(self, idx):
        item = self.data[idx]
        audio = self.load_audio(item["audio"])
        return {
            "audio": audio,
            "text": item["text"],
            "lang": item.get("lang", None)
        }

# Quick test
if __name__ == "__main__":
    ds = SpeechDataset("data/manifests/train.json")
    sample = ds[0]

    print(sample["audio"].shape)
    print(sample["text"])
    print(sample["lang"])
