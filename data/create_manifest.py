import csv
import json
from pathlib import Path

# Paths
data_dir = Path("data")
manifests_dir = data_dir / "manifests"
manifests_dir.mkdir(exist_ok=True)

train_json_path = manifests_dir / "train.json"

# TSV files
tsv_files = {
    "en": {"path": data_dir / "raw/en/train.tsv", "has_header": False},
    "hi": {"path": data_dir / "raw/hi/train.tsv", "has_header": True}
}

# Output JSON lines
with open(train_json_path, "w", encoding="utf-8") as outfile:
    for lang, info in tsv_files.items():
        tsv_path = info["path"]
        has_header = info["has_header"]

        with open(tsv_path, "r", encoding="utf-8") as f:
            if has_header:
                reader = csv.DictReader(f, delimiter="\t")
                for row in reader:
                    # Automatically detect audio and text columns
                    audio_file = next((row[c] for c in ['path', 'filename', 'audio'] if c in row), None)
                    text = next((row[c] for c in ['sentence', 'text', 'transcript'] if c in row), None)

                    if not audio_file or not text:
                        continue

                    audio_path = str(data_dir / f"raw/{lang}/clips" / audio_file)

                    json_line = {
                        "audio": audio_path,
                        "text": text,
                        "lang": lang
                    }
                    outfile.write(json.dumps(json_line, ensure_ascii=False) + "\n")
            else:
                # No header, assume first column = audio, second column = transcript
                reader = csv.reader(f, delimiter="\t")
                for row in reader:
                    if len(row) < 2:
                        continue
                    audio_file, text = row[0], row[1]
                    audio_path = str(data_dir / f"raw/{lang}/clips" / audio_file)

                    json_line = {
                        "audio": audio_path,
                        "text": text,
                        "lang": lang
                    }
                    outfile.write(json.dumps(json_line, ensure_ascii=False) + "\n")

print(f"Manifest created at {train_json_path}")
