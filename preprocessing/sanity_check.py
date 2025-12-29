import json
import os

with open("data/manifests/train.json", encoding="utf-8") as f:
    for i, line in enumerate(f):
        item = json.loads(line)
        # Remove duplicated 'clips/' if needed
        corrected_path = item["audio"].replace("clips\\clips", "clips")
        assert os.path.exists(corrected_path), f"Missing: {corrected_path}"
        assert item["lang"] in ["en", "hi"]
        if i == 10:
            break

print("✅ Manifest looks good")
