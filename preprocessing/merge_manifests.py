from pathlib import Path

out = open("data/manifests/train.jsonl", "w", encoding="utf-8")

for file in ["train_en.jsonl", "train_hi.jsonl"]:
    path = Path("data/manifests") / file
    if not path.exists():
        continue
    for line in open(path, encoding="utf-8"):
        out.write(line)

out.close()
print("✅ Combined multilingual manifest ready")
