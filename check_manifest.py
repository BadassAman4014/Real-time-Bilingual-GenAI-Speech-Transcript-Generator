import json

with open("data/manifests/train.json", "r", encoding="utf-8") as f:
    for line in f:
        json.loads(line)

print("OK")