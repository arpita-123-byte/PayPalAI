import json

SPLIT_FILES = [
    "data/splits/train.json",
    "data/splits/val.json",
    "data/splits/test.json"
]

SUMMARIES_FILE = "data/raw/summaries.json"


# Load summaries
with open(SUMMARIES_FILE, "r", encoding="utf-8") as f:
    summaries = json.load(f)

for file in SPLIT_FILES:
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)

    updated = 0

    for rec in data:
        rec_id = rec["id"]

        if rec_id in summaries:
            rec["simple_summary"] = summaries[rec_id]
            updated += 1

    with open(file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f" Updated {updated} records in {file}")