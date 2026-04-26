import json

INPUT_FILE = "data/splits/train.json"
OUTPUT_FILE = "data/splits/train_formatted.json"


def format_example(record):
    input_text = f"""
Explain this Indian law in simple terms:

Title: {record['title']}

Text:
{record['raw_text']}
"""

    output_text = record["simple_summary"]

    return {
        "input": input_text.strip(),
        "output": output_text.strip()
    }


with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

formatted = []

for rec in data:
    if rec.get("simple_summary"):  # skip empty ones
        formatted.append(format_example(rec))

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(formatted, f, indent=2, ensure_ascii=False)

print(f"Created {len(formatted)} training samples")