import json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

INPUT_FILE = "data/processed/all_records.json"
OUTPUT_FILE = "data/raw/summaries.json"

model_name = "google/flan-t5-small"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


def generate_summary(text):
    prompt = f"""
Summarize this legal text in simple English:

{text[:1000]}

Summary:
"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)

    outputs = model.generate(**inputs, max_new_tokens=80)

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


with open(INPUT_FILE, "r", encoding="utf-8") as f:
    records = json.load(f)

summaries = {}

for i, rec in enumerate(records):
    print(f"Processing {i+1}/{len(records)}")

    summary = generate_summary(rec["raw_text"])
    summaries[rec["id"]] = summary


with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(summaries, f, indent=2, ensure_ascii=False)

print(" summaries.json created!")