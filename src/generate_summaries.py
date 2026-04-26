import json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

INPUT_FILE = "data/processed/all_records.json"
OUTPUT_FILE = "data/processed/all_records_with_summaries.json"

model_name = "google/flan-t5-small"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


def generate_summary(text):
    prompt = f"""
Summarize this legal text in simple terms:

{text[:1000]}

Summary:
"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)

    outputs = model.generate(
        **inputs,
        max_new_tokens=100
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

for i, record in enumerate(data):
    if not record.get("simple_summary"):
        print(f"Processing {i+1}/{len(data)}...")
        record["simple_summary"] = generate_summary(record["raw_text"])

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print(" Summaries generated!")