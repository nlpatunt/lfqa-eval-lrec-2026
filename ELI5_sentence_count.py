import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
import json
import matplotlib.pyplot as plt
import os

# Download the sentence tokenizer
nltk.download("punkt")

# 🔁 Set the correct path to your file
file_path = r"F:\PhD\Long form research question\archive\ELI5_val.jsonl"  # <-- update this path

# Check if file exists
if not os.path.exists(file_path):
    raise FileNotFoundError(f"File not found: {file_path}")

# Load data and extract questions
questions = []
with open(file_path, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        question = item.get("question", "").strip()
        if question:
            questions.append(question)

print(f"✅ Loaded {len(questions)} questions.")

# Show sample
print("\nSample Question:\n", questions[0])

# Count number of sentences per question
sentence_counts = [len(sent_tokenize(q)) for q in questions]

# Create DataFrame
df = pd.DataFrame(sentence_counts, columns=["sentence_count"])

# Summary stats
print("\n📊 Sentence Count Stats:")
print(df.describe())

# Frequency counts
print("\n🔢 Frequency of Sentence Counts:")
print(df["sentence_count"].value_counts().sort_index())

# Plot histogram
plt.figure(figsize=(10, 5))
plt.hist(df["sentence_count"], bins=range(1, max(df["sentence_count"]) + 2), edgecolor="black", align="left")
plt.xticks(range(1, max(df["sentence_count"]) + 2))
plt.xlabel("Number of Sentences per Question")
plt.ylabel("Frequency")
plt.title("Sentence Count Distribution in ELI5 Questions")
plt.grid(axis="y")
plt.tight_layout()
plt.show()
