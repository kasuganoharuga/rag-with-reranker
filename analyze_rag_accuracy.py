import pandas as pd
import ast
import re

# === Load the result CSV ===
df = pd.read_csv("rag_accuracy_results_improvedrag.csv")

# === Convert stringified lists into actual Python lists ===
df["predicted_sources"] = df["predicted_sources"].apply(ast.literal_eval)

# === Extract expected chunk from expected_source ===
def normalize_expected(source):
    if isinstance(source, int):
        return f"A SCANDAL IN BOHEMIA.txt#chunk={source}"
    match = re.match(r"(.*)\s(\d+)\.txt", str(source))
    if match:
        base, chunk = match.groups()
        return f"{base}.txt#chunk={int(chunk)}"
    return str(source)

df["expected_chunk"] = df["expected_source"].apply(normalize_expected)

# === Function to compute hit@k ===
def compute_hit_at_k(row, k):
    return row["expected_chunk"] in row["predicted_sources"][:k]

# === Add Hit@K metrics ===
for k in [1, 3, 5]:
    df[f"hit@{k}"] = df.apply(lambda row: compute_hit_at_k(row, k), axis=1)

# === Recalculate is_correct ===
df["is_correct"] = df.apply(lambda row: row["expected_chunk"] in row["predicted_sources"], axis=1)

# === Calculate summary statistics ===
total = len(df)
correct = df["is_correct"].sum()
accuracy = correct / total
avg_predicted = df["predicted_sources"].apply(len).mean()
hit1 = df["hit@1"].mean()
hit3 = df["hit@3"].mean()
hit5 = df["hit@5"].mean()

summary = {
    "Total Questions": total,
    "Correct Predictions": correct,
    "Incorrect Predictions": total - correct,
    "Accuracy": round(accuracy, 4),
    "Average Predicted Sources": round(avg_predicted, 2),
    "Hit@1": round(hit1, 4),
    "Hit@3": round(hit3, 4),
    "Hit@5": round(hit5, 4),
}

# === Print the summary report ===
print("=== RAG Performance Summary ===")
for k, v in summary.items():
    print(f"{k}: {v}")
