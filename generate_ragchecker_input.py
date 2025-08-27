import os
import json
import pandas as pd
import ast

# === Path Configuration ===
rag_csv = r"C:\Users\49765\Desktop\Neural craft lab\5.19准确率提升\rag_accuracy_results_improvedrag.csv"
answer_csv = r"C:\Users\49765\Desktop\Neural craft lab\5.19准确率提升\answer.csv"
chunk_folder = r"C:\Users\49765\Desktop\Neural craft lab\5.19准确率提升\cuttedpage"
output_json = r"C:\Users\49765\Desktop\Neural craft lab\5.19准确率提升\ragchecker_input.json"

# === Load Data ===
rag_df = pd.read_csv(rag_csv)
rag_df["predicted_sources"] = rag_df["predicted_sources"].apply(ast.literal_eval)

gt_df = pd.read_csv(answer_csv)

# === Build a mapping from question to GT Answer ===
gt_map = dict(zip(gt_df["question"], gt_df["gt_answer"]))

results = []

for idx, row in rag_df.iterrows():
    query_id = str(idx).zfill(3)
    query = row["question"]
    response = row["answer"]
    sources = row["predicted_sources"]

    # === Fetch GT Answer from answer.csv ===
    gt_answer = gt_map.get(query, "")
    if not gt_answer:
        print(f"⚠️ GT answer not found for question: {query}")
        continue

    # === Retrieve context chunks ===
    retrieved_context = []
    for source in sources:
        if "#chunk=" not in source:
            print(f"❌ Skipping invalid source format: {source}")
            continue
        try:
            chunk_id = int(source.split("#chunk=")[-1])
        except:
            print(f"⚠️ Failed to extract chunk ID from: {source}")
            continue

        chunk_filename = f"page{chunk_id}.txt"
        chunk_path = os.path.join(chunk_folder, chunk_filename)

        if not os.path.exists(chunk_path):
            print(f"⚠️ Chunk file not found: {chunk_filename}")
            continue

        with open(chunk_path, "r", encoding="utf-8") as f:
            chunk_text = f.read()

        retrieved_context.append({
            "doc_id": source,
            "text": chunk_text
        })

    # === Combine fields into RAGChecker format ===
    entry = {
        "query_id": query_id,
        "query": query,
        "gt_answer": gt_answer,
        "response": response,
        "retrieved_context": retrieved_context
    }

    results.append(entry)

# === Write output JSON ===
with open(output_json, "w", encoding="utf-8") as f:
    json.dump({"results": results}, f, ensure_ascii=False, indent=2)

print(f"✅ ragchecker_input.json has been generated successfully with {len(results)} entries.")
