import re
import pandas as pd
from AccuracyRefinedRAG import rewrite_chain, retriever, reranker, document_chain

# === 加载问题文件 ===
df = pd.read_csv("Questions.csv")

results = []

# === 预处理 expected_source 为 chunk ID 格式 ===
def normalize_expected(source):
    match = re.match(r"(.*)\s(\d+)\.txt", str(source))
    if match:
        base, chunk = match.groups()
        return f"{base}.txt#chunk={int(chunk)}"
    return str(source)

for idx, row in df.iterrows():
    question = row["question"]
    expected_source = normalize_expected(row["expected_source"])  # ✅ 转换为 chunk 格式

    # === 重写问题 ===
    rewritten = rewrite_chain.invoke({"history": "", "question": question})

    # === 检索 + rerank ===
    docs = retriever.invoke(rewritten)
    reranked_docs = reranker.rerank(query=rewritten, documents=docs, top_k=8)

    # === 回答（模型会内嵌 source）
    result = document_chain.invoke({"context": reranked_docs, "question": rewritten})

    # === 实际使用的 source（用于准确率判断）
    actual_sources = {doc.metadata.get("source", "unknown") for doc in reranked_docs}
    is_correct = expected_source in actual_sources

    # === 提取模型生成的 source 标签（用于双轨分析）
    generated_sources = re.findall(r"\[([^\[\]]+?\.txt#chunk=\d+)\]", result)

    # === 存储结果
    results.append({
        "question": question,
        "rewritten": rewritten,
        "expected_source": expected_source,
        "predicted_sources": list(actual_sources),
        "generated_sources": list(set(generated_sources)),
        "is_correct": is_correct,
        "answer": result
    })

    # === 双轨显示
    print("\n============================")
    print(f"📌 Question: {question}")
    print(f"🔁 Rewritten: {rewritten}")
    print(f"✅ Is Correct: {is_correct}")
    print(f"💬 Model Answer:\n{result}")
    print(f"📚 source (from metadata): {sorted(actual_sources)}")
    print(f"📑 according (from model): {sorted(set(generated_sources))}")

# === 保存输出 ===
df_results = pd.DataFrame(results)
df_results.to_csv("rag_accuracy_results_improvedrag.csv", index=False)
