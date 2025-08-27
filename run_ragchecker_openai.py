import os
from ragchecker import RAGResults, RAGChecker
from ragchecker.metrics import all_metrics

# ✅ 设置 OpenAI API Key
os.environ["OPENAI_API_KEY"] = ""

# ✅ 加载 JSON 输入文件
with open("C:/Users/49765/Desktop/Neural craft lab/5.19准确率提升/ragchecker_input.json", "r", encoding="utf-8") as f:
    rag_results = RAGResults.from_json(f.read())

# ✅ 初始化 RAGChecker，使用 GPT-4.1
evaluator = RAGChecker(
    extractor_name="openai/gpt-4.1-nano",
    checker_name="openai/gpt-4.1-nano",
    batch_size_extractor=16,
    batch_size_checker=32
)

# ✅ 运行评估
evaluator.evaluate(rag_results, all_metrics)

# ✅ 输出指标结果
print("=== RAGChecker Evaluation Complete ===")
print(rag_results)

# ✅ 保存为输出 JSON
output_path = "C:/Users/49765/Desktop/Neural craft lab/5.19准确率提升/ragchecker_output.json"
with open(output_path, "w", encoding="utf-8") as f:
    f.write(rag_results.to_json(indent=2))
