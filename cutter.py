import os
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 设置文件夹路径
folder_path = r"C:\Users\49765\Desktop\Neural craft lab\5.19准确率提升\text"

# === 确保只有一个 .txt 文件 ===
txt_files = [f for f in os.listdir(folder_path) if f.endswith(".txt")]
if len(txt_files) != 1:
    raise ValueError(f"请确保目录中只有一个 .txt 文件，目前有 {len(txt_files)} 个。")

filename = txt_files[0]
filepath = os.path.join(folder_path, filename)

# === 读取文件内容 ===
with open(filepath, "r", encoding="utf-8") as f:
    full_text = f.read()

# === 切分文本（和 ImprovedRAG.py 一致）===
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = splitter.split_text(full_text)

# === 输出每一页为 pageX.txt 文件 ===
for i, chunk in enumerate(chunks):
    page_num = i + 1
    output_filename = f"page{page_num}.txt"
    output_path = os.path.join(folder_path, output_filename)
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(chunk)

print(f"✅ 从文件 {filename} 中共生成 {len(chunks)} 页，已保存为 pageX.txt 文件。")
