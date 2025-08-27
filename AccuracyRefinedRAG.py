import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.runnables import RunnableMap, RunnableLambda
from langchain_core.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.memory import ConversationBufferMemory
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import BM25Retriever, EnsembleRetriever

# === 1. Load local model ===
llm = OllamaLLM(model="llama3.1", temperature=0.2)
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# === 2. Load and split text documents ===
folder_path = r"C:\Users\49765\Desktop\Neural craft lab\5.19准确率提升\text"
raw_documents = []
for filename in os.listdir(folder_path):
    if filename.endswith(".txt"):
        filepath = os.path.join(folder_path, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
            raw_documents.append(Document(page_content=content, metadata={"source": filename}))

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
split_documents = []
for doc in raw_documents:
    chunks = splitter.split_documents([doc])
    for i, chunk in enumerate(chunks):
        # Add chunk number to source name
        original_source = doc.metadata["source"]
        chunk.metadata["source"] = f"{original_source}#chunk={i+1}"
        split_documents.append(chunk)


# === 3. Build Hybrid Retriever ===
# Vector (FAISS)
if os.path.exists("faiss_index"):
    vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
else:
    vectorstore = FAISS.from_documents(split_documents, embeddings)
    vectorstore.save_local("faiss_index")
vector_retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 20})

# Sparse (BM25)
bm25_retriever = BM25Retriever.from_documents(split_documents)
bm25_retriever.k = 20

# Hybrid Retriever
retriever = EnsembleRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    weights=[0.3, 0.7]
)

# === 4. Local Reranker ===
class LocalReranker:
    def __init__(self, model_path="./bge-reranker-base"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()

    def rerank(self, query, documents, top_k=8):
        pairs = [(query, doc.page_content) for doc in documents]
        inputs = self.tokenizer.batch_encode_plus(pairs, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            scores = self.model(**inputs).logits.squeeze(-1)
        sorted_docs = [doc for _, doc in sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)]
        return sorted_docs[:top_k]

reranker = LocalReranker(model_path="./bge-reranker-base")

# === 5. Query Rewriter ===
rewrite_prompt = PromptTemplate.from_template("""
You are a helpful assistant that rewrites follow-up questions to standalone queries.
Chat History:
{history}
Follow-up Question: {question}
Rewritten Standalone Question:""")
rewrite_chain = rewrite_prompt | llm | StrOutputParser()

# === 6. QA Answer Chain ===
qa_prompt = PromptTemplate.from_template("""
Each source comes from a file. The name of the file will be shown as [source_name#chunk=X]. Always include the file name and chunk ID for each fact you use, using square brackets—for example, [A SCANDAL IN BOHEMIA 5.txt#chunk=3]. Do not invent source names like [context1.txt].

Only use information from the provided context. If the answer cannot be found in the context, say "I don't know." Be concise and factual.

Context:
{context}

Question: {question}
Answer:
""")

document_chain = create_stuff_documents_chain(llm=llm, prompt=qa_prompt)

# === 7. Chat Loop ===
memory = ConversationBufferMemory(return_messages=True)
chat_history = []

if __name__ == "__main__":
    print("\n==== Hybrid RAG | Query Rewrite + Hybrid Retrieval + Reranker ====")
    while True:
        user_input = input("\n📥 Your Question: ")
        if user_input.lower() in ["exit", "quit"]:
            print("👋 Exiting.")
            break

        # Build conversation history
        history_text = "\n".join([f"User: {u}\nAI: {a}" for u, a in chat_history])

        # Rewrite question
        rewritten = rewrite_chain.invoke({"history": history_text, "question": user_input})
        print(f"\n🔁 Rewritten Question: {rewritten}")

        # Hybrid Retrieval + Rerank
        docs = retriever.invoke(rewritten)
        reranked_docs = reranker.rerank(query=rewritten, documents=docs, top_k=8)

        # QA Response
        inputs = {"context": reranked_docs, "question": rewritten}
        result = document_chain.invoke(inputs)
        print(f"\n💬 Answer:\n{result}")

        # Show sources
        print("📚 Top Sources Used:")
        for i, doc in enumerate(reranked_docs):
            print(f"{i+1}. [{doc.metadata.get('source', 'unknown')}]")


        chat_history.append((user_input, result))
