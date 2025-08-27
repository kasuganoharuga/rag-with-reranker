# RAG with Reranker (Accuracy Improvement)

This repo demonstrates a Retrieval-Augmented Generation (RAG) pipeline enhanced with a reranker (BGE-reranker-base).  
It includes small sample data and evaluation scripts. See details in the sections below.

## Quickstart
\\\ash
pip install -r requirements.txt
python cutter.py                 # split text/ into chunks and build index
python AccuracyRefinedRAG.py     # run RAG with reranker
python analyze_rag_accuracy.py   # evaluate accuracy
\\\


