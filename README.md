
# 🏥 Multilingual Medical Dialogue Summarization and Question Answering System

## 📘 Introduction

This project is part of the **NLP-AI4Health Shared Task (2025)**, focusing on **multilingual and low-resource NLP for healthcare**.  
It integrates two key components:

1. **Dialogue Summarization using mT5** – generating concise summaries of multilingual medical dialogues.  
2. **Question Answering using RAG** – providing accurate and safe medical responses through retrieval-augmented generation.

---

## 🧠 1. Multilingual Medical Dialogue Summarization (mT5)

### 🎯 Objective
Develop a system that summarizes patient–provider dialogues into concise, clinically meaningful summaries while preserving essential medical information.

### 💾 Dataset (Closed Task)
The **SharedTask_NLPAI4Health_Train&Dev** dataset includes multilingual dialogues in domains like **Head and Neck Cancer** and **Cystic Fibrosis**.  
Each language folder contains dialogues (`.jsonl`), Q&A pairs, key–value summaries, and text summaries.
Our Github Repository Code:- https://github.com/addygeek/Nlp4Health-Samvad-Team-IIIT-Surat

### ⚙️ Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Open `generate_summaries.py` in **Google Colab (GPU runtime)**.
3. MyDrive/NLP_A14_Health/mT5_Baseline_Model/checkpoint-9135/
4. Model_Path:- https://drive.google.com/drive/folders/1BloBv_EP8EG4EbpDg1wyz5n-UQupe5wU?usp=drive_link
5. Run all cells to produce `submission.zip` containing generated summaries.

### 📈 Output
A fine-tuned **mT5 model** that generates concise, multilingual dialogue summaries.

---

## 🤖 2. Multilingual Medical Question Answering using RAG

### 🔍 Overview
The **Enterprise Multilingual Medical RAG System** combines retrieval and generation to provide multilingual, safe, and contextually accurate medical answers.  
It uses **`sarvamai/sarvam-1`** as the generation model.

### ✨ Key Features
- **Semantic Retrieval:** Embedding-based search using `SentenceTransformer` and FAISS.  
- **Answer Generation:** Context-aware response generation using multilingual "sarvamai/sarvam-1" model.  
- **Language Detection & Translation:** Handles multilingual inputs for consistent processing.  
- **Confidence Scoring:** Evaluates answer reliability using retrieval and linguistic metrics.  
- **Safety Validation:** Filters unsafe or restricted medical topics with severity-based alerts.  
- **Caching & Logging:** Speeds up repeat queries and maintains compliance logs.

### 🧩 Core Components
| Module | Description |
|---------|-------------|
| **Embedding Model** | SentenceTransformer (`"intfloat/e5-large"`) generates dense multilingual embeddings. |
| **Retriever** | FAISS (Inner Product) for efficient similarity search across QA datasets. |
| **Answer Generator** | sarvamai/sarvam-1 models generate patient-friendly answers with context awareness and also in the same language as the user's question. |
| **Confidence Scorer** | Combines retrieval quality, language detection, and safety checks. |
| **Safety Guard** | Detects emergencies and restricted topics; ensures medical accuracy. |



### 🔄 RAG Pipeline Flow
1. **Embedding Generation:** Create dense, normalized embeddings for all QA pairs.  
2. **Retrieval:** Use FAISS to fetch top-K relevant entries with optional language filtering.  
3. **Answer Generation:** Generate safe, coherent answers using top contexts and prompts.  
4. **Confidence Scoring:** Compute reliability based on retrieval relevance and safety.
   <img width="831" height="821" alt="image" src="https://github.com/user-attachments/assets/8d3fab2c-04ef-4c8b-8342-fefbf0fdaedc" />


### 🧰 How to Run
1. Open `rag_pipeline_QnA.py` in Google Colab.  
2. Update model and dataset paths in the notebook.  
3. Run all cells sequentially to execute the full RAG pipeline.

---
## Other Scripts:
- **mT5 Summarization Training Script:** `mT5 Summarization Training Script (Final Corrected Version.ipynb)`  
- **Evaluation Script:** `Evaluation_Script.ipynb` for evaluating generated summaries
-  BASELINE MODEL (Original Data) ---
  ROUGE-L Score: 12.41
  BERTScore (F1): 80.48

--- NEW MODEL (Cleaned Data, Stable Training) ---
  ROUGE-L Score: 8.81
  BERTScore (F1): 56.74





## ⚙️ Hardware and Runtime

| Component | Recommended |
|------------|--------------|
| **GPU** | NVIDIA A100 / V100 |
| **Python** | 3.9+ |
| **Libraries** | Transformers, Datasets, SentenceTransformers, FAISS, Langdetect |
| **Platform** | Google Colab (GPU runtime) |

---

## 🏁 Summary

This project delivers a **comprehensive multilingual medical NLP solution**, combining:  
- **mT5-based dialogue summarization** for concise, domain-specific summaries.  
- **RAG-based question answering** for safe, multilingual, and explainable medical assistance.


**Authors:** Rakesh Kumar, Aditya Kumar, Devansh Khushwaha, *Janhavi Naik*  
**Affiliation:** *NLP-AI4Health Workshop Participants*  
**Year:** 2025
