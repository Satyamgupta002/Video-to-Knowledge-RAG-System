
# AI Course Assistant using RAG & LLMs

This project is a **full-scale Retrieval-Augmented Generation (RAG) system powered by Large Language Models (LLMs)** that transforms long-form course videos into **queryable, contextual knowledge**.  
It enables users to ask natural language questions and receive **context-aware answers enriched with exact video timestamps and lecture references**.

This is not a simple chatbot, it is a **production-style AI system** built to handle **hours of video content**, **semantic retrieval**, and **LLM-grounded reasoning**.

---

## What the System Does

The system bridges long-form video content and natural language queries by converting videos into searchable **context-aware knowledge**.

Users can:
- Ask questions in plain English  
- Get accurate, grounded answers  
- Instantly know **which video**, **which lecture**, and **which timestamp** explains the concept  

---

## Why This Project Stands Out

- Converts **entire video courses** into searchable AI knowledge
- Eliminates the need to manually scan through entire videos to find where a concept is explained.
- Uses **LLMs with RAG** to prevent hallucinations  
- Returns **exact timestamps** from source videos  
- Preserves **context across fragmented transcripts**  
- Runs **locally using Ollama** (no paid APIs)  
- Designed with scalability and real-world constraints in mind  

---

##  End-to-End System Workflow

### 1. Video Collection & Normalization
- Downloaded all course videos from an online source.
- Resolved filename conflicts caused by identical names.
- Extracted playlist metadata from CSV.
- Filtered out non-tutorial videos.
- Renamed videos as:  
  **`<tutorial_number> - <lecture_title>`**

### 2️. Audio Extraction
- Extracted audio using **FFmpeg**.
- Audio names derived directly from video filenames for traceability.

### 3️. Speech-to-Text at Scale
- Transcribed audio using **Whisper** with timestamps.
- Distributed processing across multiple **Google Colab** instances.
- Generated structured JSON files with chunks, timestamps, titles, and full transcripts.

### 4. Context-Preserving Chunking
- Identified context loss due to small chunks.
- Implemented a **chunk-merging strategy** to preserve semantic continuity.

### 5️. Embedding Generation
- Used **Ollama** with **bge-m3** embedding model.
- Generated embeddings for transcript chunks and user queries.
- Stored embeddings and metadata in **Pandas DataFrames**.

### 6. Persistence
- Stored embeddings using **Joblib** for fast future retrieval.

### 7. Retrieval
- Applied **cosine similarity (scikit-learn)**.
- Retrieved **Top-K** relevant chunks per query.

### 8️. LLM-Powered Answer Generation
- Used **LLaMA 3.2** for reasoning and answer synthesis.
- Passed retrieved chunks and structured prompts.
- Generated answers with precise video references and timestamps.

---

## Tech Stack

- **LLMs**: LLaMA 3.2, Whisper  
- **Embeddings**: bge-m3 (Ollama)  
- **RAG Pipeline**: Explained below
- **Core Tools**: Python, Pandas, NumPy, Scikit-learn  
- **Media Processing**: FFmpeg  
- **Infrastructure**: Ollama (local), Google Colab  
- **Persistence**: Joblib  

---

## RAG Pipeline

<p><strong>User Query</strong></p>
<p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↓</p>
<p><strong>Query Embedding (bge-m3)</strong></p>
<p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↓</p>
<p><strong>Similarity Search (Cosine Similarity)</strong></p>
<p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↓</p>
<p><strong>Top-K Relevant Transcript Chunks</strong></p>
<p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↓</p>
<p><strong>Prompt Augmentation</strong></p>
<p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↓</p>
<p><strong>LLM (LLaMA 3.2)</strong></p>
<p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↓</p>
<p><strong>Generated Grounded Answer + Video Timestamps</strong></p>



---

## Future Enhancements

- Vector databases for large scale model
- Web-based UI for Respective Client
- Multi-disciplinary Course support
- Retrieval quality improvements using similarity score thresholds

---

## Why This Matters

This project demonstrates **practical LLM engineering** using RAG to ground answers in real data.  
It solves a real-world problem: **finding exact explanations inside hours of video content**.

---

## 👨‍💻 Author

**Satyam Gupta**

B.Tech, Electronics & Communication Engineering<br>
MANIT Bhopal

---

## ⭐ Show Your Support

If you like this project, consider giving it a ⭐ on GitHub!
