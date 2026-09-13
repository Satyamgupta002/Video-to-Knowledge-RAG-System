# Video-to-Knowledge RAG System

An end-to-end **Retrieval-Augmented Generation (RAG)** system that transforms long-form course videos into searchable, timestamp-aware knowledge.

The system processes course videos, extracts and translates their spoken content, creates context-preserving transcript chunks, generates semantic embeddings, stores them in a vector database, retrieves relevant content for a user query, reranks the retrieved candidates using a cross-encoder, and finally generates a grounded answer using an LLM.

The system also preserves **video number, video title, and timestamps**, allowing users to locate where a particular concept is taught in the course.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [End-to-End Pipeline](#end-to-end-pipeline)
- [1. Video Ingestion](#1-video-ingestion)
- [2. Audio Extraction](#2-audio-extraction)
- [3. Speech-to-Text with Whisper](#3-speech-to-text-with-whisper)
- [4. Sentence-Aware Chunking](#4-sentence-aware-chunking)
- [5. Semantic Embedding Generation](#5-semantic-embedding-generation)
- [6. Vector Storage with Qdrant](#6-vector-storage-with-qdrant)
- [7. Query Processing](#7-query-processing)
- [8. Dense Semantic Retrieval](#8-dense-semantic-retrieval)
- [9. Cross-Encoder Reranking](#9-cross-encoder-reranking)
- [10. Grounded Response Generation](#10-grounded-response-generation)
- [Timestamp-Aware Source Retrieval](#timestamp-aware-source-retrieval)
- [Why Two-Stage Retrieval](#why-two-stage-retrieval)
- [Project Statistics](#project-statistics)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Environment Variables](#environment-variables)
- [Running the Project](#running-the-project)
- [Adding a New Video](#adding-a-new-video)
- [Example Query](#example-query)
- [Design Decisions](#design-decisions)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)

---

# Overview

Long-form educational videos contain a large amount of useful information, but finding a specific concept manually can be time-consuming.

This project converts course videos into a searchable knowledge base.

Instead of manually searching through hours of video, a user can ask a natural-language question such as:

> "Why do we need to use the Box Model?"

The system retrieves the most relevant portions of the course, reranks them according to query relevance, and generates an answer grounded in the retrieved course content.

The response also identifies:

- Video title
- Video number
- Relevant timestamp(s)
- Explanation based only on the retrieved course content

---

# Key Features

- **Long-form video processing**
- **Whisper-based speech transcription**
- **Hindi-to-English translation during transcription**
- **Sentence-aware context-preserving chunking**
- **1,024-dimensional BGE-M3 embeddings**
- **Dense semantic retrieval**
- **Qdrant Cloud vector database**
- **BGE-Reranker-v2-M3 cross-encoder reranking**
- **LLM-based grounded response generation**
- **Timestamp-aware source retrieval**
- **Video metadata preservation**
- **Incremental addition of new course videos**

---

# System Architecture

The system is divided into two major stages:

1. **Offline / ingestion pipeline**
2. **Online / query pipeline**

```text
                         ┌──────────────────────┐
                         │     Course Videos    │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │    FFmpeg Audio      │
                         │      Extraction      │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │       Whisper        │
                         │ Speech-to-Text +     │
                         │ Hindi → English      │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Sentence-Aware       │
                         │ Context-Preserving   │
                         │ Chunking             │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │       BGE-M3         │
                         │  Semantic Embeddings │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │      Qdrant Cloud    │
                         │   Vector Database    │
                         │ + Metadata + Timing  │
                         └──────────────────────┘


                         QUERY PIPELINE

                     ┌────────────────────┐
                     │    User Question   │
                     └─────────┬──────────┘
                               │
                               ▼
                     ┌────────────────────┐
                     │      BGE-M3        │
                     │ Query Embedding    │
                     └─────────┬──────────┘
                               │
                               ▼
                     ┌────────────────────┐
                     │   Qdrant Dense     │
                     │    Retrieval       │
                     │   Top-K Candidates │
                     └─────────┬──────────┘
                               │
                               ▼
                     ┌────────────────────┐
                     │ BGE-Reranker-v2-M3 │
                     │  Cross-Encoder     │
                     │     Reranking      │
                     └─────────┬──────────┘
                               │
                               ▼
                     ┌────────────────────┐
                     │    Top Relevant    │
                     │      Chunks        │
                     └─────────┬──────────┘
                               │
                               ▼
                     ┌────────────────────┐
                     │   Gemini 3.6 Flash │
                     │  Grounded Answer   │
                     │     Generation     │
                     └─────────┬──────────┘
                               │
                               ▼
           ┌────────────────────────────────────────────────┐
           │      Answer + Video reference + Timestamp      │
           └────────────────────────────────────────────────┘
```

---

# End-to-End Pipeline

The complete workflow can be summarized as:

```text
VIDEO
  │
  ▼
FFmpeg
  │
  ▼
Audio
  │
  ▼
Whisper
  │
  ▼
Timestamped Transcript
  │
  ▼
Sentence-Aware Chunking
  │
  ▼
Context-Preserving Chunks
  │
  ▼
BGE-M3
  │
  ▼
1,024-Dimensional Embeddings
  │
  ▼
Qdrant Cloud
  │
  │
  │            USER QUERY
  │                │
  │                ▼
  │             BGE-M3
  │                │
  │                ▼
  └──────────► Dense Retrieval
                   │
                   ▼
              Top-K Candidates
                   │
                   ▼
          BGE-Reranker-v2-M3
                   │
                   ▼
               Top-5 Chunks
                   │
                   ▼
           Gemini 3.6 Flash
                   │
                   ▼
          Grounded Response
                   │
                   ▼
       Video + Timestamp + Answer
```

---

# 1. Video Ingestion

The system accepts course videos as input.

Each video is associated with metadata such as:

- Video number
- Video title
- Video file path

This metadata is preserved throughout the processing pipeline.

The metadata is later stored along with each transcript chunk so that retrieved content can be traced back to its original lecture.

---

# 2. Audio Extraction

The first processing stage extracts audio from the input video using **FFmpeg**.

```text
Course Video
     │
     ▼
   FFmpeg
     │
     ▼
   MP3 Audio
```

The video stream itself is not directly used for semantic retrieval.

Instead, the audio is extracted because the primary knowledge source is the instructor's spoken content.

The audio is temporarily generated during processing and passed to Whisper.

---

# 3. Speech-to-Text with Whisper

The extracted audio is processed using:

```text
Whisper large-v2
```

The transcription configuration uses:

```text
Language: Hindi
Task: translate
```

This allows the system to process Hindi course lectures while producing English text for the downstream retrieval and generation pipeline.

Whisper also provides segment-level timestamps.

Conceptually:

```json
{
  "start": 125.4,
  "end": 132.7,
  "text": "..."
}
```

These timestamps are important because the final answer can point the user back to the relevant portion of the lecture.

---

# 4. Sentence-Aware Chunking

Raw Whisper output is composed of many small transcript segments.

Using each Whisper segment directly as a retrieval chunk can produce fragmented and semantically incomplete pieces of text.

To address this, the system performs **sentence-aware context-preserving chunking**.

### Chunking configuration

```text
Target chunk size     ≈ 100 words
Maximum chunk size    = 150 words
Sentence overlap      = 1 sentence
```

The chunking process is:

```text
Whisper Segments
      │
      ▼
Sentence Tokenization
      │
      ▼
Long Sentence Handling
      │
      ▼
Accumulate Sentences
      │
      ├── Target ≈ 100 words
      │
      ├── Hard limit = 150 words
      │
      └── 1-sentence overlap
      │
      ▼
Final Retrieval Chunks
```

NLTK is used for sentence tokenization.

Long sentences that exceed the maximum chunk size are split into smaller pieces so that the hard word limit is maintained.

### Why sentence-aware chunking?

A fixed-size chunking strategy can split content in the middle of an idea.

For example:

```text
Chunk 1:
"HTTP is a protocol that allows clients and servers
to communicate. The client sends a request..."
```

Instead of splitting strictly after a fixed number of words, the system attempts to preserve complete sentence boundaries.

The one-sentence overlap also helps preserve context between neighboring chunks.

For example:

```text
Chunk 1:
A B C D

Chunk 2:
D E F G
```

Here, sentence `D` provides contextual continuity between the chunks.

---

# 5. Semantic Embedding Generation

Each final transcript chunk is converted into a numerical vector using:

```text
BGE-M3
```

The embedding dimension is:

```text
1,024 dimensions
```

Conceptually:

```text
Transcript Chunk
       │
       ▼
     BGE-M3
       │
       ▼
[0.12, -0.04, 0.87, ...]
       │
       ▼
1,024-dimensional vector
```

The purpose of embeddings is to represent the semantic meaning of the text in a vector space.

This allows semantically similar queries and chunks to be located near each other even when they do not use exactly the same words.

For example:

```text
Query:
"How does a browser communicate with a server?"

Possible retrieved chunk:
"The client sends an HTTP request to the web server..."
```

The wording is different, but the underlying meaning is related.

---

# 6. Vector Storage with Qdrant

The generated BGE-M3 embeddings are stored in **Qdrant Cloud**.

Qdrant stores both the vector and associated metadata.

A conceptual point looks like:

```json
{
  "id": "unique-point-id",

  "vector": [
    0.12,
    -0.04,
    0.87
  ],

  "payload": {
    "number": "12",
    "title": "Introduction to CSS",
    "start": 125.4,
    "end": 137.8,
    "text": "Transcript chunk...",
    "chunk_id": "12_5"
  }
}
```

The actual embedding contains 1,024 values.

### Why store metadata?

The vector is responsible for retrieval, while metadata provides source information.

This separation allows the system to retrieve a chunk semantically while still knowing:

- Which video it came from
- Which lecture it belongs to
- Where it occurs in the video
- What the original transcript says

---

# 7. Query Processing

When a user asks a question, the query follows a separate retrieval pipeline.

For example:

```text
"Why do we need to use the Box Model?"
```

The query is first converted into an embedding using the same BGE-M3 model used during document embedding.

```text
User Query
    │
    ▼
BGE-M3
    │
    ▼
Query Embedding
    │
    ▼
Qdrant
```

Using the same embedding model for both documents and queries places them into the same vector space.

---

# 8. Dense Semantic Retrieval

Qdrant performs dense vector similarity search using the query embedding.

The current retrieval pipeline first obtains a candidate set from Qdrant.

For example:

```text
User Query
    │
    ▼
BGE-M3 Query Embedding
    │
    ▼
Qdrant Dense Search
    │
    ▼
Top-20 Candidate Chunks
```

The retrieved candidates include both their content and stored metadata.

Conceptually:

```text
Rank 1 → Chunk A
Rank 2 → Chunk B
Rank 3 → Chunk C
...
Rank 20 → Chunk T
```

These candidates are not immediately sent to the LLM.

Instead, they go through a second-stage relevance model.

---

# 9. Cross-Encoder Reranking

The system uses:

```text
BGE-Reranker-v2-M3
```

for second-stage reranking.

Unlike the embedding model, the reranker directly evaluates the relationship between:

```text
Query + Candidate Chunk
```

The process is:

```text
User Query
     │
     ├─────────────────────────┐
     │                         │
     ▼                         ▼
Candidate 1              Candidate 2
     │                         │
     └──────────┬──────────────┘
                │
                ▼
       BGE-Reranker-v2-M3
                │
                ▼
        Relevance Scores
                │
                ▼
          Sorted Results
                │
                ▼
            Top 5 Chunks
```

For each candidate:

```text
(query, passage)
```

is scored by the cross-encoder.

The candidates are then sorted according to their reranker scores.

### Why rerank?

Dense retrieval is efficient and suitable for searching the entire collection, but the initial ranking may not always place the most relevant passages at the top.

A cross-encoder can perform a more direct query-passage relevance assessment.

Therefore, the system uses:

```text
Fast retrieval
      +
More precise reranking
```

instead of applying the expensive reranker to every chunk in the database.

---

# 10. Grounded Response Generation

After reranking, only the most relevant chunks are passed to the LLM.

Current configuration:

```text
Qdrant retrieval → Top 20
        ↓
BGE Reranker     → Top 5
        ↓
LLM              → Grounded response
```

The retrieved chunks are formatted into the prompt along with their metadata.

The prompt instructs the model to:

- Answer using only the provided course content
- Explain where the topic is taught
- Mention the video title
- Mention the video number
- Mention relevant timestamps
- Avoid inventing information
- Guide the user toward the relevant lecture

This makes the LLM a **generation layer over retrieved course knowledge**, rather than the primary knowledge source.

---

# Timestamp-Aware Source Retrieval

One of the important features of this project is preservation of timestamps throughout the pipeline.

The flow is:

```text
Video
  │
  ▼
Whisper
  │
  ├── Transcript
  │
  └── Timestamps
          │
          ▼
       Chunking
          │
          ▼
       Qdrant
          │
          ▼
    Retrieved Chunk
          │
          ▼
Video + Timestamp
          │
          ▼
       LLM Answer
```

Each chunk retains:

```text
Video Number
Video Title
Start Time
End Time
Transcript Text
```

This allows the final answer to point users toward the relevant lecture section.

For example:

```text
The Box Model is explained in:

Video: CSS Box Model
Timestamp: 00:14:32 – 00:17:10
```

---

# Why Two-Stage Retrieval?

A single retrieval stage can be insufficient for high-quality RAG.

The system therefore separates retrieval into two stages.

## Stage 1 — Candidate Retrieval

Qdrant performs fast dense semantic search.

```text
All Chunks
   │
   ▼
Dense Vector Search
   │
   ▼
Top 20
```

The purpose is **high recall** — retrieve a sufficiently broad set of potentially relevant chunks.

## Stage 2 — Reranking

The cross-encoder then examines those candidates more carefully.

```text
Top 20 Candidates
       │
       ▼
BGE-Reranker-v2-M3
       │
       ▼
Relevance Ranking
       │
       ▼
Top 5
```

The purpose is to improve **context precision** before the retrieved information is passed to the LLM.

### Why not rerank all chunks?

Suppose the database contains hundreds or thousands of chunks.

A cross-encoder evaluates query-document pairs directly and is more computationally expensive than vector retrieval.

Therefore:

```text
Entire Collection
       │
       ▼
Fast Dense Retrieval
       │
       ▼
Small Candidate Set
       │
       ▼
Expensive Reranking
```

This provides a practical trade-off between retrieval coverage and computational cost.

---

# Project Statistics

| Metric | Value |
|---|---:|
| Course video duration | 33+ hours |
| Original transcript segments | 17K+ |
| Final chunks | 825 |
| Reduction | ~95.4% |
| Target chunk size | ~100 words |
| Maximum chunk size | 150 words |
| Chunk overlap | 1 sentence |
| Embedding model | BGE-M3 |
| Embedding dimension | 1,024 |
| Initial retrieval | Top 20 |
| Reranked context | Top 5 |

The reduction from 17K+ transcript segments to 825 final chunks significantly reduces the number of searchable units while maintaining larger, context-preserving pieces of transcript content.

---

# Technology Stack

### Programming

- Python

### Speech Processing

- Whisper large-v2
- FFmpeg

### NLP / Chunking

- NLTK

### Embeddings

- BGE-M3
- Ollama

### Vector Database

- Qdrant Cloud

### Reranking

- BGE-Reranker-v2-M3
- FlagEmbedding

### Generation

- Google Gemini 3.6 Flash

---

# Project Structure

A conceptual project structure is:

```text
AI-Course-Assistant-Based-on-RAG/
│
├── videos/
│   └── course videos
│
├── sentence_jsons/
│   └── processed transcript chunks
│
├── add_video_to_qdrant.py
│
├── embed_and_upload_to_qdrant.py
│
├── to_generate_prompt_for_llm.py
│
├── prompt.txt
│
├── response.txt
│
├── .env
│
└── README.md
```

> File names may differ depending on the current version of the project.

---

# Installation

## 1. Clone the Repository

```bash
git clone https://github.com/Satyamgupta002/AI-Course-Assistant-Based-on-RAG.git
cd AI-Course-Assistant-Based-on-RAG
```

---

## 2. Create a Python Environment

```bash
python -m venv venv
```

### Windows

```powershell
venv\Scripts\activate
```

---

## 3. Install Dependencies

Install the required Python libraries:

```bash
pip install openai-whisper
pip install qdrant-client
pip install FlagEmbedding
pip install nltk
pip install requests
pip install python-dotenv
```

Additional dependencies may be required depending on the local environment.

---

# Environment Variables

Create a `.env` file in the project directory.

```env
QDRANT_URL=your_qdrant_cloud_cluster_url
QDRANT_KEY=your_qdrant_database_api_key
```

If Gemini API access is used directly from the application, also configure the Gemini API key according to the implementation.

> Never commit API keys or other secrets to GitHub.

A `.gitignore` file should contain entries such as:

```gitignore
.env
venv/
__pycache__/
*.pyc
```

---

# Running the Project

The project can be considered as two workflows:

```text
                 ┌─────────────────────┐
                 │   Ingestion Phase   │
                 └──────────┬──────────┘
                            │
                            ▼
                   Process Course Videos
                            │
                            ▼
                   Create Embeddings
                            │
                            ▼
                       Qdrant
                            │
                            │
                            ▼
                 ┌─────────────────────┐
                 │    Query Phase      │
                 └──────────┬──────────┘
                            │
                            ▼
                     Ask a Question
                            │
                            ▼
                    Retrieve + Rerank
                            │
                            ▼
                    Generate Response
```

---

# Adding a New Video

A new course video can be added without rebuilding the entire retrieval pipeline.

The video processing workflow is:

```text
New Video
   │
   ▼
FFmpeg
   │
   ▼
Whisper
   │
   ▼
Sentence-Aware Chunking
   │
   ▼
BGE-M3 Embeddings
   │
   ▼
Qdrant Upsert
```

The ingestion script accepts:

```text
Video Path
Video Number
Video Title
```

For example:

```text
Video Path   → E:\Course\lecture36.mp4
Video Number → 36
Video Title  → CSS Box Model
```

The resulting chunks are embedded and uploaded to the existing Qdrant collection.

The current ingestion implementation uses deterministic UUID5-based point IDs derived from the video number, title, and chunk index. This allows the same video/chunk identity to be reproduced when the same input is processed again.

---

# Example Query

### User Query

```text
Why do we need to use Box Model?
```

### Retrieval Pipeline

```text
Question
   │
   ▼
BGE-M3 Query Embedding
   │
   ▼
Qdrant Dense Search
   │
   ▼
Top 20 Candidates
   │
   ▼
BGE-Reranker-v2-M3
   │
   ▼
Top 5 Relevant Chunks
   │
   ▼
Gemini 3.6 Flash
   │
   ▼
Grounded Answer
```

### Expected Response Structure

```text
The CSS Box Model is used to understand how the dimensions
of an element are calculated, including its content, padding,
border, and margin.

This topic is explained in:

Video: <Video Title>
Video Number: <Video Number>
Timestamp: <HH:MM:SS>
```

The exact answer and timestamp depend on the retrieved course content.

---

# Design Decisions

## Why RAG?

RAG allows the system to retrieve information from the course dynamically instead of requiring the entire course knowledge to be encoded into model parameters.

This is particularly useful because new videos can be added to the knowledge base without retraining the language model.

---

## Why BGE-M3?

BGE-M3 is used to convert both transcript chunks and user queries into dense semantic representations.

This enables semantic retrieval rather than relying only on exact keyword matching.

---

## Why Qdrant?

The project initially used local embedding persistence and similarity-based retrieval.

Qdrant provides a dedicated vector database for:

- Vector similarity search
- Persistent storage
- Metadata payloads
- Scalable retrieval
- Cloud-hosted deployment

The current implementation uses Qdrant Cloud.

---

## Why Sentence-Aware Chunking?

Transcript segments produced by speech recognition can be too fragmented for effective retrieval.

Sentence-aware chunking allows multiple related sentences to be grouped into a coherent retrieval unit.

The system also uses one-sentence overlap to preserve context between neighboring chunks.

---

## Why a 100–150 Word Chunk Size?

Very small chunks can lose contextual information.

Very large chunks can dilute the semantic focus of an embedding.

The project therefore targets approximately 100 words while enforcing a 150-word maximum.

This provides a balance between:

```text
Context Preservation
        +
Semantic Focus
        +
Retrieval Efficiency
```

---

## Why Reranking?

Dense retrieval is useful for efficiently finding candidate passages, but the initial ranking may not always be optimal.

BGE-Reranker-v2-M3 performs a more direct query-passage relevance assessment on the candidate set.

Therefore:

```text
BGE-M3
→ Efficient candidate retrieval

BGE-Reranker-v2-M3
→ More focused relevance ranking
```

---

## Why Metadata Is Stored Separately?

The system stores video information and timestamps as Qdrant payload metadata.

This separates:

```text
Vector
  ↓
Semantic Retrieval
```

from:

```text
Metadata
  ↓
Source Identification
```

The metadata can then be returned with the retrieved chunk and used when generating the final response.
---

# Limitations

The current implementation has several limitations.

### 1. Audio-Centric Retrieval

Although the input is video, the current knowledge representation is derived primarily from spoken audio.

Visual information present only on slides or in demonstrations is not directly represented in the retrieval index.

---

### 2. No Explicit Retrieval Evaluation Dataset

The retrieval pipeline can be improved further by creating a benchmark.
---

### 3. Dense Retrieval Only

The current first-stage retrieval is dense semantic retrieval.

Exact keyword matching is not explicitly incorporated.
---

# Future Improvements

## 1. Hybrid Search

Combine:

```text
Dense Retrieval
       +
BM25 / Sparse Retrieval
```

to improve retrieval for both semantic queries and exact technical terminology.
Reciprocal Rank Fusion (RRF) can be used to combine the ranked results from dense and sparse retrieval.

---

## 2. Retrieval Evaluation

Create a manually labelled evaluation dataset and measure:

- Recall@K
- Precision@K
- Faithfullness
- Answer Correctness
- Answer Relevance
- Context Precision
- Context Recall
---
