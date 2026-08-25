import os
import json
import requests

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from FlagEmbedding import FlagReranker


load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_KEY")

if not QDRANT_URL:
    raise ValueError("QDRANT_URL is missing from .env")

if not QDRANT_API_KEY:
    raise ValueError("QDRANT_KEY is missing from .env")


client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

COLLECTION_NAME = "sigma_course_chunks_v2"

EMBEDDING_URL = "http://localhost:11434/api/embed"
EMBEDDING_MODEL = "bge-m3"

LLM_URL = "http://localhost:11434/api/generate"
LLM_MODEL = "llama3.2:3b"

RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

CANDIDATE_COUNT = 20
FINAL_CONTEXT_COUNT = 5

reranker = FlagReranker(
    RERANKER_MODEL,
    use_fp16=False
)


def create_embedding(text):
    response = requests.post(
        EMBEDDING_URL,
        json={
            "model": EMBEDDING_MODEL,
            "input": [text]
        },
        timeout=120
    )

    response.raise_for_status()

    embeddings = response.json()["embeddings"]

    if not embeddings:
        raise RuntimeError("No embedding was returned by Ollama.")

    return embeddings[0]


def inference(prompt):
    response = requests.post(
        LLM_URL,
        json={
            "model": LLM_MODEL,
            "prompt": prompt,
            "stream": False
        },
        timeout=300
    )

    response.raise_for_status()

    return response.json()


def retrieve_chunks(question, top_results=CANDIDATE_COUNT):
    question_embedding = create_embedding(question)

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=question_embedding,
        limit=top_results,
        with_payload=True
    )

    retrieved_chunks = []

    for point in results.points:
        payload = point.payload or {}

        text = payload.get("text")

        if not text:
            continue

        retrieved_chunks.append(
            {
                "title": payload.get("title"),
                "number": payload.get("number"),
                "start": payload.get("start"),
                "end": payload.get("end"),
                "text": text,
                "qdrant_score": point.score
            }
        )

    return retrieved_chunks


def rerank_chunks(question, chunks, top_n=FINAL_CONTEXT_COUNT):
    if not chunks:
        return []

    pairs = [
        [question, chunk["text"]]
        for chunk in chunks
    ]

    scores = reranker.compute_score(
        pairs,
        normalize=True
    )

    if not isinstance(scores, list):
        scores = [scores]

    reranked_chunks = []

    for chunk, score in zip(chunks, scores):
        reranked_chunk = chunk.copy()
        reranked_chunk["rerank_score"] = float(score)
        reranked_chunks.append(reranked_chunk)

    reranked_chunks.sort(
        key=lambda item: item["rerank_score"],
        reverse=True
    )

    return reranked_chunks[:top_n]


incoming_query = input("Ask a Question: ")

retrieved_chunks = retrieve_chunks(
    incoming_query,
    top_results=CANDIDATE_COUNT
)

if not retrieved_chunks:
    print("No relevant course content was found.")
    raise SystemExit


reranked_chunks = rerank_chunks(
    incoming_query,
    retrieved_chunks,
    top_n=FINAL_CONTEXT_COUNT
)

context = json.dumps(
    reranked_chunks,
    ensure_ascii=False,
    indent=2
)


prompt = f"""
You are an AI assistant for the Sigma Web Development course.

Here is given video subtitle chunks from this course.
Each chunk contains:
video title, video number, start time (in seconds),
end time (in seconds), spoken text.

{context}

---------------------------------

"{incoming_query}"

User asked this question related to the video chunks.
You have to answer in a human way.

If the user asks an unrelated question, tell them
that you can only answer questions related to the course.

Answer only using the information present in the given
course content.

Explain where the topic is taught in the course and
guide the user to go to that particular video.

Clearly mention:
- the video title
- the video number
- the timestamp(s) where the topic is explained
  (as they are in seconds give timestamp in HH:MM:SS format)

Guide the user to watch the relevant video(s) in a
natural, human, teacher-like tone.

Do not mention subtitle chunks, JSON, datasets,
reranking, Qdrant, embeddings, or internal formats.

Do not assume or invent any information outside
the given data.

If the user asks an unrelated question, tell them
that you can only answer questions related to the course.
"""


with open("prompt.txt", "w", encoding="utf-8") as f:
    f.write(prompt)


response = inference(prompt)["response"]

print("\n" + response)


with open("response.txt", "w", encoding="utf-8") as f:
    f.write(response)