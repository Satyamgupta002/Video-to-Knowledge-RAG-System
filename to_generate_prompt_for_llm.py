import os
import json
import requests

from dotenv import load_dotenv
from qdrant_client import QdrantClient


# ============================================================
# 1. Load environment variables
# ============================================================

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_KEY")

if not QDRANT_URL:
    raise ValueError("QDRANT_URL is missing from .env")

if not QDRANT_API_KEY:
    raise ValueError("QDRANT_API_KEY is missing from .env")


# ============================================================
# 2. Connect to Qdrant Cloud
# ============================================================

client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

COLLECTION_NAME = "sigma_web_d_course_chunks"


# ============================================================
# 3. Generate BGE-M3 embedding for the user's query
# ============================================================

def create_embedding(text_list):
    r = requests.post(
        "http://localhost:11434/api/embed",
        json={
            "model": "bge-m3",
            "input": text_list
        },
        timeout=120
    )

    r.raise_for_status()

    embedding = r.json()["embeddings"]
    return embedding


# ============================================================
# 4. Generate answer using Llama 3.2 3B
# ============================================================

def inference(prompt):
    r = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3.2:3b",
            "prompt": prompt,
            "stream": False
        },
        timeout=300
    )

    r.raise_for_status()

    response = r.json()
    return response


# ============================================================
# 5. Retrieve top-K chunks from Qdrant Cloud
# ============================================================

def retrieve_chunks(question, top_results=10):

    question_embedding = create_embedding([question])[0]

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=question_embedding,
        limit=top_results,
        with_payload=True
    )

    retrieved_chunks = []

    for point in results.points:
        payload = point.payload or {}

        retrieved_chunks.append({
            "title": payload.get("title"),
            "number": payload.get("number"),
            "start": payload.get("start"),
            "end": payload.get("end"),
            "text": payload.get("text")
        })

    return retrieved_chunks


# ============================================================
# 6. Ask the user for a question
# ============================================================

incoming_query = input("Ask a Question: ")


# ============================================================
# 7. Retrieve relevant course chunks
# ============================================================

retrieved_chunks = retrieve_chunks(
    incoming_query,
    top_results=10
)

if not retrieved_chunks:
    print("No relevant course content was found.")
    exit()


# ============================================================
# 8. Build context for the LLM
# ============================================================

context = json.dumps(
    retrieved_chunks,
    ensure_ascii=False,
    indent=2
)


# ============================================================
# 9. Build prompt
# ============================================================

prompt = f'''
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
- the timestamp(s) where the topic is explained(as they are in seconds give timestamp in hour:minute:second format)

Guide the user to watch the relevant video(s) in a
natural, human, teacher-like tone.

Do not mention subtitle chunks, JSON, datasets,
or internal formats.

Do not assume or invent any information outside
the given data.

If the user asks an unrelated question, tell them
that you can only answer questions related to the course.
'''


# ============================================================
# 10. Save prompt for debugging
# ============================================================

with open("prompt.txt", "w", encoding="utf-8") as f:
    f.write(prompt)


# ============================================================
# 11. Generate final answer
# ============================================================

response = inference(prompt)["response"]

print("\n" + response)


# ============================================================
# 12. Save response for debugging
# ============================================================

with open("response.txt", "w", encoding="utf-8") as f:
    f.write(response)