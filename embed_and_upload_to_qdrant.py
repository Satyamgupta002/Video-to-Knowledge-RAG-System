import os
import json
import uuid
import requests

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct
)


load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_KEY")

COLLECTION_NAME = "sigma_course_chunks_v2"

JSON_FOLDER = r"E:\RAG Based AI Course Assistant\sentence_jsons"

OLLAMA_EMBED_URL = "http://localhost:11434/api/embed"
EMBEDDING_MODEL = "bge-m3"

EMBEDDING_BATCH_SIZE = 32
QDRANT_BATCH_SIZE = 100


def create_embedding(text_list):

    response = requests.post(
        OLLAMA_EMBED_URL,
        json={
            "model": EMBEDDING_MODEL,
            "input": text_list
        },
        timeout=600
    )

    response.raise_for_status()

    embeddings = response.json()["embeddings"]

    if len(embeddings) != len(text_list):
        raise RuntimeError(
            f"Ollama returned {len(embeddings)} embeddings "
            f"for {len(text_list)} texts."
        )

    return embeddings


def create_collection(client, vector_size):

    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=vector_size,
            distance=Distance.COSINE
        )
    )


def main():

    if not QDRANT_URL:
        raise ValueError("QDRANT_URL is missing from .env")

    if not QDRANT_API_KEY:
        raise ValueError("QDRANT_API_KEY is missing from .env")

    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY
    )

    all_chunks = []

    json_files = sorted(
        file for file in os.listdir(JSON_FOLDER)
        if file.endswith(".json")
    )

    if not json_files:
        raise ValueError("No JSON files found.")

    for json_file in json_files:

        file_path = os.path.join(
            JSON_FOLDER,
            json_file
        )

        with open(
            file_path,
            "r",
            encoding="utf-8"
        ) as f:

            data = json.load(f)

        chunks = data.get("chunks", [])

        for index, chunk in enumerate(chunks):

            chunk_copy = {
                "number": chunk["number"],
                "title": chunk["title"],
                "start": chunk["start"],
                "end": chunk["end"],
                "text": chunk["text"],
                "chunk_id": f"{chunk['number']}_{index}",
                "source_file": json_file
            }

            all_chunks.append(chunk_copy)

    if not all_chunks:
        raise ValueError("No chunks found in the JSON files.")

    for start in range(
        0,
        len(all_chunks),
        EMBEDDING_BATCH_SIZE
    ):

        batch = all_chunks[
            start:start + EMBEDDING_BATCH_SIZE
        ]

        texts = [
            chunk["text"]
            for chunk in batch
        ]

        embeddings = create_embedding(texts)

        for chunk, embedding in zip(
            batch,
            embeddings
        ):

            chunk["embedding"] = embedding

    vector_size = len(
        all_chunks[0]["embedding"]
    )

    create_collection(
        client,
        vector_size
    )

    for start in range(
        0,
        len(all_chunks),
        QDRANT_BATCH_SIZE
    ):

        batch = all_chunks[
            start:start + QDRANT_BATCH_SIZE
        ]

        points = []

        for chunk in batch:

            point_id = str(
                uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    (
                        f"sigma-course/"
                        f"{chunk['number']}/"
                        f"{chunk['title']}/"
                        f"{chunk['chunk_id']}"
                    )
                )
            )

            points.append(
                PointStruct(
                    id=point_id,
                    vector=chunk["embedding"],
                    payload={
                        "number": chunk["number"],
                        "title": chunk["title"],
                        "start": chunk["start"],
                        "end": chunk["end"],
                        "text": chunk["text"],
                        "chunk_id": chunk["chunk_id"],
                        "source_file": chunk["source_file"]
                    }
                )
            )

        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points
        )

    collection_info = client.get_collection(
        COLLECTION_NAME
    )

    print("Migration completed.")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Total chunks: {len(all_chunks)}")
    print(
        f"Vector dimension: "
        f"{collection_info.config.params.vectors.size}"
    )


if __name__ == "__main__":
    main()