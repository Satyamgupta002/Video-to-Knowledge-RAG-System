import argparse
import os
import subprocess
import tempfile
import uuid
from pathlib import Path

import requests
import whisper
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_KEY")

COLLECTION_NAME = "sigma_web_d_course_chunks"

WHISPER_MODEL = "large-v2"
WHISPER_LANGUAGE = "hi"
WHISPER_TASK = "translate"

SEGMENTS_PER_CHUNK = 5

EMBEDDING_BATCH_SIZE = 32

QDRANT_BATCH_SIZE = 100

OLLAMA_EMBED_URL = "http://localhost:11434/api/embed"
EMBEDDING_MODEL = "bge-m3"

def validate_environment(video_path: str) -> None:
    if not QDRANT_URL:
        raise ValueError("QDRANT_URL is missing from .env")

    if not QDRANT_API_KEY:
        raise ValueError("QDRANT_API_KEY is missing from .env")

    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        raise RuntimeError(
            "FFmpeg was not found. Install FFmpeg and make sure "
            "the ffmpeg command is available in PATH."
        )

def extract_audio(video_path: str, output_mp3: str) -> None:

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            video_path,
            "-vn",
            "-acodec",
            "libmp3lame",
            "-q:a",
            "2",
            output_mp3,
        ],
        check=True,
    )

def transcribe_audio(audio_path: str, model) -> dict:

    result = model.transcribe(
        audio=audio_path,
        language=WHISPER_LANGUAGE,
        task=WHISPER_TASK,
        word_timestamps=False,
    )
    return result

def create_chunks(result: dict, video_number: str, video_title: str) -> list:

    segments = result.get("segments", [])

    if not segments:
        raise ValueError("Whisper returned no transcript segments.")

    chunks = []

    for i in range(0, len(segments), SEGMENTS_PER_CHUNK):
        group = segments[i:i + SEGMENTS_PER_CHUNK]

        chunk_index = (i // SEGMENTS_PER_CHUNK) + 1

        chunk_id = f"{video_number}_{chunk_index}"

        point_id = str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"sigma-course/{video_number}/{video_title}/{chunk_index}"
            )
        )

        chunks.append(
            {
                "point_id": point_id,
                "chunk_id": chunk_id,
                "number": str(video_number),
                "title": video_title,
                "start": group[0]["start"],
                "end": group[-1]["end"],
                "text": " ".join(
                    segment["text"].strip()
                    for segment in group
                    if segment.get("text")
                ),
            }
        )
    return chunks

def create_embeddings(texts: list[str]) -> list[list[float]]:

    all_embeddings = []

    for start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batch = texts[start:start + EMBEDDING_BATCH_SIZE]

        response = requests.post(
            OLLAMA_EMBED_URL,
            json={
                "model": EMBEDDING_MODEL,
                "input": batch,
            },
            timeout=600,
        )

        response.raise_for_status()

        data = response.json()
        embeddings = data.get("embeddings")

        if not embeddings:
            raise RuntimeError(
                "Ollama did not return embeddings. "
                "Make sure Ollama is running and bge-m3 is installed."
            )

        if len(embeddings) != len(batch):
            raise RuntimeError(
                f"Ollama returned {len(embeddings)} embeddings "
                f"for {len(batch)} texts."
            )

        all_embeddings.extend(embeddings)

        print(
            f"Embedded "
            f"{min(start + len(batch), len(texts))}/{len(texts)}"
        )

    if not all_embeddings:
        raise RuntimeError("No embeddings were generated.")

    dimension = len(all_embeddings[0])

    return all_embeddings

def upload_to_qdrant(chunks: list, embeddings: list[list[float]]) -> None:

    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
    )

    if not client.collection_exists(COLLECTION_NAME):
        raise RuntimeError(
            f"Qdrant collection '{COLLECTION_NAME}' does not exist.\n"
            "Run your migrate_to_qdrant.py first to create the collection "
            "and upload the original course embeddings."
        )

    collection_info = client.get_collection(COLLECTION_NAME)

    existing_dimension = collection_info.config.params.vectors.size
    new_dimension = len(embeddings[0])

    if existing_dimension != new_dimension:
        raise ValueError(
            f"Embedding dimension mismatch: Qdrant collection uses "
            f"{existing_dimension}, but BGE-M3 returned {new_dimension}."
        )

    for start in range(0, len(chunks), QDRANT_BATCH_SIZE):
        chunk_batch = chunks[start:start + QDRANT_BATCH_SIZE]
        embedding_batch = embeddings[start:start + QDRANT_BATCH_SIZE]

        points = []

        for chunk, embedding in zip(chunk_batch, embedding_batch):
            points.append(
                PointStruct(
                    id=chunk["point_id"],
                    vector=embedding,
                    payload={
                        "number": chunk["number"],
                        "title": chunk["title"],
                        "start": chunk["start"],
                        "end": chunk["end"],
                        "text": chunk["text"],
                        "chunk_id": chunk["chunk_id"],
                    },
                )
            )

        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
        )

        uploaded = min(
            start + len(chunk_batch),
            len(chunks),
        )

def add_video(video_path: str, video_number: str, video_title: str) -> None:
    validate_environment(video_path)
    whisper_model = whisper.load_model(WHISPER_MODEL)

    with tempfile.TemporaryDirectory(prefix="sigma_rag_") as temp_dir:
        audio_path = os.path.join(temp_dir, "audio.mp3")

        extract_audio(video_path, audio_path)

        result = transcribe_audio(audio_path, whisper_model)

    chunks = create_chunks(
        result=result,
        video_number=video_number,
        video_title=video_title,
    )

    texts = [chunk["text"] for chunk in chunks]
    embeddings = create_embeddings(texts)

    upload_to_qdrant(
        chunks=chunks,
        embeddings=embeddings,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Add a new course video to Qdrant Cloud."
    )

    parser.add_argument(
        "video_path",
        help="Path to the video file"
    )

    parser.add_argument(
        "video_number",
        help="Course video number, e.g. 36"
    )

    parser.add_argument(
        "video_title",
        help="Course video title"
    )

    args = parser.parse_args()

    add_video(
        video_path=args.video_path,
        video_number=args.video_number,
        video_title=args.video_title,
    )