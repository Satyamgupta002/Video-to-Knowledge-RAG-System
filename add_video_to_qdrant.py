import os
import re
import subprocess
import tempfile
import uuid

import requests
import whisper
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct


load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_KEY")

COLLECTION_NAME = "sigma_course_chunks_v2"

WHISPER_MODEL = "large-v2"
WHISPER_LANGUAGE = "hi"
WHISPER_TASK = "translate"

TARGET_WORDS = 100
MAX_WORDS = 150
OVERLAP_SENTENCES = 1

EMBEDDING_BATCH_SIZE = 32
QDRANT_BATCH_SIZE = 100

OLLAMA_EMBED_URL = "http://localhost:11434/api/embed"
EMBEDDING_MODEL = "bge-m3"


def validate_environment(video_path):
    if not QDRANT_URL:
        raise ValueError("QDRANT_URL is missing from .env")

    if not QDRANT_API_KEY:
        raise ValueError("QDRANT_KEY is missing from .env")

    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        raise RuntimeError(
            "FFmpeg was not found. Install FFmpeg and make sure "
            "the ffmpeg command is available in PATH."
        )


def extract_audio(video_path, output_mp3):
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
            output_mp3
        ],
        check=True
    )


def transcribe_audio(audio_path, model):
    return model.transcribe(
        audio=audio_path,
        language=WHISPER_LANGUAGE,
        task=WHISPER_TASK,
        word_timestamps=False
    )


def split_into_sentences(text):
    sentences = re.split(
        r"(?<=[.!?])\s+",
        text.strip()
    )

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


def build_sentence_units(segments, video_number, video_title):
    sentence_units = []

    for segment in segments:
        text = segment.get("text", "").strip()

        if not text:
            continue

        sentences = split_into_sentences(text)

        for sentence in sentences:
            sentence_units.append(
                {
                    "number": str(video_number),
                    "title": video_title,
                    "start": float(segment["start"]),
                    "end": float(segment["end"]),
                    "text": sentence
                }
            )

    return sentence_units


def split_long_sentence(sentence, max_words):
    words = sentence["text"].split()

    if len(words) <= max_words:
        return [sentence]

    pieces = []

    for start in range(0, len(words), max_words):
        pieces.append(
            {
                "number": sentence["number"],
                "title": sentence["title"],
                "start": sentence["start"],
                "end": sentence["end"],
                "text": " ".join(words[start:start + max_words])
            }
        )

    return pieces


def create_chunks(result, video_number, video_title):
    segments = result.get("segments", [])

    if not segments:
        raise ValueError("Whisper returned no transcript segments.")

    sentence_units = build_sentence_units(
        segments,
        video_number,
        video_title
    )

    expanded_units = []

    for sentence in sentence_units:
        expanded_units.extend(
            split_long_sentence(sentence, MAX_WORDS)
        )

    chunks = []
    current = []
    current_words = 0
    i = 0

    while i < len(expanded_units):
        unit = expanded_units[i]
        unit_words = len(unit["text"].split())

        if current and current_words + unit_words > MAX_WORDS:
            chunks.append(
                {
                    "number": current[0]["number"],
                    "title": current[0]["title"],
                    "start": current[0]["start"],
                    "end": current[-1]["end"],
                    "text": " ".join(item["text"] for item in current)
                }
            )

            overlap = (
                current[-OVERLAP_SENTENCES:]
                if OVERLAP_SENTENCES > 0
                else []
            )

            current = overlap.copy()
            current_words = sum(
                len(item["text"].split())
                for item in current
            )

            continue

        current.append(unit)
        current_words += unit_words

        if current_words >= TARGET_WORDS:
            chunks.append(
                {
                    "number": current[0]["number"],
                    "title": current[0]["title"],
                    "start": current[0]["start"],
                    "end": current[-1]["end"],
                    "text": " ".join(item["text"] for item in current)
                }
            )

            overlap = (
                current[-OVERLAP_SENTENCES:]
                if OVERLAP_SENTENCES > 0
                else []
            )

            current = overlap.copy()
            current_words = sum(
                len(item["text"].split())
                for item in current
            )

        i += 1

    if current:
        if not chunks or len(current) > OVERLAP_SENTENCES:
            chunks.append(
                {
                    "number": current[0]["number"],
                    "title": current[0]["title"],
                    "start": current[0]["start"],
                    "end": current[-1]["end"],
                    "text": " ".join(item["text"] for item in current)
                }
            )

    final_chunks = []

    for index, chunk in enumerate(chunks, start=1):
        point_id = str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"sigma-course/{video_number}/{video_title}/{index}"
            )
        )

        final_chunks.append(
            {
                "point_id": point_id,
                "chunk_id": f"{video_number}_{index}",
                "number": str(video_number),
                "title": video_title,
                "start": chunk["start"],
                "end": chunk["end"],
                "text": chunk["text"]
            }
        )

    return final_chunks


def create_embeddings(texts):
    all_embeddings = []

    for start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batch = texts[start:start + EMBEDDING_BATCH_SIZE]

        response = requests.post(
            OLLAMA_EMBED_URL,
            json={
                "model": EMBEDDING_MODEL,
                "input": batch
            },
            timeout=600
        )

        response.raise_for_status()

        embeddings = response.json().get("embeddings")

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

    return all_embeddings


def upload_to_qdrant(chunks, embeddings):
    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY
    )

    if not client.collection_exists(COLLECTION_NAME):
        raise RuntimeError(
            f"Qdrant collection '{COLLECTION_NAME}' does not exist."
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
                        "chunk_id": chunk["chunk_id"]
                    }
                )
            )

        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points
        )


def add_video(video_path, video_number, video_title):
    validate_environment(video_path)

    whisper_model = whisper.load_model(WHISPER_MODEL)

    with tempfile.TemporaryDirectory(prefix="sigma_rag_") as temp_dir:
        audio_path = os.path.join(temp_dir, "audio.mp3")

        extract_audio(video_path, audio_path)
        result = transcribe_audio(audio_path, whisper_model)

    chunks = create_chunks(
        result=result,
        video_number=video_number,
        video_title=video_title
    )

    texts = [chunk["text"] for chunk in chunks]
    embeddings = create_embeddings(texts)

    upload_to_qdrant(
        chunks=chunks,
        embeddings=embeddings
    )

    print(
        f"Added video {video_number}: "
        f"{video_title} ({len(chunks)} chunks)"
    )


if __name__ == "__main__":
    video_path = input("Enter video file path: ").strip().strip('"')
    video_number = input("Enter video number: ").strip()
    video_title = input("Enter video title: ").strip()

    add_video(
        video_path=video_path,
        video_number=video_number,
        video_title=video_title
    )
