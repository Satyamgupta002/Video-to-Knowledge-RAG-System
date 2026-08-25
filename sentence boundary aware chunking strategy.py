import os
import json
import nltk
from nltk.tokenize import sent_tokenize

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)


def build_sentence_units(segments):
    if not segments:
        return []

    clean_segments = []
    parts = []
    cursor = 0

    for seg in segments:
        text = seg.get("text", "").strip()
        if not text:
            continue

        if parts:
            parts.append(" ")
            cursor += 1

        start_char = cursor
        parts.append(text)
        cursor += len(text)

        clean_segments.append({
            "start_char": start_char,
            "end_char": cursor,
            "start": seg["start"],
            "end": seg["end"],
            "number": seg.get("number", 0),
            "title": seg.get("title", ""),
            "text": text
        })

    full_text = "".join(parts)
    sentence_units = []
    search_start = 0

    for sentence_text in sent_tokenize(full_text):
        start_char = full_text.find(sentence_text, search_start)
        if start_char == -1:
            continue

        end_char = start_char + len(sentence_text)
        search_start = end_char

        overlapping = [
            seg for seg in clean_segments
            if seg["end_char"] > start_char
            and seg["start_char"] < end_char
        ]

        if not overlapping:
            continue

        sentence_units.append({
            "text": sentence_text.strip(),
            "start": overlapping[0]["start"],
            "end": overlapping[-1]["end"],
            "number": overlapping[0]["number"],
            "title": overlapping[0]["title"],
            "segments": overlapping
        })

    return sentence_units


def word_count(item):
    return len(item["text"].split())


def make_chunk(items):
    return {
        "number": items[0]["number"],
        "title": items[0]["title"],
        "start": items[0]["start"],
        "end": items[-1]["end"],
        "text": " ".join(item["text"] for item in items).strip()
    }


def split_long_sentence(sentence, max_words):
    if word_count(sentence) <= max_words:
        return [sentence]

    pieces = []
    current = []
    current_words = 0

    for seg in sentence["segments"]:
        seg_words = len(seg["text"].split())

        if current and current_words + seg_words > max_words:
            pieces.append({
                "text": " ".join(item["text"] for item in current).strip(),
                "start": current[0]["start"],
                "end": current[-1]["end"],
                "number": sentence["number"],
                "title": sentence["title"],
                "segments": current
            })
            current = []
            current_words = 0

        if seg_words > max_words:
            words = seg["text"].split()
            for i in range(0, len(words), max_words):
                word_group = words[i:i + max_words]
                pieces.append({
                    "text": " ".join(word_group),
                    "start": seg["start"],
                    "end": seg["end"],
                    "number": sentence["number"],
                    "title": sentence["title"],
                    "segments": [seg]
                })
            continue

        current.append(seg)
        current_words += seg_words

    if current:
        pieces.append({
            "text": " ".join(item["text"] for item in current).strip(),
            "start": current[0]["start"],
            "end": current[-1]["end"],
            "number": sentence["number"],
            "title": sentence["title"],
            "segments": current
        })

    return pieces


def create_sentence_aware_chunks(
    segments,
    target_words=100,
    max_words=150,
    overlap_sentences=1
):
    sentence_units = build_sentence_units(segments)

    if not sentence_units:
        return []

    chunks = []
    current = []
    current_key = None

    def flush_current():
        nonlocal current
        if not current:
            return

        chunks.append(make_chunk(current))

        if overlap_sentences > 0:
            overlap_count = min(overlap_sentences, len(current))
            overlap = current[-overlap_count:]
            if sum(word_count(item) for item in overlap) < max_words:
                current = overlap.copy()
            else:
                current = []
        else:
            current = []

    for sentence in sentence_units:
        sentence_key = (sentence["number"], sentence["title"])

        if current_key is not None and sentence_key != current_key:
            flush_current()
            current = []

        current_key = sentence_key

        long_pieces = split_long_sentence(sentence, max_words)

        if len(long_pieces) > 1 or word_count(sentence) > max_words:
            flush_current()
            current = []

            for piece in long_pieces:
                chunks.append(make_chunk([piece]))

            current = []
            continue

        sentence_words = word_count(sentence)
        current_words = sum(word_count(item) for item in current)

        if current and current_words >= target_words:
            if current_words + sentence_words > max_words:
                flush_current()

        if current and current_words < target_words:
            if current_words + sentence_words > max_words:
                flush_current()

        current.append(sentence)

        if sum(word_count(item) for item in current) >= max_words:
            flush_current()

    if current:
        chunks.append(make_chunk(current))

    return chunks


def main():
    os.makedirs("sentence_jsons", exist_ok=True)

    for filename in os.listdir("jsons"):
        if not filename.endswith(".json"):
            continue

        input_path = os.path.join("jsons", filename)
        output_path = os.path.join("sentence_jsons", filename)

        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        chunks = create_sentence_aware_chunks(
            data["chunks"],
            target_words=100,
            max_words=150,
            overlap_sentences=1
        )

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "chunks": chunks,
                    "text": data.get("text", "")
                },
                f,
                indent=4,
                ensure_ascii=False
            )


if __name__ == "__main__":
    main()