import os
import joblib
from dotenv import load_dotenv

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct
)

# 1. Load Qdrant credentials
load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_KEY")

# 2. Connect to Qdrant Cloud
client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

collection_name = "sigma_web_d_course_chunks"

# 3. Load existing embeddings
df = joblib.load("embeddings.joblib")

print("Rows:", len(df))
print("Columns:", df.columns)
print("Embedding dimension:", len(df["embedding"].iloc[0]))

# 4. Create collection
vector_size = len(df["embedding"].iloc[0])

if not client.collection_exists(collection_name):
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=vector_size,
            distance=Distance.COSINE
        )
    )

print("Collection ready!")

# 5. Upload existing embeddings + metadata
batch_size = 100

for start_idx in range(0, len(df), batch_size):

    batch = df.iloc[start_idx:start_idx + batch_size]

    points = []

    for index, row in batch.iterrows():

        points.append(
            PointStruct(
                id=int(index),

                vector=row["embedding"],

                payload={
                    "title": row["title"],
                    "number": row["number"],
                    "start": row["start"],
                    "end": row["end"],
                    "text": row["text"]
                }
            )
        )

    client.upsert(
        collection_name=collection_name,
        points=points
    )

    print(
        f"Uploaded "
        f"{min(start_idx + batch_size, len(df))}"
        f"/{len(df)}"
    )

# 6. Verify
collection_info = client.get_collection(collection_name)

print("\nMigration completed!")
print(collection_info)