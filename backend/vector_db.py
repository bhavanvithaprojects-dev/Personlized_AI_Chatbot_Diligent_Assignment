import os
from pinecone import Pinecone, ServerlessSpec
from backend.embeddings import get_embedding

# ---------- CONFIG ----------
PINECONE_API_KEY = "pcsk_5z7hJG_E5Tpkdwt7RzLgi29vHNLLpbm6HrgZVh6PEbT1VcnatikxakBTGjBqFD1WrojEbe"
INDEX_NAME = "jarvis-index"
DIMENSION = 384  # all-MiniLM-L6-v2
# ----------------------------

# Create Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

# Create index if it does not exist
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

# Connect to index
index = pc.Index(INDEX_NAME)


def store_document(doc_id: str, text: str):
    embedding = get_embedding(text)
    index.upsert(
        vectors=[
            {
                "id": doc_id,
                "values": embedding,
                "metadata": {"text": text}
            }
        ]
    )


def retrieve_context(query: str, top_k: int = 3) -> str:
    query_embedding = get_embedding(query)

    result = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )

    if not result.matches:
        return "No relevant context found."

    return " ".join(
        match.metadata["text"] for match in result.matches
    )
