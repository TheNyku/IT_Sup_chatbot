from __future__ import annotations

import os
from pathlib import Path

import chromadb
import voyageai
from dotenv import load_dotenv

BASE_DIR = Path(__file__).parent
load_dotenv(BASE_DIR / ".env")

FAQ_PATH = BASE_DIR / "data" / "it_support_faq.md"
DB_PATH = BASE_DIR / "chroma_db"
COLLECTION_NAME = "it_support_faq"
EMBED_MODEL = "voyage-3.5"


def load_faq_chunks(path: Path = FAQ_PATH) -> list[dict]:
    """Split the FAQ markdown into one chunk per '## ' section."""
    text = path.read_text(encoding="utf-8")
    chunks = []
    for section in text.split("\n## ")[1:]:  
        title, _, body = section.partition("\n")
        chunks.append({"title": title.strip(), "text": f"{title.strip()}\n{body.strip()}"})
    return chunks


class ITSupportRAG:
    
    def __init__(self, faq_path: Path = FAQ_PATH, db_path: Path = DB_PATH):
        self.voyage = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"), max_retries=3)
        self.chroma = chromadb.PersistentClient(path=str(db_path))
        self.collection = self.chroma.get_or_create_collection(
            COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
        )
        self.faq_path = faq_path
        if self.collection.count() == 0:
            self.index_documents()

    def _embed(self, texts: list[str], input_type: str) -> list[list[float]]:
        return self.voyage.embed(texts, model=EMBED_MODEL, input_type=input_type).embeddings

    def index_documents(self) -> int:
        
        existing = self.collection.get()["ids"]
        if existing:
            self.collection.delete(ids=existing)
        chunks = load_faq_chunks(self.faq_path)
        texts = [c["text"] for c in chunks]
        self.collection.add(
            ids=[f"faq-{i}" for i in range(len(chunks))],
            documents=texts,
            embeddings=self._embed(texts, input_type="document"),
            metadatas=[{"title": c["title"]} for c in chunks],
        )
        return len(chunks)

    def retrieve(self, query: str, k: int = 3) -> list[dict]:
        
        result = self.collection.query(
            query_embeddings=self._embed([query], input_type="query"),
            n_results=k,
        )
        return [
            {"title": meta["title"], "text": doc, "score": round(1 - dist, 3)}
            for doc, meta, dist in zip(
                result["documents"][0], result["metadatas"][0], result["distances"][0]
            )
        ]

    def search_as_text(self, query: str, k: int = 3) -> str:
        """Retrieve and format the results as plain text for the LLM."""
        hits = self.retrieve(query, k)
        if not hits:
            return "No relevant documentation found."
        return "\n\n---\n\n".join(
            f"[{h['title']}] (relevance {h['score']})\n{h['text']}" for h in hits
        )


if __name__ == "__main__":
    rag = ITSupportRAG()
    print(f"Indexed chunks: {rag.collection.count()}")
    print(rag.search_as_text("I forgot my password"))
