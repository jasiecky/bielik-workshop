"""
Lokalna wersja orchestration API (Bielik + EmbeddingGemma + Qdrant).

Odpowiednik `orchestration/main.py`, ale zamiast:
  - Cloud Run + tokenów tożsamości GCP -> lokalna Ollama (bez auth),
  - BigQuery Vector Search               -> Qdrant.

Te same endpointy i ten sam interfejs WWW co w wersji chmurowej:
  GET  /            -> statyczny index.html
  POST /ingest      -> wgrywa CSV (kolumny: id,text) do bazy wektorowej
  POST /ask         -> RAG: embedding -> Qdrant top_k -> Bielik
  POST /ask_direct  -> pytanie bezpośrednio do Bielika (baseline, bez RAG)
"""

import os
import csv
import io
import re
import requests
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# --- Konfiguracja (wszystko przez zmienne środowiskowe, z sensownymi domyślnymi) ---
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
COLLECTION = os.environ.get("QDRANT_COLLECTION", "hotel_rules")

EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "embeddinggemma")
LLM_MODEL = os.environ.get("LLM_MODEL", "SpeakLeash/bielik-4.5b-v3.0-instruct:Q8_0")
EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", "768"))
TOP_K = int(os.environ.get("TOP_K", "3"))

# Katalog ze statycznym UI - współdzielony z wersją chmurową.
STATIC_DIR = os.environ.get(
    "STATIC_DIR",
    os.path.join(os.path.dirname(__file__), "..", "orchestration", "static"),
)

app = FastAPI(title="RAG API - LOCAL (Bielik & EmbeddingGemma & Qdrant)")

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

qdrant = QdrantClient(url=QDRANT_URL)


def ensure_collection() -> None:
    """Tworzy kolekcję w Qdrant jeśli jeszcze nie istnieje (odpowiednik init_db.py)."""
    if not qdrant.collection_exists(COLLECTION):
        qdrant.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )


@app.on_event("startup")
def _startup() -> None:
    ensure_collection()


@app.get("/")
def read_root():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


def get_embedding(text: str) -> list[float]:
    """Zamiana tekstu na wektor przy pomocy modelu EmbeddingGemma w lokalnej Ollamie."""
    url = f"{OLLAMA_URL}/api/embed"
    payload = {"model": EMBEDDING_MODEL, "input": text}
    response = requests.post(url, json=payload, timeout=120)
    response.raise_for_status()
    return response.json().get("embeddings", [[]])[0]


# Tokeny sterujące używane w szablonie czatu Bielika spakowanego w Ollamie
# (szablon w stylu Llama-3 - sam Bielik nie jest pochodną Llama-3; jego oficjalny
# format to ChatML). Modelfile zatrzymuje generację tylko na części z nich, przez co
# pozostałe (np. <|eom_id|>, <|chat_token|>) potrafią wyciec do treści odpowiedzi.
# Przekazujemy je jako dodatkowe stop-tokeny, a ewentualne resztki usuwamy niżej.
LLM_STOP_TOKENS = [
    "<|eot_id|>",
    "<|eom_id|>",
    "<|start_header_id|>",
    "<|end_header_id|>",
    "<|chat_token|>",
]

# Dowolny token w formacie <|...|> - do wyczyszczenia z gotowej odpowiedzi.
_SPECIAL_TOKEN_RE = re.compile(r"<\|[^|]*\|>")


def clean_answer(text: str) -> str:
    """Usuwa resztkowe specjalne tokeny (<|...|>) i nadmiarowe białe znaki."""
    return _SPECIAL_TOKEN_RE.sub("", text).strip()


def call_llm(user_content: str, system: str | None = None) -> str:
    """Wysłanie zapytania do modelu Bielik w lokalnej Ollamie.

    Instrukcję przekazujemy jako wiadomość systemową, a właściwe pytanie jako
    wiadomość użytkownika. Rozdzielenie ról (zamiast upychania wszystkiego w
    turze `user`) sprawia, że mały model nie powtarza szablonu promptu.
    """
    url = f"{OLLAMA_URL}/api/chat"
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user_content})

    payload = {
        "model": LLM_MODEL,
        "messages": messages,
        "stream": False,
        "options": {"stop": LLM_STOP_TOKENS},
    }
    response = requests.post(url, json=payload, timeout=300)
    response.raise_for_status()
    answer = clean_answer(response.json().get("message", {}).get("content", ""))
    if not answer:
        answer = "Nie udało się wygenerować odpowiedzi. Spróbuj przeformułować pytanie."
    return answer


class AskRequest(BaseModel):
    query: str


@app.post("/ingest")
async def ingest_csv(file: UploadFile = File(...)):
    ensure_collection()

    content = await file.read()
    csv_reader = csv.DictReader(io.StringIO(content.decode("utf-8")))

    points = []
    for row in csv_reader:
        doc_id = row.get("id")
        text = row.get("text")
        if not doc_id or not text:
            continue
        try:
            embedding = get_embedding(text)
            points.append(
                PointStruct(
                    id=int(doc_id) if str(doc_id).isdigit() else doc_id,
                    vector=embedding,
                    payload={"content": text},
                )
            )
        except Exception as e:
            print(f"Błąd w generowaniu osadzenia dla '{text}': {e}")

    if points:
        qdrant.upsert(collection_name=COLLECTION, points=points)

    return {"status": "success", "inserted_count": len(points)}


@app.post("/ask")
async def ask_question(request_data: AskRequest):
    query = request_data.query

    try:
        query_embedding = get_embedding(query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Błąd generowania wektora zapytania: {e}")

    # Krok 1: Wyszukiwanie wektorowe w Qdrant (odpowiednik BigQuery VECTOR_SEARCH)
    try:
        hits = qdrant.query_points(
            collection_name=COLLECTION,
            query=query_embedding,
            limit=TOP_K,
            with_payload=True,
        ).points
        context_docs = [h.payload.get("content", "") for h in hits]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Błąd przeszukiwania wektorowego w Qdrant: {e}")

    # Krok 2: Zbudowanie promptu z kontekstem i zapytanie do LLM
    context_text = "\n\n".join(context_docs)
    system_prompt = (
        "Jesteś pomocnym asystentem odpowiadającym na pytania dotyczące zasad hotelowych. "
        "Odpowiedz na pytanie użytkownika bazując TYLKO na dostarczonym kontekście.\n\n"
        f"KONTEKST:\n{context_text}"
    )

    try:
        answer = call_llm(query, system=system_prompt)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Błąd podczas komunikacji z modelem LLM: {e}")

    return {"answer": answer, "context_used": context_docs}


@app.post("/ask_direct")
async def ask_direct(request_data: AskRequest):
    query = request_data.query
    system_prompt = "Jesteś pomocnym asystentem. Odpowiedz na pytanie użytkownika w sposób jasny i zwięzły."

    try:
        answer = call_llm(query, system=system_prompt)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Błąd podczas komunikacji z modelem LLM: {e}")

    return {"answer": answer}
