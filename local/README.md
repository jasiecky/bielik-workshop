# Lokalne uruchomienie (bez Google Cloud)

Lokalny odpowiednik architektury RAG z chmury — wszystko na Twojej maszynie,
bez GCP, BigQuery ani tokenów tożsamości.

| Warstwa           | Wersja chmurowa (GCP)          | Wersja lokalna                     |
|-------------------|--------------------------------|------------------------------------|
| LLM               | Bielik na Cloud Run (Ollama)   | Bielik w **natywnej Ollamie**      |
| Embedding         | EmbeddingGemma na Cloud Run    | EmbeddingGemma w **natywnej Ollamie** |
| Baza wektorowa    | BigQuery Vector Search         | **Qdrant** (Docker)                |
| Orchestration API | FastAPI na Cloud Run           | FastAPI (Docker lub natywnie)      |

Endpointy i interfejs WWW są identyczne jak w wersji chmurowej (`/`, `/ingest`,
`/ask`, `/ask_direct`). UI współdzieli plik `orchestration/static/index.html`.

## Wymagania

- **Ollama** zainstalowana i uruchomiona na hoście (`http://localhost:11434`).
  Zalecane GPU, ale działa też na CPU (wolniej).
- **Docker** + Docker Compose.

Pobierz modele (jednorazowo):

```bash
ollama pull SpeakLeash/bielik-4.5b-v3.0-instruct:Q8_0
ollama pull embeddinggemma
```

## Wariant A — pełny Docker (zalecany)

Orchestration + Qdrant w kontenerach; modele serwuje Ollama na hoście
(kontener łączy się z nią przez `host.docker.internal`).

```bash
cd local
docker compose up --build -d
```

Zasil bazę wiedzy i przetestuj:

```bash
# z katalogu głównego repo:
curl -X POST "http://localhost:8080/ingest" -F "file=@vector_store/hotel_rules.csv"

curl -X POST "http://localhost:8080/ask" -H "Content-Type: application/json" \
     -d '{"query": "Ile kosztuje parking hotelowy?"}'
```

Otwórz UI: **http://localhost:8080**

Zatrzymanie: `docker compose down` (dodaj `-v`, aby skasować dane w Qdrant).

## Wariant B — aplikacja natywnie, tylko Qdrant w Dockerze

Wygodne do debugowania kodu `app.py`.

```bash
# 1. Qdrant
docker run -d --name qdrant -p 6333:6333 qdrant/qdrant:v1.12.1

# 2. Aplikacja
cd local
python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt
./.venv/bin/uvicorn app:app --host 0.0.0.0 --port 8080
```

## Konfiguracja (zmienne środowiskowe)

Wszystkie mają sensowne domyślne wartości — zmieniaj tylko w razie potrzeby.

| Zmienna             | Domyślnie                                        | Opis                          |
|---------------------|--------------------------------------------------|-------------------------------|
| `OLLAMA_URL`        | `http://localhost:11434`                         | Adres Ollamy                  |
| `QDRANT_URL`        | `http://localhost:6333`                          | Adres Qdrant                  |
| `QDRANT_COLLECTION` | `hotel_rules`                                    | Nazwa kolekcji wektorów       |
| `LLM_MODEL`         | `SpeakLeash/bielik-4.5b-v3.0-instruct:Q8_0`      | Model LLM w Ollamie           |
| `EMBEDDING_MODEL`   | `embeddinggemma`                                 | Model embeddingowy w Ollamie  |
| `EMBEDDING_DIM`     | `768`                                            | Wymiar wektora (EmbeddingGemma)|
| `TOP_K`             | `3`                                              | Liczba dokumentów do RAG      |

## Jak to działa (RAG)

1. `/ingest` — każdy wiersz CSV (`id,text`) → embedding (EmbeddingGemma) → zapis do Qdrant.
2. `/ask` — pytanie → embedding → wyszukanie `TOP_K` najbliższych wektorów w Qdrant
   (odległość cosinusowa) → zbudowanie promptu z kontekstem → odpowiedź Bielika.
3. `/ask_direct` — pytanie prosto do Bielika, bez RAG (baseline do porównania).

Różnicę widać dobrze np. przy pytaniu o parking: bez RAG model zmyśla
(„bezpłatny"), z RAG odpowiada zgodnie z regulaminem („40 PLN za dzień").
