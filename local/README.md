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

**Zasil bazę wiedzy (krok wymagany).** Baza wektorowa startuje pusta — bez tego
kroku endpoint `/ask` nie znajdzie żadnego kontekstu (RAG zwróci puste źródła,
a model będzie zmyślał). Wgraj dane raz po uruchomieniu stacku:

```bash
# z katalogu głównego repo:
curl -X POST "http://localhost:8080/ingest" -F "file=@vector_store/hotel_rules.csv"
```

Sprawdź, że dane się zapisały (powinno pokazać 19 punktów):

```bash
curl -s http://localhost:6333/collections/hotel_rules | grep -o '"points_count":[0-9]*'
```

Zadaj przykładowe pytanie:

```bash
curl -X POST "http://localhost:8080/ask" -H "Content-Type: application/json" \
     -d '{"query": "Ile kosztuje parking hotelowy?"}'
```

Otwórz UI: **http://localhost:8080**

> [!IMPORTANT]
> Dane w Qdrant leżą na nazwanym wolumenie `local_qdrant_storage` i **przetrwają**
> zwykły restart (`docker compose restart`, a także `down`/`up` bez `-v`).
> Znikają dopiero po `docker compose down -v` lub usunięciu wolumenu — po takim
> „czystym" starcie trzeba ponownie uruchomić `/ingest`.

Zatrzymanie: `docker compose down` (dodaj `-v`, aby skasować też dane w Qdrant).

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

Następnie **zasil bazę** tak samo jak w Wariancie A (endpoint `/ingest`) — to
krok wymagany, inaczej `/ask` nie znajdzie kontekstu.

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
