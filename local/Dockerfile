# Obraz lokalnej aplikacji orchestration.
# Budowany z kontekstu katalogu głównego repo (patrz docker-compose.yml),
# aby mieć dostęp do współdzielonego katalogu orchestration/static.
FROM python:3.11-slim

WORKDIR /app

COPY local/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Kod lokalnej aplikacji + współdzielony statyczny UI
COPY local/app.py /app/app.py
COPY orchestration/static /orchestration/static

# W kontenerze static leży pod /orchestration/static (app.py szuka ../orchestration/static
# względem /app, czyli /orchestration/static) - ścieżka się zgadza.
ENV STATIC_DIR=/orchestration/static

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
