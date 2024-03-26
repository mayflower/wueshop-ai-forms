# Formular-Helfer

Dieser Agent soll einem Nutzer dabei helfen, herauszufinden, welche Formulare er für ein Anliegen ausfüllen muss.

Dabei weiß der Nutzer vielleicht nicht, ob oder welche Formulare er braucht um Vorschriften für sein Vorhaben einzuhalten.


## Techstack

- Fürs Frontend ist Streamlit vorinstalliert. Falls das nicht ausreicht, muss man ggf. mit einem echten Frontend aufrüsten
- Der Agent wird mit LangChain und einem OpenAI-Modell gebaut.

## Setup

- Wie bekant die .env.dist kopieren, in .env umbenennen und einen OpenAI-Key eintragen
- Sicherstellen, dass man im Verzeichnis ist

### Mit Docker Compose

- `docker compose build --no-cache`
- `docker compose up`

### Lokal mit poetry
- [poetry installieren](https://python-poetry.org/docs/#installation)
- `poetry install` installiert die Python-Umgebung
- `poetry shell` öffnet eine shell im Terminal mit der eben installierten Python-Umgebung
- `streamlit run streamlit_app.py --server.port=8501` startet Gradio

Die App sollte dann unter `http://127.0.0.1:8501/` laufen

### PDF Loaders

Dazu gibt es eine [Langchain Doku](https://python.langchain.com/docs/modules/data_connection/document_loaders/pdf)
