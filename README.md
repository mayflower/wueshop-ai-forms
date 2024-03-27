# Formular-Helfer

Dieser Agent soll einem Nutzer dabei helfen, herauszufinden, welche Formulare er für ein Anliegen ausfüllen muss.

Dabei weiß der Nutzer vielleicht nicht, ob oder welche Formulare er braucht um Vorschriften für sein Vorhaben einzuhalten.


## Techstack

- Fürs Frontend ist [Streamlit](https://docs.streamlit.io/get-started/fundamentals/main-concepts#development-flow) vorinstalliert. Falls das nicht ausreicht, muss man ggf. mit einem echten Frontend aufrüsten
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


###

Entscheidungshilfe bitte. Was soll ich machen. Docker oder Lokal?
- Lokal ist meist simpler. Debuggen, Hot reload und Python-Pakete nachinstallieren geht flotter. Für dieses Projekt vermutlich empfehlenswert.
- Mit Docker stellt man sicher, dass es wirklich auch bei jedem Teammitglied funktioniert. Falls man mehrere Unterprojekte hat (Frontend, 2 Backends, Datenbank etc...) kommt man um Docker nicht mehr herum.

### Fertig?
Die App sollte dann unter `http://127.0.0.1:8501/` laufen

## Code Formatierung

Wenn man hübschen Code haben möchte, kann man [Ruff](https://docs.astral.sh/ruff/) installieren (Ist dabei, wenn man `poetry install` macht). Ruff kann:
- Code formatieren (alles sieht immer gleich und hübsch aus). `ruff format`
- Code linten (Python Syntax prüfen und bei Bedarf Fehler korrigieren) `ruff check` und `ruff check --fix`

Idealerweise führt man dann immer `ruff format` vor einem `git commit` aus.

## PDF Loaders

Dazu gibt es eine [Langchain Doku](https://python.langchain.com/docs/modules/data_connection/document_loaders/pdf)
