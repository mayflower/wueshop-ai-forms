FROM python:3.11-slim as form-helper-base

# from https://python-poetry.org/docs/configuration/#available-settings
ENV POETRY_VERSION=1.8.2 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

RUN pip install poetry==${POETRY_VERSION}

WORKDIR /app

COPY pyproject.toml poetry.lock ./

RUN poetry install --no-root

FROM python:3.11-slim as form-helper

ENV VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

COPY --from=form-helper-base ${VIRTUAL_ENV} ${VIRTUAL_ENV}

COPY ./ ./

EXPOSE 8501

CMD ["streamlit" "run" "streamlit_app.py" "--server.port=8501", "--server.address=0.0.0.0"]