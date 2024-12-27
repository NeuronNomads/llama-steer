FROM python:3.12.4-slim

ARG HF_TOKEN
ENV HF_TOKEN=$HF_TOKEN
ENV POETRY_VERSION=1.8.3

RUN pip install --no-cache-dir "poetry==$POETRY_VERSION"

WORKDIR /app

COPY poetry.lock pyproject.toml /app/

COPY api/ /app/api/

RUN poetry config virtualenvs.in-project true

RUN poetry install --no-ansi

EXPOSE 8000

CMD ["poetry", "run", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]