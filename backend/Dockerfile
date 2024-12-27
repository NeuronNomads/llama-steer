# Build stage for model download
# We don't need these tools after downloading 
FROM google/cloud-sdk:slim as model-downloader

# Download model files to a specific directory
RUN mkdir /model
RUN gsutil -m cp -r gs://llama-steer-models/5f0b02c75b57c5855da9ae460ce51323ea669d8a/* /model/

FROM python:3.12.4-slim

ARG HF_TOKEN
ENV HF_TOKEN=$HF_TOKEN
ENV POETRY_VERSION=1.8.3
ENV WEIGHTS_PATH=/app/model

RUN pip install --no-cache-dir "poetry==$POETRY_VERSION"

WORKDIR /app

# Copy model files from the downloader stage
COPY --from=model-downloader /model /app/model

COPY poetry.lock pyproject.toml /app/

COPY api/ /app/api/

RUN poetry config virtualenvs.in-project true

RUN poetry install --no-ansi

EXPOSE 8080

CMD ["poetry", "run", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]