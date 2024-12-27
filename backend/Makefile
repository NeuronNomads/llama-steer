.PHONY: install
install:
	@poetry config virtualenvs.in-project true
	@poetry install

.PHONY: auth
auth:
	@gcloud auth application-default login
	@gcloud config set project llama-steer
	@gcloud auth configure-docker us-central1-docker.pkg.dev

.PHONY: run-be
run-be:
	@poetry run python -m uvicorn api.main:app --reload --reload-dir api --port=8000

.PHONY: build
build: 
	@docker build --platform linux/amd64 --build-arg HF_TOKEN=${HF_TOKEN} -t us-central1-docker.pkg.dev/llama-steer/backend/llama-steer .

.PHONY: run
run: 
	@docker run -p 8000:8000 llama-steer

.PHONY: push
push:
	@docker push us-central1-docker.pkg.dev/llama-steer/backend/llama-steer

