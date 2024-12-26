.PHONY: install
install:
	@poetry config virtualenvs.in-project true
	@poetry install

.PHONY: auth
auth:
	@gcloud auth application-default login
	@gcloud config set project llama-steer

.PHONY: run
run:
	@poetry run python -m uvicorn api.main:app --reload --reload-dir api --port=8000