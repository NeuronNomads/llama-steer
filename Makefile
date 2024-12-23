.PHONY: install
install:
	@poetry config virtualenvs.in-project true
	@poetry install

.PHONY: auth
auth:
	@gcloud auth application-default login
	@gcloud config set project llama-steer