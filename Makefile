RHAI_INDEX_URL := https://console.redhat.com/api/pypi/public-rhai/rhoai/3.4/cpu-ubi9/simple/

.PHONY: test coverage lint format typecheck check build lock install install-dev

test:
	pytest tests -v

coverage:
	pytest tests -v --cov=llama_stack_provider_trustyai_garak --cov-report=term-missing

lint:
	ruff check src/ tests/

format:
	ruff format src/ tests/

typecheck:
	mypy src/

check: lint typecheck test

build:
	docker build -f Containerfile -t trustyai-garak:dev .

lock:
	uv pip compile \
		--python-platform linux \
		--extra inline \
		--emit-index-url \
		--default-index $(RHAI_INDEX_URL) \
		pyproject.toml \
		--index-url $(RHAI_INDEX_URL) \
		-o requirements.txt

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"
	pre-commit install
