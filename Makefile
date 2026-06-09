.PHONY: install install-live test lint run live

install:
	pip install -e ".[dev,phase2,phase3]"

install-live:
	pip install -e ".[dev,live,phase2,phase3]"

test:
	pytest tests/ -v

lint:
	ruff check src/ tests/

run:
	streamlit run src/tradegame/dashboard/app.py

live:
	python -m tradegame.live.runner
