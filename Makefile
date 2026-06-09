.PHONY: install test lint run

install:
	pip install -e ".[dev,phase2,phase3]"

test:
	pytest tests/ -v

lint:
	ruff check src/ tests/

run:
	streamlit run src/tradegame/dashboard/app.py
