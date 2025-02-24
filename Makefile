.PHONY: install test format clean

install:
	python -m venv .venv
	. .venv/bin/activate && pip install -r requirements.txt

install-arm64:
	python -m venv .venv
	. .venv/bin/activate && pip install wheels/numpy-2.2.3-cp312-cp312-macosx_14_0_arm64.whl wheels/scipy-1.13.1-cp312-cp312-macosx_12_0_arm64.whl
	. .venv/bin/activate && pip install -r requirements.txt

test:
	. .venv/bin/activate && pytest

format:
	. .venv/bin/activate && black .
	. .venv/bin/activate && isort .

clean:
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type d -name "*.egg-info" -exec rm -r {} +
	find . -type d -name ".pytest_cache" -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".venv" -exec rm -rf {} +