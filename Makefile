# Copyright (c) 2026 Takenori Yoshimura
# Released under the MIT License.

PROJECT := lfeats

PYTHON_VERSION     := 3.11
TORCH_VERSION      := 2.4.0
TORCHAUDIO_VERSION := 2.4.0
PLATFORM           := cu121

venv:
	test -d .venv || python$(PYTHON_VERSION) -m venv .venv
	. .venv/bin/activate && python -m pip install --upgrade pip
	. .venv/bin/activate && python -m pip install --upgrade wheel
	. .venv/bin/activate && python -m pip install torch==$(TORCH_VERSION)+$(PLATFORM) torchaudio==$(TORCHAUDIO_VERSION)+$(PLATFORM) \
		--index-url https://download.pytorch.org/whl/$(PLATFORM)
	. .venv/bin/activate && python -m pip install -e .[dev]

dist:
	. .venv/bin/activate && python -m build
	. .venv/bin/activate && python -m twine check dist/*

dist-clean:
	rm -rf dist

doc:
	. .venv/bin/activate && cd docs && make html

doc-clean:
	@if [ -f .venv/bin/activate ]; then \
		. .venv/bin/activate && cd docs && make clean; \
	fi

check: tool
	. .venv/bin/activate && python -m ruff check $(PROJECT) tests
	. .venv/bin/activate && python -m ruff format --check $(PROJECT) tests
	. .venv/bin/activate && python -m pyright $(PROJECT) tests
	. .venv/bin/activate && python -m mdformat --check *.md
	./tools/taplo/taplo fmt --check *.toml
	./tools/yamlfmt/yamlfmt --lint *.yml .github/workflows/*.yml

format: tool
	. .venv/bin/activate && python -m ruff check --fix $(PROJECT) tests
	. .venv/bin/activate && python -m ruff format $(PROJECT) tests
	. .venv/bin/activate && python -m mdformat *.md
	./tools/taplo/taplo fmt *.toml
	./tools/yamlfmt/yamlfmt *.yml .github/workflows/*.yml

test-all: test-example test

test: tool
	. .venv/bin/activate && python -m pytest

test-example: tool
	. .venv/bin/activate &&	python -m pytest --doctest-modules --no-cov --ignore=$(PROJECT)/third_party

test-clean:
	rm -rf tests/__pycache__

tool:
	cd tools && make

tool-clean:
	cd tools && make clean

update: tool
	. .venv/bin/activate && python -m pip install --upgrade pip
	@./tools/taplo/taplo get -f pyproject.toml project.optional-dependencies.dev | while read -r package; do \
		. .venv/bin/activate && python -m pip install --upgrade "$$package"; \
	done

clean: dist-clean doc-clean test-clean tool-clean
	rm -rf .venv
	find . -name "__pycache__" -type d | xargs rm -rf

.PHONY: venv dist dist-clean doc doc-clean check format test test-clean tool tool-clean update clean
