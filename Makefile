VENV_NAME := venv
PYTHON := python3
SCRIPT := srcs/main.py

all: run

setup:
	@test -d $(VENV_NAME) || $(PYTHON) -m venv $(VENV_NAME)
	@echo "Activating virtual environment and installing dependencies..."
	@. $(VENV_NAME)/bin/activate; pip install -r config/requirements

freeze:
	@. $(VENV_NAME)/bin/activate; pip freeze > config/requirements

run:
	@if [ ! -d $(VENV_NAME) ]; then make setup; fi
	@. $(VENV_NAME)/bin/activate; $(PYTHON) $(SCRIPT)

predict:
	@. $(VENV_NAME)/bin/activate; $(PYTHON) srcs/predict.py

fclean:
	rm -rf $(VENV_NAME)
	rm -rf __pycache__
	rm -rf models/*

.PHONY: all setup run fclean
