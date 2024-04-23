VENV_NAME := ml-env
PYTHON := python3
SCRIPT := main.py

all: run

setup:
	@test -d $(VENV_NAME) || $(PYTHON) -m venv $(VENV_NAME)
	@echo "Activating virtual environment and installing dependencies..."
	@. $(VENV_NAME)/bin/activate; pip install -r requirements

run:
	@. $(VENV_NAME)/bin/activate; $(PYTHON) $(SCRIPT)

predict:
	@. $(VENV_NAME)/bin/activate; $(PYTHON) predict.py

fclean:
	@echo "Cleaning up..."
	rm -rf $(VENV_NAME)
	@echo "Virtual environment removed."

.PHONY: all setup run fclean
