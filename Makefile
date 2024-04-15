VENV_NAME := ml-env
PYTHON := python3
SCRIPT := test.py

all: run

setup:
	@test -d $(VENV_NAME) || $(PYTHON) -m venv $(VENV_NAME)
	@echo "Activating virtual environment and installing dependencies..."
	@. $(VENV_NAME)/bin/activate; pip install pandas matplotlib numpy scikit-learn

run:
	@echo "Running script $(SCRIPT)..."
	@. $(VENV_NAME)/bin/activate; $(PYTHON) $(SCRIPT)

fclean:
	@echo "Cleaning up..."
	rm -rf $(VENV_NAME)
	@echo "Virtual environment removed."

.PHONY: all setup run fclean