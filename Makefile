.PHONY: venv predictions clean

# Abbreviations
PY := python3
PIP := python3 -m pip 
VENV_DIR := .venv
VENV_PY := $(VENV_DIR)/bin/$(PY)
VENV_PIP := $(VENV_DIR)/bin/$(PIP)
# Virtual environment

$(VENV_DIR) : requirements.txt # check requirement changes
	$(PY) -m venv $(VENV_DIR)
	$(VENV_PIP) install -r requirements.txt
	touch $(VENV_DIR)

predictions : $(VENV_DIR)
	$(VENV_PY) src/dummy.py

venv : $(VENV_DIR)

# Clean ups
clean : 
	rm -rf $(VENV_DIR)
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".pytest_cache"  -exec rm -rf {} +


