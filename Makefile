.PHONY: all venv predictions clean clean-data process-data\
		process-weather-data process-metered-data rawdata\
		notebooks pca-params

# Abbreviations
PY := python3
PIP := python3 -m pip 
VENV_DIR := .venv
VENV_PY := $(VENV_DIR)/bin/$(PY)
VENV_PIP := $(VENV_DIR)/bin/$(PIP)
DATA_METERED_PROCESSED_DIR = data/data_metered_processed
DATA_WEATHER_DAILY_DIR = data/data_weather_daily
DATA_WEATHER_HOURLY_RAW_DIR = data/data_weather_hourly_raw
DATA_WEATHER_HOURLY_PROCESSED_DIR = data/data_weather_hourly_processed
PCA_ENCODING_PARAMS_DIR = pca_params
# Virtual environment

all : venv process-data pca-params notebooks

notebooks : $(VENV_DIR) notebook_pca_anls.py notebook_pipeline_anls.py
	@echo "Creating .ipynb notebooks ... "
	@$(VENV_PY) -m jupytext --to ipynb notebook_pca_anls.py
	@$(VENV_PY) -m jupytext --to ipynb notebook_pipeline_anls.py
	@echo "Finished; two notebooks created."

$(VENV_DIR) : requirements.txt # check requirement changes
	@$(PY) -m venv $(VENV_DIR)
	@$(VENV_PIP) install -r requirements.txt
	@touch $(VENV_DIR)

venv : $(VENV_DIR)

$(DATA_METERED_PROCESSED_DIR) : $(VENV_DIR) src/format_metered.py
	@echo "Processing metered data ... "
	@mkdir -p $(DATA_METERED_PROCESSED_DIR)
	@$(VENV_PY) -m src.format_metered
	@touch $(DATA_METERED_PROCESSED_DIR)

$(DATA_WEATHER_DAILY_DIR) : $(VENV_DIR) src/fetch_weather.py
	@echo "Fetching daily weather data ... "
	@mkdir -p $(DATA_WEATHER_DAILY_DIR)
	@$(VENV_PY) -m src.fetch_weather
	@touch $(DATA_WEATHER_DAILY_DIR)

$(DATA_WEATHER_HOURLY_RAW_DIR) : $(VENV_DIR) src/fetch_weather_hourly.py
	@echo "Fetching hourly weather data ... "
	@mkdir -p $(DATA_WEATHER_HOURLY_RAW_DIR)
	@$(VENV_PY) -m src.fetch_weather_hourly
	@touch $(DATA_WEATHER_HOURLY_RAW_DIR)

$(DATA_WEATHER_HOURLY_PROCESSED_DIR) : $(VENV_DIR) $(DATA_WEATHER_HOURLY_RAW_DIR) src/format_weather_hourly.py
	@echo "Processing hourly weather data ... "
	@mkdir -p $(DATA_WEATHER_HOURLY_PROCESSED_DIR)
	@$(VENV_PY) -m src.format_weather_hourly
	@touch $(DATA_WEATHER_HOURLY_PROCESSED_DIR)

$(PCA_ENCODING_PARAMS_DIR) : $(VENV_DIR) $(DATA_WEATHER_HOURLY_PROCESSED_DIR) $(DATA_WEATHER_DAILY_DIR) src/train_correction.py
	@echo "Creating global pca metered encodings ... "
	@mkdir -p $(PCA_ENCODING_PARAMS_DIR)
	@$(VENV_PY) -m src.train_correction
	@touch $(PCA_ENCODING_PARAMS_DIR)

pca-params : $(PCA_ENCODING_PARAMS_DIR)

predictions : $(VENV_DIR) $(DATA_METERED_PROCESSED_DIR) $(DATA_WEATHER_HOURLY_PROCESSED_DIR) $(PCA_ENCODING_PARAMS_DIR) src/make_predictions.py
	@$(VENV_PY) -m src.make_predictions

process-weather-data : $(DATA_WEATHER_DAILY_DIR) $(DATA_WEATHER_HOURLY_PROCESSED_DIR)

process-metered-data : $(DATA_METERED_PROCESSED_DIR)

process-data : process-metered-data process-weather-data

rawdata : clean-data $(DATA_WEATHER_DAILY_DIR) $(DATA_WEATHER_HOURLY_RAW_DIR)

# Clean ups
clean-data: 
	@rm -rf $(DATA_METERED_PROCESSED_DIR)
	@rm -rf $(DATA_WEATHER_DAILY_DIR)
	@rm -rf $(DATA_WEATHER_HOURLY_RAW_DIR)
	@rm -rf $(DATA_WEATHER_HOURLY_PROCESSED_DIR)
	@rm -rf $(PCA_ENCODING_PARAMS_DIR)

clean : clean-data
	@rm -rf $(VENV_DIR)
	@find . -type f -name "*.pyc" -delete
	@find . -type f -name "*.ipynb" -delete
	@find . -type d -name "__pycache__" -delete
	@find . -type d -name ".pytest_cache"  -exec rm -rf {} +
	@find . -type d -name '.ipynb_checkpoints' -exec rm -rf {} +
	@find src -type d -name '.ipynb_checkpoints' -exec rm -rf {} +
