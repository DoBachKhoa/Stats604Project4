# Stats 604 - Project 4

Implementation of Project 4 in the course Stats 604. Project's aim: to produce hourly PJM electric load predictions of 29 zones
for 10 days: from 20 to 29 of November 2025. For each day of the 10 days, the task is to predict
* Hourly electric loads of 29 zones
* Peak hour per zone
* Per zone, whether that day is one of the two peak days over that 10-day period

The project includes fetcher that loads data saved at the professor's provided [OSF link](https://files.osf.io/v1/resources/Py3u6/providers/osfstorage/?zip=). It also includes code to download weather data and weather prediction.

## Contents
```
.
├─ data/
│  ├─ data_metered_raw/             # raw PJM data, 2016 - Oct 2025
│  ├─ data_metered_processed/       # processed electric load data
│  ├─ data_weather_daily/           # daily weather data from meteostat
│  ├─ data_weather_hourly_raw/      # hourly weather data from meteostat
│  ├─ data_weather_hourly_process/  # pca decomposed hourly weather data
│  └─ zones_locations.csv           # zone → lat/lon
├─ src/
│  ├─ __init__.py                   # init file
│  ├─ constants.py                  # storing constant variables
│  ├─ data_loader.py                # helper data handling functions
│  ├─ fetch_metered.py              # loads raw PJM data
│  ├─ fetch_weather_hourly.py       # loads daily weather data
│  ├─ fetch_weather_predictions.py  # loads open meteo weather prediction
│  ├─ fetch_weather.py              # loads daily weather data
│  ├─ format_metered.py             # formats raw PJM data into processed
│  ├─ format_weather_hourly.py      # formats hourly weather data
│  ├─ make_predictions.py           # script that makes predictions for next day
│  ├─ pipeline.py                   # prediction pipelines
│  ├─ train_correction.py           # trains correction models for holiday effects
│  └─ utils.py                      # utilization functions
├─ Makefile
├─ notebook_pca_anls.py
├─ notebook_pipeline_anls.py
├─ README.md
└─ requirements.txt
```

Zone order (fixed):
AECO, AEPAPT, AEPIMP, AEPKPT, AEPOPT, AP, BC, CE, DAY, DEOK, DOM, DPLCO, DUQ, EASTON, EKPC, JC, ME, OE, OVEC, PAPWR, PE, PEPCO, PLCO, PN, PS, RECO, SMECO, UGI, VMEU

## Quick start (local)

```
make env                # create venv, download dependencies
make process-data       # download rawdata and process them
make pca-params         # trains pca encodings and correction models for holiday effects
make notebooks          # generate analysis notebooks
make predictions        # generate next-day predictions
```

### Useful Make targets
```
make                     # creates venv, process data, generate analysis notebooks
make predictions         # print out next-day predictions
make raw-data            # deletes and reloads raw electric load and weather data
make process-data        # generates processed data from raw data
make clean               # delete derived artifacts (keeps raw data and code)
```

Note: prediction outputs are hourly EPT timed
  
## Docker

Multi-arch docker image
```
dobachkhoa/stats604_proj4:latest
```

Interactive shell:
```
docker run -it --rm dobachkhoa/stats604_proj4:latest
```

One-liner predictions (prints and exits):
```
docker run -it --rm dobachkhoa/stats604_proj4:latest make predictions
```
