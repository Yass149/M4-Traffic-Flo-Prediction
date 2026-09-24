# Traffic flow prediction

An end-to-end traffic forecasting study combining historical traffic counts, weather signals and calendar features. The repository compares regression and time-series approaches on 15-minute traffic-flow observations and keeps modelling logic outside the presentation notebook.

> **Project status:** reproducible analysis project. The repository contains the modelling pipeline and rendered analysis, but no public prediction API or continuously deployed service.

## What this project demonstrates

- Resampling traffic observations onto a consistent 15-minute time grid.
- Cleaning and transforming weather and calendar inputs.
- Engineering time, holiday and seasonal features.
- Comparing regression and time-series forecasts with visual diagnostics.
- Separating reusable Python modules from the final notebook/report.

## Pipeline

```text
traffic + weather + calendar data
                │
                ▼
      parsing, validation and joins
                │
                ▼
       15-minute feature table
                │
        ┌───────┴────────┐
        ▼                ▼
   regression        time series
        └───────┬────────┘
                ▼
     forecast comparison and plots
```

The feature contract is explicit: `timestamp` is the datetime column, `total_volume` is the target, and all observations are aligned to 15-minute intervals before modelling.

## Repository layout

```text
modules/
  audit.py          data quality checks
  preprocessing.py  parsing and cleaning
  features.py       calendar and modelling features
  regression.py     regression workflows
  timeseries.py     time-series workflows
  visualization.py  charts and diagnostics
count.py             command-line entry point
02_CSMAD_CW2.ipynb   presentation notebook
02_CSMAD_CW2.html    rendered notebook output
data/                source data and documentation
```

The notebook is the readable report; the implementation lives in `modules/` so it can be tested, reviewed and reused.

## Run locally

```bash
git clone https://github.com/Yass149/M4-Traffic-Flo-Prediction.git
cd M4-Traffic-Flo-Prediction
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt  # add project dependencies if using a fresh environment
python count.py --help
```

The repository was developed with Python 3.11. Run the notebook or open the committed HTML report to inspect the analysis. Dataset availability and licensing follow the source-data documentation in `data/`.

## Engineering notes

The code is organised so data loading, preprocessing, features, models and visualisation can be changed independently. The current repository should be treated as a research/portfolio pipeline rather than a production service: it does not include CI, model registry, scheduled retraining, online monitoring or an authenticated serving layer.

## Next production step

Package the feature pipeline behind a versioned batch job or API, add time-based backtesting and regression tests for data contracts, persist model artefacts with metadata, and monitor forecast error and data drift after deployment.

## Context

Built for the University of Reading CSMAD coursework in collaboration with the project team. Assessment deadlines and team-work instructions are intentionally kept out of the project landing page.
