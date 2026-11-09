# Wine Quality Prediction System

This project is a classroom prototype that predicts whether a red wine sample is likely to be `GOOD QUALITY` or `NEEDS IMPROVEMENT` based on its physicochemical measurements.

## Problem
A small wine-production business needs a simple preliminary screening tool to review batches before further laboratory or sensory testing.

## Dataset
The model is trained on the UCI Wine Quality dataset for red wine. The original target is the wine quality score, which is converted into a binary label:

- `quality >= 6` -> `GOOD QUALITY`
- `quality < 6` -> `NEEDS IMPROVEMENT`

Source: https://archive.ics.uci.edu/dataset/186/wine+quality

## Prediction rule
The model uses the physicochemical attributes as input features, but the original `quality` score is excluded from the predictor list to avoid target leakage.

## Technologies used
- Python 3.10+
- pandas
- scikit-learn
- Streamlit
- joblib

## Project files
- `train_model.py` retrieves the dataset, trains the model, evaluates it, and saves `wine_model.pkl`
- `app.py` builds the Streamlit dashboard and prediction interface
- `requirements.txt` lists the required Python packages

## Installation
```bash
pip install -r requirements.txt
```

## Run the model trainer
```bash
python train_model.py
```

## Run the app
```bash
streamlit run app.py
```

## Responsible use
This is a classroom prototype based on a public dataset. It is not a substitute for laboratory testing, sensory evaluation, food-safety procedures, or professional quality-control decisions.

## Limitations
The model is trained on one specific red-wine dataset and may not generalize perfectly to other producers, regions, or testing conditions.
