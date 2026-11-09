# Wine Quality Prediction System

## Problem
A small wine-production business needs a quick preliminary screening aid to decide whether a wine sample appears to be of good quality or if it needs further improvement before more detailed testing.

## Dataset
This prototype uses the UCI Wine Quality dataset for red wine. The original dataset is available at:
https://archive.ics.uci.edu/dataset/186/wine+quality

## Prediction rule
The model classifies a wine sample as:
- GOOD QUALITY when quality >= 6
- NEEDS IMPROVEMENT when quality < 6

The original quality score is used only to generate the target label and is excluded from the model input features.

## Technologies
- Python 3.10+
- pandas
- scikit-learn
- Streamlit
- joblib

## Project structure
- `train_model.py` — downloads the dataset, prepares the target, trains the Random Forest model, and saves `wine_model.pkl`
- `app.py` — Streamlit dashboard for uploading CSV inputs and displaying predictions
- `requirements.txt` — Python dependencies
- `wine_quality.csv` — downloaded red-wine dataset
- `wine_model.pkl` — trained model artifact

## Setup
1. Create a virtual environment (optional but recommended).
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Train the model:
   ```bash
   python train_model.py
   ```
4. Run the app:
   ```bash
   streamlit run app.py
   ```

## Usage
- Open the local Streamlit URL shown in the terminal.
- Select **Open** on the welcome screen to enter the predictor dashboard.
- Download the sample input CSV from the app.
- Upload a CSV file containing the same chemical measurements used during model training.
- Review the quality prediction and download the results file.

## Deploy on Streamlit Community Cloud
1. Sign in at [Streamlit Community Cloud](https://share.streamlit.io/) with the GitHub account that can access this repository.
2. Select **Create app** and choose `fesbokdot11-netizen/wine-quality-prediction`, branch `main`, and `app.py`.
3. Deploy the app. The trained model and dataset are included in the repository, so no secrets or additional configuration are required.

## Limitations
This is a classroom prototype based on a public red-wine dataset. It is intended for preliminary screening only and should not replace formal laboratory, sensory, or food-safety testing. The model may not generalize well to other producers, regions, or grape varieties.
