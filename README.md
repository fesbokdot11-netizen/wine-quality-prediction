# Wine Quality Prediction System

## Data Mining Laboratory Exercise

### Wine Quality Prediction | Section B | Python • GitHub • Streamlit • Predictive Modeling • SAD

Duration: 2 hours and 30 minutes (150 minutes)
Mode: Individual or pairs (instructor's choice)
Tools: Python, VS Code or Google Colab, GitHub, Streamlit
Dataset: UCI Wine Quality Dataset — Red Wine CSV
Final output: A working local Streamlit prediction app, GitHub repository, and brief SAD artifacts

## 1. Learning outcomes

At the end of this laboratory, students should be able to:

- Retrieve and inspect a real-world dataset from an online repository.
- Clean and prepare numerical data and define a binary classification target.
- Train and evaluate a supervised classification model using appropriate metrics.
- Build a Streamlit interface that accepts wine measurements and displays predictions.
- Use GitHub to publish source code and document a small information-system prototype.
- Relate the prototype to the basic SAD stages: problem, requirements, design, implementation, and testing.

## 2. Scenario and task

A small wine-production business wants to develop a decision-support prototype that can classify a wine sample as GOOD QUALITY or NEEDS IMPROVEMENT based on its physicochemical measurements. The business intends to use the prototype as an initial screening aid when reviewing batches for further quality assessment.

Your task is to build a prototype called Wine Quality Prediction System. The system will use historical red-wine measurements from the UCI Machine Learning Repository to estimate the quality category of a sample.

Prediction rule: quality >= 6 = GOOD QUALITY; quality < 6 = NEEDS IMPROVEMENT. The original quality score is the target only and must never be included as an input feature.

Responsible-use note: This is a classroom prototype based on a public dataset. A prediction is not a substitute for laboratory testing, sensory evaluation, food-safety procedures, or professional quality-control decisions.

## 3. SAD mini-activity

Before coding, complete this short analysis and design exercise (10 minutes).

| SAD element | Student response / expected content |
|---|---|
| Problem statement | A producer needs a consistent preliminary way to screen wine samples using recorded physicochemical measurements. |
| Users | Quality-control staff, production supervisor, or data analyst. |
| Inputs | Fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free/total sulfur dioxide, density, pH, sulphates, and alcohol. |
| Process | Validate input columns → preprocess data → apply trained model → classify quality category. |
| Outputs | GOOD QUALITY / NEEDS IMPROVEMENT label and model evaluation summary. |
| Key limitation | The model is trained on a particular red-wine dataset and may not generalize to other wine types, producers, regions, or testing conditions. |

Mini use-case: The user enters sample measurements or uploads a CSV file → the system checks required columns → the model predicts the quality category → the system displays and exports the results.

## 4. Time plan (150 minutes)

| Time | Activity | Expected output |
|---|---|---|
| 0–10 min | Scenario, target definition, and SAD mini-activity | Problem, user, inputs/process/outputs |
| 10–25 min | Set up environment and retrieve online dataset | wine_quality.csv; inspect first rows |
| 25–50 min | Preprocess data and train the model | Saved wine_model.pkl |
| 50–65 min | Evaluate model and discuss metrics | Accuracy and classification report |
| 65–105 min | Build and run Streamlit application | Working local app |
| 105–125 min | Create GitHub repository and push files | Repository URL and commits |
| 125–145 min | Test application and complete reflection | Test evidence and answers |
| 145–150 min | Submit outputs | Final submission checklist |

## 5. Requirements and setup

Use Python 3.10 or later. In the VS Code terminal or a Google Colab cell, install the packages:

```bash
pip install pandas scikit-learn streamlit joblib
```

Create these files in one project folder:

```text
wine-quality-prediction/
├── train_model.py
├── app.py
├── requirements.txt
└── README.md
```

For this compressed lab, students use one classifier (Random Forest) and a CSV prediction-upload flow. A second model, confusion-matrix visualization, feature-importance analysis, and public deployment are optional extensions.

## 6. Activity A — Retrieve and inspect the dataset (15 minutes)

Dataset source: UCI Machine Learning Repository — Wine Quality. Open:
https://archive.ics.uci.edu/dataset/186/wine+quality

For this activity, use the red-wine CSV file. The code downloads the CSV directly from the UCI repository and saves it locally.

```python
# train_model.py — Activity A
import pandas as pd

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
df = pd.read_csv(url, sep=";")
df.to_csv("wine_quality.csv", index=False)
print("Rows and columns:", df.shape)
print(df.head())
print(df.dtypes)
print("Missing values per column:")
print(df.isnull().sum())
print("Duplicate rows:", df.duplicated().sum())
print("Quality distribution:")
print(df["quality"].value_counts().sort_index())
```

Checkpoint: Confirm that `wine_quality.csv` is created, that it contains the physicochemical attributes, and that the target column is named `quality`.

## 7. Activity B — Preprocess, train, and evaluate (25 minutes)

Continue in `train_model.py`. The target is derived from `quality`. The original quality column is removed from the input features to avoid target leakage. The pipeline imputes missing numeric values, scales the measurements, and trains a Random Forest classifier.

```python
# Activity B — append below Activity A
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)
import joblib

# Remove duplicate records and define the binary target
df = df.drop_duplicates().copy()
y = (df["quality"] >= 6).astype(int)  # 1 = Good; 0 = Needs improvement

# Do not include the original quality score as a predictor
X = df.drop(columns=["quality"])

# Split the data while preserving class proportions
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Preprocessing and classification in one pipeline
model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("classifier", RandomForestClassifier(
        n_estimators=150,
        random_state=42,
        class_weight="balanced"
    ))
])

model.fit(X_train, y_train)
predictions = model.predict(X_test)
print("Accuracy:", round(accuracy_score(y_test, predictions), 3))
print("Confusion matrix:")
print(confusion_matrix(y_test, predictions))
print(classification_report(
    y_test, predictions,
    target_names=["Needs improvement", "Good quality"],
    zero_division=0
))
joblib.dump(model, "wine_model.pkl")
print("Saved wine_model.pkl")
```

Run the complete script once:

```bash
python train_model.py
```

Checkpoint: `wine_model.pkl` should be created. Record accuracy, precision, recall, and F1-score for both classes. Do not interpret accuracy alone. If using Google Colab, run the code cells in sequence and download the generated files when needed.

## 8. Activity C — Build the Streamlit app (40 minutes)

Create `app.py`. The application displays basic dataset information, accepts a CSV upload, checks required columns, predicts the quality category, and lets the user download the results.

```python
# app.py
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Wine Quality Predictor",
    page_icon="🍷",
    layout="wide"
)

@st.cache_resource
def load_model():
    return joblib.load("wine_model.pkl")

model = load_model()
data = pd.read_csv("wine_quality.csv")

st.title("🍷 Wine Quality Prediction System")
st.caption("Section B classroom prototype for preliminary quality classification")

c1, c2, c3 = st.columns(3)
c1.metric("Records", len(data))
c2.metric("Average quality score", round(data["quality"].mean(), 2))
c3.metric("Good-quality records", int((data["quality"] >= 6).sum()))

st.subheader("Dataset preview")
st.dataframe(data.head(10), use_container_width=True)

st.subheader("Predict uploaded wine samples")
st.write("Upload a CSV containing the same predictor columns as the training data.")

template = data.drop(columns=["quality"]).head(5)
st.download_button(
    "Download sample input CSV",
    template.to_csv(index=False),
    "sample_wine_input.csv",
    "text/csv"
)

uploaded = st.file_uploader("Choose wine sample CSV", type=["csv"])
if uploaded is not None:
    new_data = pd.read_csv(uploaded)
    required_columns = list(model.feature_names_in_)
    missing = [c for c in required_columns if c not in new_data.columns]

    if missing:
        st.error("Missing required columns: " + ", ".join(missing))
    else:
        predictions = model.predict(new_data[required_columns])
        results = new_data.copy()
        results["Prediction"] = [
            "GOOD QUALITY" if value == 1 else "NEEDS IMPROVEMENT"
            for value in predictions
        ]

        st.subheader("Prediction results")
st.dataframe(results, use_container_width=True)
        st.metric(
            "Predicted good-quality samples",
            int((predictions == 1).sum())
        )
        st.download_button(
            "Download prediction results",
            results.to_csv(index=False),
            "wine_quality_predictions.csv",
            "text/csv"
        )
```

Create `requirements.txt`:

```text
pandas
scikit-learn
streamlit
joblib
```

Run the app in the project folder:

```bash
streamlit run app.py
```

Open the local address shown in the terminal (normally http://localhost:8501). Download `sample_wine_input.csv` from the app and upload it to test predictions.

## 9. Activity D — Publish the project to GitHub (20 minutes)

Create a public GitHub repository named `wine-quality-prediction`. Add `train_model.py`, `app.py`, `requirements.txt`, and `README.md`. Include the generated CSV and model files only if permitted by the instructor.

```bash
git init
git add train_model.py app.py requirements.txt README.md
git commit -m "Build wine quality prediction prototype"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/wine-quality-prediction.git
git push -u origin main
```

Replace `YOUR-USERNAME` with your GitHub username. Use the approved GitHub sign-in method; do not place passwords or access tokens in source code.

`README.md` must briefly state the problem, dataset source, prediction rule, technologies, installation/run steps, and limitations.

## 10. Testing and reflection (20 minutes)

| Test | Expected result | Pass? |
|---|---|---|
| Run `train_model.py` | CSV and model files are created without errors. | ☐ |
| Run Streamlit app | Dashboard loads in browser. | ☐ |
| Upload sample CSV | Predictions are displayed. | ☐ |
| Upload CSV missing a required column | The app displays a helpful error message. | ☐ |
| Download results | A CSV containing predictions is downloaded. | ☐ |
| Open GitHub repository | Source files and README are visible. | ☐ |

Answer briefly:

1. What is the prediction target, and why is the original quality column excluded from the input features?
2. What does recall for the GOOD QUALITY class tell you? What could low recall mean for the business?
3. Why might a model trained on this red-wine dataset perform differently on another producer's wine samples?
4. Why should the prediction be treated as a preliminary screening result rather than a final quality decision?
5. What additional data or testing would help improve the reliability of the system?

### Project reflection

This project demonstrated how a data-mining workflow can be turned into a small, usable information-system prototype. We retrieved and inspected the UCI red-wine dataset, removed duplicate rows for model training, and transformed the quality score into two categories: GOOD QUALITY for scores of 6 or higher and NEEDS IMPROVEMENT for scores below 6. The original quality score was excluded from the model inputs because it defines the target; including it would cause target leakage and make the evaluation misleading.

The Random Forest pipeline reports accuracy, a confusion matrix, and precision, recall, and F1-score for both classes. Looking at recall for GOOD QUALITY helps show how many truly good-quality examples are identified. Low recall could result in acceptable samples being flagged for improvement, while other classification errors could also affect screening decisions. For this reason, accuracy alone is not enough to judge whether the model is useful.

The Streamlit app connects the trained model to a CSV workflow: users can download an input template, upload measurements, review predictions, and download the results. Separating this workflow from the overview dashboard makes the two main tasks easier to find. Building and testing the app also highlighted that uploaded data needs the required measurement columns and that clear error messages are important when a file is incomplete or unreadable.

The model is limited by the dataset it learned from. Differences in producers, regions, grape varieties, equipment, and measurement practices may affect its performance on new samples. The predictions should therefore be treated only as preliminary screening results, not as a substitute for laboratory tests, sensory evaluation, food-safety procedures, or professional decisions. To improve confidence, the system should be tested on independent and more diverse samples, and its performance and error trade-offs should be reviewed with quality-control professionals.

## 11. Submission checklist

- ☐ SAD mini-analysis (problem, user, inputs, process, outputs, limitation)
- ☐ `train_model.py` and `app.py`
- ☐ `requirements.txt` and `README.md`
- ☐ Model evaluation results (accuracy, precision, recall, F1-score)
- ☐ Screenshot of the Streamlit dashboard and prediction results
- ☐ GitHub repository URL
- ☐ Brief answers to the reflection questions

## 12. Suggested grading rubric

| Criteria | Weight |
|---|---:|
| SAD mini-analysis and understanding | 10% |
| Dataset retrieval and preprocessing | 15% |
| Model training and evaluation | 25% |
| Streamlit application and functional testing | 30% |
| GitHub repository and README | 10% |
| Reflection and responsible interpretation | 10% |
| TOTAL | 100% |

## 13. Optional extension (outside the 150-minute limit)

- Compare Random Forest with Logistic Regression or Decision Tree using the same train/test split.
- Add a confusion matrix visualization and feature-importance discussion.
- Use cross-validation to estimate model performance across multiple data splits.
- Deploy the app using Streamlit Community Cloud.
- Experiment with a different threshold for the GOOD QUALITY class and explain its effect on precision and recall.

## References and links

- UCI Wine Quality Dataset: https://archive.ics.uci.edu/dataset/186/wine+quality
- Scikit-learn documentation: https://scikit-learn.org/stable/
- Streamlit documentation: https://docs.streamlit.io/
- GitHub getting started: https://docs.github.com/en/get-started

END OF LABORATORY EXERCISE

## Run or deploy the app

The app loads `wine_model.pkl` and `wine_quality.csv` at startup, so both files
must be present alongside `app.py` in the repository.

Run locally:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Deploy with Streamlit Community Cloud:

1. Sign in at https://share.streamlit.io/ using the GitHub account that owns this repository.
2. Select **Create app**, then choose the `wine-quality-prediction` repository and its `main` branch.
3. Set the app file path to `app.py` and select **Deploy**.

The deployed app is a classroom prototype. Do not use its predictions as a
substitute for laboratory testing or professional quality-control decisions.
