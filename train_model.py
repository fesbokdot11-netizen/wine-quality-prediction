import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"


def main():
    df = pd.read_csv(DATA_URL, sep=";")
    df.to_csv("wine_quality.csv", index=False)

    print("Rows and columns:", df.shape)
    print(df.head())
    print(df.dtypes)
    print("Missing values per column:")
    print(df.isnull().sum())
    print("Duplicate rows:", df.duplicated().sum())
    print("Quality distribution:")
    print(df["quality"].value_counts().sort_index())

    df = df.drop_duplicates().copy()
    y = (df["quality"] >= 6).astype(int)
    X = df.drop(columns=["quality"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

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
        y_test,
        predictions,
        target_names=["Needs improvement", "Good quality"],
        zero_division=0
    ))

    joblib.dump(model, "wine_model.pkl")
    print("Saved wine_model.pkl")


if __name__ == "__main__":
    main()
