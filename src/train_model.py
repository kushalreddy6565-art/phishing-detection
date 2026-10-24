import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src.features import build_feature_dataframe

MODEL_DIR = "models"
DATASET_PATH = "data/sample_urls.csv"

def build_model():
    df = pd.read_csv(DATASET_PATH)
    feature_df = build_feature_dataframe(df, url_col="url", label_col="label")

    X = feature_df.drop(columns=["label"])
    y = feature_df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        min_samples_leaf=2
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    joblib.dump(model, os.path.join(MODEL_DIR, "phishing_model.pkl"))
    joblib.dump(list(X.columns), os.path.join(MODEL_DIR, "feature_columns.pkl"))

    print("Model saved to:", os.path.join(MODEL_DIR, "phishing_model.pkl"))
    print("Columns saved to:", os.path.join(MODEL_DIR, "feature_columns.pkl"))

if __name__ == "__main__":
    build_model()