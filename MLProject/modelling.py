import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import mlflow

DATA_PATH = "heart_failure_preprocessing/heart_failure_clean.csv"
print(f"Loading data from {DATA_PATH}...")
original_df = pd.read_csv(DATA_PATH)

X_features_df = original_df.drop("HeartDisease", axis=1)
y_target_series = original_df["HeartDisease"]

X = X_features_df.astype(float)
y = y_target_series.astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mlflow.autolog()

print("Starting training...")
with mlflow.start_run():
    clf = RandomForestClassifier(random_state=42)

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

print("Training complete. Check MLflow UI for details.")
