import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

def train_and_evaluate():
    # 1. Resolve paths dynamically
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "..", "MP_1_EDA", "data", "preprocessed_titanic.csv")
    model_path = os.path.join(script_dir, "model.pkl")

    print(f"Loading preprocessed dataset from: {data_path}")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}. Ensure Mini Project 1 is completed and dataset exists.")

    df = pd.read_csv(data_path)

    # 2. Define Features (X) and Target (y)
    # Dropping metadata/identifiers that do not help in classification
    drop_cols = ["PassengerId", "Name", "Ticket", "Survived"]
    X = df.drop(columns=drop_cols)
    y = df["Survived"]

    print(f"Dataset Loaded. Features shape: {X.shape}, Target shape: {y.shape}")

    # 3. Identify feature types
    numeric_features = ["Age", "Fare", "SibSp", "Parch"]
    categorical_features = ["Pclass", "Sex", "Embarked", "HasCabin"]

    print(f"Numeric features: {numeric_features}")
    print(f"Categorical features: {categorical_features}")

    # 4. Define Preprocessor (ColumnTransformer)
    # Standard scale numeric columns and One-Hot encode categorical columns
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), categorical_features)
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Data Split complete. Train size: {len(X_train)}, Test size: {len(X_test)}")

    pipeline_lr = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(random_state=42, max_iter=1000))
    ])

    pipeline_rf = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(random_state=42, n_estimators=100, max_depth=8))
    ])

    models = {
        "Logistic Regression": pipeline_lr,
        "Random Forest": pipeline_rf
    }

    best_model_name = None
    best_f1 = -1.0
    best_pipeline = None
    results = {}

    # 7. Train and Evaluate each model
    for name, pipeline in models.items():
        print(f"\n--- Training {name} ---")
        pipeline.fit(X_train, y_train)
        
        # Predict on test set
        y_pred = pipeline.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        results[name] = {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-score": f1,
            "Confusion Matrix": cm
        }

        print(f"Results for {name}:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-score:  {f1:.4f}")
        print("  Confusion Matrix:")
        print(f"    TN: {cm[0, 0]}  FP: {cm[0, 1]}")
        print(f"    FN: {cm[1, 0]}  TP: {cm[1, 1]}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        # Select the best model based on F1-score
        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name
            best_pipeline = pipeline

    # 8. Save the best model
    print(f"\nSelecting best model: {best_model_name} (F1-score: {best_f1:.4f})")
    print(f"Saving best pipeline to {model_path}...")
    joblib.dump(best_pipeline, model_path)
    print("Model serialized and saved successfully!")

if __name__ == "__main__":
    train_and_evaluate()
