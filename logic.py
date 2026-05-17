```python
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


def preprocess_data(df, target_column):

    df = df.copy()

    # LIMPIAR COLUMNAS
    df.columns = df.columns.str.strip()

    # ELIMINAR NULOS
    df = df.dropna()

    encoders = {}

    categorical_values = {}

    # CONVERTIR NUMÉRICOS CUANDO SEA POSIBLE
    for col in df.columns:

        df[col] = pd.to_numeric(
            df[col],
            errors="ignore"
        )

    # ENCODING COLUMNAS CATEGÓRICAS
    for col in df.columns:

        if df[col].dtype == "object":

            categorical_values[col] = list(
                df[col].unique()
            )

            encoder = LabelEncoder()

            df[col] = encoder.fit_transform(
                df[col].astype(str)
            )

            encoders[col] = encoder

    # FEATURES Y TARGET
    X = df.drop(columns=[target_column])

    y = df[target_column]

    # FORZAR FLOAT
    X = X.astype(float)

    feature_columns = X.columns.tolist()

    # ESCALADO
    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    X_scaled = pd.DataFrame(
        X_scaled,
        columns=feature_columns
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=0.2,
        random_state=42
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        encoders,
        scaler,
        feature_columns,
        categorical_values
    )


def train_models(df, target_column):

    (
        X_train,
        X_test,
        y_train,
        y_test,
        encoders,
        scaler,
        feature_columns,
        categorical_values
    ) = preprocess_data(df, target_column)

    models = {
        "Random Forest": RandomForestClassifier(),
        "Logistic Regression": LogisticRegression(max_iter=5000),
        "SVM": SVC(probability=True)
    }

    results = []

    best_accuracy = 0
    best_model = None

    for name, model in models.items():

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        precision = precision_score(
            y_test,
            predictions
        )

        recall = recall_score(
            y_test,
            predictions
        )

        f1 = f1_score(
            y_test,
            predictions
        )

        results.append({
            "Model": name,
            "Accuracy": round(accuracy, 4),
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "F1-Score": round(f1, 4)
        })

        if accuracy > best_accuracy:

            best_accuracy = accuracy
            best_model = model

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="Accuracy",
        ascending=False
    )

    bundle = {
        "model": best_model,
        "encoders": encoders,
        "scaler": scaler,
        "feature_columns": feature_columns,
        "categorical_values": categorical_values,
        "results_df": results_df
    }

    return bundle


def predict_patient(bundle, input_data):

    model = bundle["model"]

    encoders = bundle["encoders"]

    scaler = bundle["scaler"]

    feature_columns = bundle["feature_columns"]

    df = pd.DataFrame([input_data])

    # ENCODING
    for col, encoder in encoders.items():

        df[col] = encoder.transform(
            df[col].astype(str)
        )

    # ASEGURAR ORDEN
    df = df[feature_columns]

    # FLOAT
    df = df.astype(float)

    # ESCALAR
    scaled = scaler.transform(df)

    prediction = model.predict(scaled)[0]

    probability = model.predict_proba(
        scaled
    )[0][1]

    return prediction, probability


def plot_comparison(df):

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.bar(
        df["Model"],
        df["Accuracy"]
    )

    ax.set_title(
        "Model Accuracy Comparison"
    )

    ax.set_ylabel("Accuracy")

    return fig
```
