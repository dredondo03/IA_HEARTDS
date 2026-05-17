import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


def preprocess_data(df, target_column):

    df = df.copy()

    # ELIMINAR NULOS
    df = df.dropna()

    # ENCODING AUTOMÁTICO
    encoders = {}

    for col in df.columns:

        if df[col].dtype == "object":

            encoder = LabelEncoder()

            df[col] = encoder.fit_transform(
                df[col]
            )

            encoders[col] = encoder

    X = df.drop(columns=[target_column])

    y = df[target_column]

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )


def train_models(df, target_column):

    X_train, X_test, y_train, y_test = preprocess_data(
        df,
        target_column
    )

    models = {
        "Random Forest": RandomForestClassifier(),
        "Logistic Regression": LogisticRegression(max_iter=5000),
        "SVM": SVC()
    }

    results = []

    for name, model in models.items():

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        precision = precision_score(
            y_test,
            predictions,
            average='weighted'
        )

        recall = recall_score(
            y_test,
            predictions,
            average='weighted'
        )

        f1 = f1_score(
            y_test,
            predictions,
            average='weighted'
        )

        results.append({
            "Model": name,
            "Accuracy": round(accuracy, 4),
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "F1-Score": round(f1, 4)
        })

    return results


def compare_models(results):

    df = pd.DataFrame(results)

    df = df.sort_values(
        by="Accuracy",
        ascending=False
    )

    return df


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
