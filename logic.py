import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score


def load_data():

    data = load_breast_cancer()

    X = pd.DataFrame(
        data.data,
        columns=data.feature_names
    )

    y = data.target

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )


def train_models():

    X_train, X_test, y_train, y_test = load_data()

    models = {
        "Random Forest": RandomForestClassifier(),
        "Logistic Regression": LogisticRegression(max_iter=5000),
        "SVM": SVC()
    }

    results = {}

    for name, model in models.items():

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        results[name] = {
            "model": model,
            "accuracy": accuracy
        }

    best_name = max(
        results,
        key=lambda x: results[x]["accuracy"]
    )

    best_model = results[best_name]["model"]

    return models, results, best_model


def compare_models(results):

    data = []

    for name, info in results.items():

        data.append({
            "Model": name,
            "Accuracy": round(
                info["accuracy"],
                4
            )
        })

    return pd.DataFrame(data)


def plot_comparison(df):

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.bar(
        df["Model"],
        df["Accuracy"]
    )

    ax.set_title("Accuracy Comparison")

    return fig


def make_prediction(
    model,
    age,
    cholesterol,
    max_hr,
    oldpeak
):

    import numpy as np

    sample = np.zeros((1, 30))

    sample[0][0] = age
    sample[0][1] = cholesterol
    sample[0][2] = max_hr
    sample[0][3] = oldpeak

    prediction = model.predict(sample)

    return prediction[0]
