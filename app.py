```python
import streamlit as st
import pandas as pd

from logic import (
    train_models,
    predict_patient,
    plot_comparison
)

st.set_page_config(
    page_title="Heart Disease AI",
    layout="wide"
)

st.title("❤️ Heart Disease Prediction")

st.markdown(
    "Entrenamiento y predicción en tiempo real"
)

# SUBIR DATASET
uploaded_file = st.file_uploader(
    "Upload Heart Dataset",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")

    st.dataframe(df.head())

    # TARGET
    target_column = "HeartDisease"

    st.divider()

    if st.button("Train Models"):

        with st.spinner("Training AI models..."):

            bundle = train_models(
                df,
                target_column
            )

        st.success("Training completed!")

        st.session_state["bundle"] = bundle

# PREDICCIÓN
if "bundle" in st.session_state:

    bundle = st.session_state["bundle"]

    st.divider()

    st.subheader("Patient Prediction")

    input_data = {}

    feature_columns = bundle["feature_columns"]

    for col in feature_columns:

        if col == "Age":

            input_data[col] = st.slider(
                "Age",
                20,
                100,
                50
            )

        elif col == "Cholesterol":

            input_data[col] = st.slider(
                "Cholesterol",
                0,
                600,
                200
            )

        elif col == "MaxHR":

            input_data[col] = st.slider(
                "Max Heart Rate",
                60,
                220,
                150
            )

        elif col == "Oldpeak":

            input_data[col] = st.slider(
                "Oldpeak",
                0.0,
                6.0,
                1.0
            )

        elif col in bundle["categorical_values"]:

            input_data[col] = st.selectbox(
                col,
                bundle["categorical_values"][col]
            )

        else:

            input_data[col] = st.number_input(
                col,
                value=0
            )

    if st.button("Predict Risk"):

        prediction, probability = predict_patient(
            bundle,
            input_data
        )

        st.subheader("Prediction Result")

        if prediction == 1:

            st.error(
                f"High Risk of Heart Disease ({probability:.2%})"
            )

        else:

            st.success(
                f"Low Risk of Heart Disease ({probability:.2%})"
            )

    st.divider()

    st.subheader("Model Comparison")

    st.dataframe(bundle["results_df"])

    st.divider()

    st.subheader("Accuracy Comparison")

    fig = plot_comparison(
        bundle["results_df"]
    )

    st.pyplot(fig)
```
