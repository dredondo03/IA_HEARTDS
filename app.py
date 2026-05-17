import streamlit as st
from logic import (
    train_models,
    make_prediction,
    compare_models,
    plot_comparison
)

st.set_page_config(
    page_title="Heart Disease Prediction",
    layout="wide"
)

st.title("❤️ Heart Disease Prediction")

st.markdown("Demo comparativa de modelos de Machine Learning")

# ENTRENAR MODELOS
models, results, best_model = train_models()

# SIDEBAR
st.sidebar.header("Patient Data")

age = st.sidebar.slider("Age", 20, 100, 50)
cholesterol = st.sidebar.slider("Cholesterol", 50, 600, 200)
max_hr = st.sidebar.slider("Max Heart Rate", 60, 220, 150)
oldpeak = st.sidebar.slider("Oldpeak", 0.0, 6.0, 1.0)

# PREDICCIÓN
if st.sidebar.button("Predict"):

    prediction = make_prediction(
        best_model,
        age,
        cholesterol,
        max_hr,
        oldpeak
    )

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("High Risk of Heart Disease")
    else:
        st.success("Low Risk of Heart Disease")

st.divider()

# TABLA COMPARATIVA
st.subheader("Model Comparison")

comparison_df = compare_models(results)

st.dataframe(comparison_df)

st.divider()

# GRÁFICA
st.subheader("Performance Comparison")

fig = plot_comparison(comparison_df)

st.pyplot(fig)
