import streamlit as st
import pandas as pd

from logic import (
    train_models,
    compare_models,
    plot_comparison
)

st.set_page_config(
    page_title="ML Model Trainer",
    layout="wide"
)

st.title("🧠 Machine Learning Trainer")

st.markdown(
    "Carga un dataset CSV y entrena modelos en tiempo real."
)

# SUBIR CSV
uploaded_file = st.file_uploader(
    "Upload CSV Dataset",
    type=["csv"]
)

if uploaded_file is not None:

    # LEER DATASET
    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")

    st.dataframe(df.head())

    st.divider()

    # SELECCIONAR TARGET
    target_column = st.selectbox(
        "Select Target Column",
        df.columns
    )

    if st.button("Train Models"):

        with st.spinner("Training models..."):

            results = train_models(
                df,
                target_column
            )

            comparison_df = compare_models(
                results
            )

        st.success("Training completed!")

        st.divider()

        # TABLA
        st.subheader("Model Comparison")

        st.dataframe(comparison_df)

        st.divider()

        # MEJOR MODELO
        best_model = comparison_df.iloc[0]

        st.subheader("Best Model")

        st.info(
            f"{best_model['Model']} "
            f"- Accuracy: {best_model['Accuracy']}"
        )

        st.divider()

        # GRÁFICA
        st.subheader("Performance Comparison")

        fig = plot_comparison(
            comparison_df
        )

        st.pyplot(fig)
