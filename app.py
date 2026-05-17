import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

import logic

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Heart Disease AI",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');
html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: #0d1117; color: #e6edf3;
}
h1,h2,h3 { font-family:'IBM Plex Mono',monospace; color:#f0483e; }
div[data-testid="stSidebar"] { background:#161b22; border-right:1px solid #30363d; }
.stButton>button {
    background:#f0483e; color:white; border:none;
    border-radius:6px; font-family:'IBM Plex Mono',monospace;
    font-weight:600; padding:10px 24px;
}
.stButton>button:hover { background:#c73830; }
.block-container { padding-top:2rem; }
.metric-box {
    background:#161b22; border:1px solid #30363d;
    border-radius:10px; padding:16px; text-align:center;
}
.metric-value { font-size:2rem; font-weight:700; color:#f0483e; }
.metric-label { font-size:.85rem; color:#8b949e; margin-top:4px; }
.best-badge {
    background:#1a4731; color:#3fb950;
    padding:6px 16px; border-radius:20px;
    font-family:'IBM Plex Mono',monospace; font-size:.9rem;
    display:inline-block; margin-bottom:12px;
}
.warn-badge {
    background:#4a2c1a; color:#e3b341;
    padding:6px 16px; border-radius:20px;
    font-family:'IBM Plex Mono',monospace; font-size:.9rem;
    display:inline-block;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# MATPLOTLIB DARK THEME
# ─────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":"#161b22","axes.facecolor":"#161b22",
    "axes.edgecolor":"#30363d","axes.labelcolor":"#e6edf3",
    "xtick.color":"#8b949e","ytick.color":"#8b949e",
    "text.color":"#e6edf3","grid.color":"#21262d","grid.linestyle":"--",
    "legend.facecolor":"#161b22","legend.edgecolor":"#30363d",
})
RED    = "#f0483e"
BLUE   = "#58a6ff"
GREEN  = "#3fb950"
YELLOW = "#e3b341"
PURPLE = "#a371f7"
PALETTE = [RED, BLUE, GREEN, YELLOW, PURPLE]

# ─────────────────────────────────────────────
# LOAD BUNDLE (cached — only once per session)
# ─────────────────────────────────────────────
@st.cache_resource
def load_model_bundle():
    return logic.load_bundle("heart_model_bundle.pkl")

try:
    bundle = load_model_bundle()
    BUNDLE_OK = True
except FileNotFoundError:
    BUNDLE_OK = False

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("# 🫀 HeartAI")
    if BUNDLE_OK:
        st.markdown(
            f"<div class='best-badge'>✅ Model loaded:<br/>"
            f"<b>{bundle['model_name']}</b></div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown("<div class='warn-badge'>⚠️ No model bundle found</div>",
                    unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio(
        "Navigation",
        ["🏠 Overview", "📊 EDA & Dataset",
         "📈 Model Evaluation", "🔍 Predict Patient"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption("UNAB · Ingeniería de Sistemas\nArtificial Intelligence 1\nPh.D. Andrés Felipe Jerez Ariza")

# Guard
if not BUNDLE_OK:
    st.error(
        "**`heart_model_bundle.pkl` not found.**\n\n"
        "Run `heart_disease_training.ipynb` in Google Colab, download the "
        "generated `.pkl` file, and place it in the same folder as `app.py`."
    )
    st.stop()

# ═══════════════════════════════════════════════════════════════════════
#  PAGE 1 – OVERVIEW
# ═══════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.title("🫀 Heart Disease Prediction")
    st.markdown(
        "**Supervised learning pipeline** trained on the "
        "[Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction) "
        "(Kaggle). The best-performing model is pre-loaded — no training happens here."
    )

    summary = logic.get_dataset_summary(bundle)
    best_m  = logic.get_best_metrics(bundle)

    st.markdown("### Dataset Split")
    total = summary["train_samples"] + summary["test_samples"]
    c1, c2, c3, c4 = st.columns(4)
    for col, label, val in zip(
        [c1, c2, c3, c4],
        ["Total Samples","Train Samples","Test Samples","Features"],
        [total, summary["train_samples"], summary["test_samples"], summary["n_features"]]
    ):
        col.markdown(
            f"<div class='metric-box'>"
            f"<div class='metric-value'>{val}</div>"
            f"<div class='metric-label'>{label}</div></div>",
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.markdown(f"### Best Model — `{bundle['model_name']}`")
    c1, c2, c3, c4 = st.columns(4)
    for col, label, key in zip(
        [c1, c2, c3, c4],
        ["Accuracy (Test)","F1-Score","Precision","Recall"],
        ["Accuracy Test","F1-Score","Precision","Recall"]
    ):
        col.markdown(
            f"<div class='metric-box'>"
            f"<div class='metric-value'>{best_m.get(key, best_m.get(label, 'N/A')):.1%}</div>"
            f"<div class='metric-label'>{label}</div></div>",
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.markdown("### Pipeline Summary")
    st.markdown("""
| Stage | Method |
|---|---|
| Preprocessing | Duplicate removal · Outlier fix (Cholesterol, RestingBP) |
| Encoding | LabelEncoder for categorical features |
| Scaling | StandardScaler for numerical features |
| Feature Reduction | PCA (95% variance) + LDA (1 component) |
| Models trained | Random Forest · Logistic Regression · SVM |
| Best model selected by | F1-Score + ROC-AUC |
| Deployment | Loaded from pre-trained `.pkl` via joblib |
""")


# ═══════════════════════════════════════════════════════════════════════
#  PAGE 2 – EDA & DATASET
# ═══════════════════════════════════════════════════════════════════════
elif page == "📊 EDA & Dataset":
    st.title("📊 EDA & Dataset")

    summary = logic.get_dataset_summary(bundle)

    # ── Class distribution ──
    st.markdown("### Target Distribution")
    class_dist = summary["class_dist"]
    labels = ["No Heart Disease (0)", "Heart Disease (1)"]
    vals   = [class_dist.get(0, 0), class_dist.get(1, 0)]

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    axes[0].bar(labels, vals, color=[BLUE, RED], width=0.5, edgecolor="#0d1117")
    axes[0].set_title("Class Counts")
    axes[0].set_ylabel("Count")
    for i, v in enumerate(vals):
        axes[0].text(i, v + 3, str(v), ha="center", fontweight="bold")

    axes[1].pie(vals, labels=labels, autopct="%1.1f%%",
                colors=[BLUE, RED], startangle=90,
                wedgeprops={"edgecolor": "#0d1117"})
    axes[1].set_title("Class Balance")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # ── Dataset before vs after preprocessing ──
    st.markdown("### Before vs After Preprocessing")
    st.markdown(
        "The raw dataset contained records where **Cholesterol = 0** and "
        "**RestingBP = 0** — physiologically impossible values treated as "
        "data entry errors. These were replaced with the column median "
        "(computed excluding zeros)."
    )
    before_after = pd.DataFrame({
        "Issue": ["Zero Cholesterol", "Zero RestingBP", "Duplicate rows"],
        "Before (raw)": ["Present", "Present", "Possible"],
        "After (cleaned)": ["Replaced with median", "Replaced with median", "Removed"],
    })
    st.table(before_after)

    # ── Feature list ──
    st.markdown("### Feature Overview")
    feat_df = pd.DataFrame({
        "Feature": summary["feature_cols"],
        "Type": [
            "Numerical" if f in summary["numerical"] else "Categorical"
            for f in summary["feature_cols"]
        ],
    })
    st.dataframe(feat_df, use_container_width=True, hide_index=True)

    # ── Feature importances (from RF stored in bundle) ──
    st.markdown("### Feature Importances (Random Forest)")
    fi = logic.get_feature_importances(bundle)
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.bar(fi.index, fi.values, color=BLUE, edgecolor="#0d1117")
    ax.set_ylabel("Importance")
    ax.set_title("Feature Importances — Random Forest")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown(
        "**Insight:** The features with the highest predictive power for "
        "heart disease are typically `ST_Slope`, `ExerciseAngina`, "
        "`ChestPainType`, and `MaxHR`. These align with known clinical "
        "risk indicators."
    )

    # ── Train / Test split visual ──
    st.markdown("### Train / Test Split")
    split_vals = [summary["train_samples"], summary["test_samples"]]
    fig, ax = plt.subplots(figsize=(6, 2))
    ax.barh(["Samples"], [split_vals[0]], color=BLUE, label="Train")
    ax.barh(["Samples"], [split_vals[1]], left=split_vals[0], color=RED, label="Test")
    ax.set_xlim(0, sum(split_vals))
    ax.set_xlabel("Number of samples")
    ax.legend(loc="upper right")
    for i, (v, left, color) in enumerate(zip(
        split_vals, [0, split_vals[0]], [BLUE, RED]
    )):
        ax.text(left + v/2, 0, f"{v}\n({v/sum(split_vals)*100:.0f}%)",
                ha="center", va="center", color="white", fontweight="bold")
    ax.set_title("80% Train / 20% Test (stratified)")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# ═══════════════════════════════════════════════════════════════════════
#  PAGE 3 – MODEL EVALUATION
# ═══════════════════════════════════════════════════════════════════════
elif page == "📈 Model Evaluation":
    st.title("📈 Model Evaluation")

    df_cmp = logic.get_comparison_df(bundle)
    cms    = logic.get_confusion_matrices(bundle)
    roc_data, roc_auc = logic.get_roc_data(bundle)
    best_name = bundle["model_name"]

    # ── Comparison table ──
    st.markdown("### All Models — Performance Comparison")
    def highlight_best(row):
        style = [""] * len(row)
        if row["Model"] == best_name:
            style = ["background-color:#1a4731; color:#3fb950"] * len(row)
        return style

    st.dataframe(
        df_cmp.style.apply(highlight_best, axis=1),
        use_container_width=True, hide_index=True
    )
    st.markdown(
        f"<div class='best-badge'>🏆 Best Model: <b>{best_name}</b></div>",
        unsafe_allow_html=True
    )

    # ── Bar chart ──
    st.markdown("### Metrics Bar Chart")
    metrics_keys = ["Accuracy Test", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    existing = [k for k in metrics_keys if k in df_cmp.columns]
    x = np.arange(len(df_cmp))
    bar_w = 0.15

    fig, ax = plt.subplots(figsize=(11, 4.5))
    for i, (metric, color) in enumerate(zip(existing, PALETTE)):
        bars = ax.bar(x + i * bar_w, df_cmp[metric], width=bar_w,
                      label=metric, color=color)
    ax.set_xticks(x + bar_w * (len(existing) - 1) / 2)
    ax.set_xticklabels(df_cmp["Model"])
    ax.set_ylim(0, 1.1)
    ax.set_title("Model Performance Comparison")
    ax.legend()
    ax.axhline(0.85, linestyle="--", color="#8b949e", alpha=0.5)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # ── Confusion matrices ──
    st.markdown("### Confusion Matrices")
    model_names = list(cms.keys())
    fig, axes = plt.subplots(1, len(model_names), figsize=(5 * len(model_names), 4))
    if len(model_names) == 1:
        axes = [axes]
    for ax, name in zip(axes, model_names):
        cm = cms[name]
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["No Disease", "Heart Disease"],
            yticklabels=["No Disease", "Heart Disease"],
            linewidths=0.5
        )
        title_suffix = " 🏆" if name == best_name else ""
        ax.set_title(f"{name}{title_suffix}")
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # ── ROC Curves ──
    st.markdown("### ROC Curves")
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, color in zip(model_names, PALETTE):
        fpr, tpr, _ = roc_data[name]
        auc_val = roc_auc[name]
        lw = 3 if name == best_name else 1.5
        ls = "-" if name == best_name else "--"
        ax.plot(fpr, tpr, color=color, lw=lw, ls=ls,
                label=f"{name} (AUC={auc_val:.3f})")
    ax.plot([0,1],[0,1],"k--",lw=1)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — All Models")
    ax.legend(loc="lower right")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # ── Overfitting analysis ──
    st.markdown("### Overfitting Analysis — Train vs Test Accuracy")
    all_m = logic.get_all_metrics(bundle)
    names_list  = list(all_m.keys())
    train_accs  = [all_m[n].get("Accuracy Train", 0) for n in names_list]
    test_accs   = [all_m[n].get("Accuracy Test", 0)  for n in names_list]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(names_list, train_accs, "o--", color=BLUE, lw=2, label="Train Accuracy")
    ax.plot(names_list, test_accs,  "o-",  color=RED,  lw=2, label="Test Accuracy")
    for i, (tr, te) in enumerate(zip(train_accs, test_accs)):
        ax.annotate(f"{tr:.3f}", (i, tr), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=9)
        ax.annotate(f"{te:.3f}", (i, te), textcoords="offset points",
                    xytext=(0,-15), ha="center", fontsize=9)
    ax.set_ylim(0.7, 1.05)
    ax.set_ylabel("Accuracy")
    ax.set_title("Train vs Test Accuracy — Overfitting Check")
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown(
        "A large gap between train and test accuracy indicates **overfitting**. "
        "Models with gaps > 10% are flagged. Cross-validation F1 (`CV F1 Mean`) "
        "gives a more robust estimate of generalization."
    )

    # ── Cross-validation ──
    st.markdown("### Cross-Validation F1 (5-Fold)")
    cv_means = [all_m[n].get("CV F1 Mean", 0) for n in names_list]
    cv_stds  = [all_m[n].get("CV F1 Std", 0)  for n in names_list]

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.bar(names_list, cv_means, yerr=cv_stds, capsize=6,
           color=PURPLE, edgecolor="#0d1117", error_kw={"color": "#e6edf3"})
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1-Score")
    ax.set_title("5-Fold Cross-Validation F1 ± Std")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# ═══════════════════════════════════════════════════════════════════════
#  PAGE 4 – PREDICT PATIENT
# ═══════════════════════════════════════════════════════════════════════
elif page == "🔍 Predict Patient":
    st.title("🔍 Predict Patient")
    st.markdown(
        f"Using pre-trained model: **`{bundle['model_name']}`**  \n"
        "Fill in the patient's clinical data and click **Predict**."
    )

    with st.form("patient_form"):
        c1, c2, c3 = st.columns(3)

        age     = c1.number_input("Age", 20, 100, 54)
        sex     = c2.selectbox("Sex", ["M", "F"])
        chest   = c3.selectbox("ChestPainType", ["ATA", "NAP", "ASY", "TA"])
        rbp     = c1.number_input("RestingBP (mmHg)", 60, 220, 130)
        chol    = c2.number_input("Cholesterol (mg/dl)", 50, 600, 200)
        fbs     = c3.selectbox("FastingBS > 120 mg/dl", [0, 1])
        ecg     = c1.selectbox("RestingECG", ["Normal", "ST", "LVH"])
        maxhr   = c2.number_input("MaxHR", 60, 220, 140)
        exang   = c3.selectbox("ExerciseAngina", ["Y", "N"])
        oldpeak = c1.number_input("Oldpeak", -3.0, 7.0, 1.0, step=0.1)
        slope   = c2.selectbox("ST_Slope", ["Up", "Flat", "Down"])

        submitted = st.form_submit_button("🫀 Predict")

    if submitted:
        input_dict = {
            "Age": age, "Sex": sex, "ChestPainType": chest,
            "RestingBP": rbp, "Cholesterol": chol, "FastingBS": fbs,
            "RestingECG": ecg, "MaxHR": maxhr, "ExerciseAngina": exang,
            "Oldpeak": oldpeak, "ST_Slope": slope,
        }

        try:
            prediction, prob = logic.predict_single(bundle, input_dict)
        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.stop()

        st.markdown("---")
        label = "❤️ Heart Disease Detected" if prediction == 1 else "💚 No Heart Disease Detected"
        color = RED if prediction == 1 else GREEN

        st.markdown(
            f"<h2 style='color:{color};'>{label}</h2>",
            unsafe_allow_html=True
        )

        c1, c2 = st.columns([1, 2])
        if prob is not None:
            c1.metric("Model Confidence", f"{prob*100:.1f}%")
            c1.metric("Model Used", bundle["model_name"])

            fig, ax = plt.subplots(figsize=(5, 1.2))
            ax.barh([""], [prob], color=color, height=0.4)
            ax.barh([""], [1 - prob], left=[prob], color="#21262d", height=0.4)
            ax.set_xlim(0, 1)
            ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
            ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
            ax.set_title("Prediction Confidence")
            c2.pyplot(fig)
            plt.close()

        st.markdown("---")
        st.markdown("### Clinical Interpretation")
        if prediction == 1:
            st.warning(
                "The model predicts a **high risk** of heart disease based on the "
                "provided features. This is a screening tool — always confirm with "
                "a qualified cardiologist."
            )
        else:
            st.success(
                "The model predicts **low risk** of heart disease. Regular checkups "
                "and a healthy lifestyle remain important preventive measures."
            )
