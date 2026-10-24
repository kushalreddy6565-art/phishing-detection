import os
import joblib
import pandas as pd
import streamlit as st

from src.features import extract_features

# ---------------- CONFIGURATION ----------------

MODEL_PATH = os.path.join("models", "phishing_model.pkl")
COLUMNS_PATH = os.path.join("models", "feature_columns.pkl")

st.set_page_config(
    page_title="Sentinel AI | Phishing Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------- PROFESSIONAL DESIGN ----------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

.stApp {
    background:
        radial-gradient(ellipse at 10% 0%, #102b55 0%, transparent 40%),
        radial-gradient(ellipse at 90% 20%, #10213f 0%, transparent 35%),
        #070d19;
    color: #e8eefc;
    font-family: 'Inter', sans-serif;
}

.block-container {
    max-width: 1150px;
    padding-top: 2.5rem;
    padding-bottom: 3rem;
}

header[data-testid="stHeader"] {
    background: transparent;
}

#MainMenu, footer {
    visibility: hidden;
}

.hero {
    padding: 28px 30px;
    border: 1px solid #243c61;
    border-radius: 22px;
    background: linear-gradient(120deg, #111f38, #101a2c 70%, #10284b);
    margin-bottom: 25px;
}

.eyebrow {
    color: #63b3ff;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
}

.hero h1 {
    color: #f4f8ff;
    font-size: clamp(30px, 4vw, 44px);
    font-weight: 800;
    line-height: 1.15;
    margin: 12px 0;
}

.hero p {
    color: #a9bad4;
    font-size: 15px;
    line-height: 1.7;
    margin-bottom: 0;
}

.section-title {
    color: #eaf2ff;
    font-size: 21px;
    font-weight: 700;
    margin: 25px 0 12px;
}

.panel {
    background: #101b2d;
    border: 1px solid #253a59;
    border-radius: 16px;
    padding: 22px;
    margin: 12px 0;
}

.metric-card {
    background: linear-gradient(145deg, #14233a, #0e192a);
    border: 1px solid #2b4264;
    border-radius: 16px;
    padding: 20px;
    min-height: 125px;
}

.metric-label {
    color: #9fb2ce;
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.metric-value {
    color: #f4f8ff;
    font-size: 27px;
    font-weight: 800;
    margin-top: 12px;
    overflow-wrap: anywhere;
}

.metric-sub {
    color: #8ba5c8;
    font-size: 12px;
    margin-top: 7px;
}

.stTextInput label {
    color: #cbd9ef !important;
    font-weight: 600 !important;
}

.stTextInput input {
    background: #0a1424 !important;
    color: #ffffff !important;
    border: 1px solid #355178 !important;
    border-radius: 11px !important;
    min-height: 48px;
}

.stTextInput input:focus {
    border-color: #48a9ff !important;
    box-shadow: 0 0 0 2px #2475c733 !important;
}

.stButton > button {
    background: linear-gradient(100deg, #1687f8, #4267f5);
    color: white;
    border: none;
    border-radius: 11px;
    min-height: 46px;
    font-weight: 700;
    width: 100%;
    transition: 0.2s;
}

.stButton > button:hover {
    background: linear-gradient(100deg, #329bff, #637eff);
    color: white;
    border: none;
}

div[data-testid="stDataFrame"] {
    border: 1px solid #263c5a;
    border-radius: 12px;
    overflow: hidden;
}

div[data-testid="stAlert"] {
    border-radius: 12px;
}

hr {
    border-color: #243650;
}

.footer {
    color: #7186a5;
    font-size: 12px;
    text-align: center;
    padding-top: 30px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- MODEL LOADING ----------------

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "Trained model not found. Run: python -m src.train_model"
        )

    if not os.path.exists(COLUMNS_PATH):
        raise FileNotFoundError(
            "Feature columns not found. Run: python -m src.train_model"
        )

    model = joblib.load(MODEL_PATH)
    columns = joblib.load(COLUMNS_PATH)

    return model, list(columns)


def get_features(url):
    """Extract features and return one dictionary."""
    result = extract_features(url)

    if isinstance(result, pd.DataFrame):
        return result.iloc[0].to_dict()

    if isinstance(result, pd.Series):
        return result.to_dict()

    if isinstance(result, dict):
        return result

    return dict(result)


def is_phishing_label(label):
    value = str(label).strip().lower()

    if value in {"1", "1.0"}:
        return True

    if value in {"0", "0.0"}:
        return False

    if "phish" in value or "malicious" in value:
        return True

    if "legitimate" in value or "benign" in value or "safe" == value:
        return False

    return None


# ---------------- HEADER ----------------

st.markdown("""
<div class="hero">
    <div class="eyebrow">🛡️ SENTINEL AI · CYBERSECURITY</div>
    <h1>Phishing URL Detector</h1>
    <p>
        Analyze suspicious URLs using machine learning.
        Explore the prediction, inspect URL features, and
        understand the signals behind the model's decision.
    </p>
</div>
""", unsafe_allow_html=True)

# ---------------- URL SCANNER ----------------

st.markdown('<div class="section-title">🔎 URL Security Scanner</div>',
            unsafe_allow_html=True)

with st.container(border=True):
    url = st.text_input(
        "Enter the URL you want to analyze",
        placeholder="https://example.com",
        key="url_input"
    )

    analyze = st.button("⚡  Analyze URL", use_container_width=True)

if analyze:
    if not url.strip():
        st.warning("Please enter a URL first.")

    else:
        try:
            model, feature_columns = load_model()

            feature_dict = get_features(url.strip())

            # Preserve the exact feature order used during training.
            X = pd.DataFrame([feature_dict])
            X = X.reindex(columns=feature_columns, fill_value=0)
            X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

            prediction = model.predict(X)[0]
            classes = list(model.classes_)

            probabilities = None
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(X)[0]

            phishing = is_phishing_label(prediction)

            if phishing is None:
                st.error(
                    "The model's class labels are not recognized. "
                    "Check the labels in your training dataset."
                )
                st.stop()

            if probabilities is not None:
                phishing_indices = [
                    i for i, label in enumerate(classes)
                    if is_phishing_label(label) is True
                ]

                if len(phishing_indices) != 1:
                    st.error(
                        "Could not identify the phishing class "
                        "in the trained model."
                    )
                    st.stop()

                phishing_probability = float(
                    probabilities[phishing_indices[0]]
                )

                predicted_index = classes.index(prediction)
                confidence = float(probabilities[predicted_index])

            else:
                phishing_probability = None
                confidence = None

            st.session_state["analysis"] = {
                "url": url.strip(),
                "features": feature_dict,
                "X": X,
                "phishing": phishing,
                "phishing_probability": phishing_probability,
                "confidence": confidence,
            }

        except Exception as error:
            st.error(f"Analysis failed: {error}")

# ---------------- RESULTS ----------------

if "analysis" in st.session_state:
    result = st.session_state["analysis"]

    st.markdown('<div class="section-title">📊 Analysis Results</div>',
                unsafe_allow_html=True)

    st.markdown(
        f'<div class="panel"><span class="metric-label">'
        f'ANALYZED URL</span><br><span style="color:#d8e8ff;'
        f'overflow-wrap:anywhere">{result["url"]}</span></div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    prediction_text = (
        "PHISHING" if result["phishing"] else "LEGITIMATE"
    )
    prediction_color = (
        "#ff647c" if result["phishing"] else "#38d9a9"
    )

    with col1:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">Prediction</div>'
            f'<div class="metric-value" style="color:{prediction_color}">'
            f'{prediction_text}</div>'
            f'<div class="metric-sub">Model classification</div>'
            f'</div>',
            unsafe_allow_html=True
        )

    with col2:
        probability = result["phishing_probability"]
        probability_text = (
            f"{probability:.2%}" if probability is not None else "N/A"
        )

        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">Phishing Probability</div>'
            f'<div class="metric-value">{probability_text}</div>'
            f'<div class="metric-sub">Estimated phishing-class probability</div>'
            f'</div>',
            unsafe_allow_html=True
        )

    with col3:
        confidence = result["confidence"]
        confidence_text = (
            f"{confidence:.2%}" if confidence is not None else "N/A"
        )

        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">Model Confidence</div>'
            f'<div class="metric-value">{confidence_text}</div>'
            f'<div class="metric-sub">Probability assigned to predicted class</div>'
            f'</div>',
            unsafe_allow_html=True
        )

    if probability is not None:
        st.progress(
            max(0.0, min(1.0, probability)),
            text="Estimated phishing probability"
        )

    # ---------------- FEATURE VALUES ----------------

    st.markdown(
        '<div class="section-title">🧩 URL Feature Analysis</div>',
        unsafe_allow_html=True
    )

    feature_table = pd.DataFrame({
        "Feature": list(result["features"].keys()),
        "Value": list(result["features"].values())
    })

    st.dataframe(
        feature_table,
        use_container_width=True,
        hide_index=True
    )

    # ---------------- GLOBAL IMPORTANCE ----------------

    st.markdown(
        '<div class="section-title">📈 Important Model Features</div>',
        unsafe_allow_html=True
    )

    try:
        model, feature_columns = load_model()

        if hasattr(model, "feature_importances_"):
            importance = pd.Series(
                model.feature_importances_,
                index=feature_columns
            ).sort_values(ascending=False).head(8)

            st.bar_chart(importance, horizontal=True)

            st.caption(
                "This chart shows overall feature importance in the model, "
                "not the direction or cause of this individual prediction."
            )
        else:
            st.info(
                "This model does not expose built-in feature importances."
            )

    except Exception as error:
        st.info(f"Feature importance is unavailable: {error}")

# ---------------- FOOTER ----------------

st.markdown("""
<div class="footer">
    SENTINEL AI · MACHINE LEARNING SECURITY ANALYSIS
</div>
""", unsafe_allow_html=True)