import os

import joblib
import pandas as pd
import streamlit as st


@st.cache_resource
def load_model():
    if not os.path.exists("wine_model.pkl"):
        st.error("Model not found. Run `python train_model.py` before launching the app.")
        st.stop()
    return joblib.load("wine_model.pkl")


@st.cache_data
def load_dataset():
    if not os.path.exists("wine_quality.csv"):
        st.error("Dataset not found. Run `python train_model.py` before launching the app.")
        st.stop()
    return pd.read_csv("wine_quality.csv")


st.set_page_config(page_title="Wine Quality Predictor", page_icon="🍷", layout="wide")

st.markdown(
    """
    <style>
        .stApp {
            background: linear-gradient(180deg, #fffaf2 0%, #f5efe7 100%);
        }
        .main-header {
            background: linear-gradient(135deg, #7d1d1d, #af3d27);
            padding: 1.5rem 1.5rem 1.2rem 1.5rem;
            border-radius: 1rem;
            color: white;
            box-shadow: 0 8px 24px rgba(125, 29, 29, 0.18);
            margin-bottom: 1rem;
        }
        .splash-card {
            max-width: 850px;
            margin: 8vh auto 1.5rem auto;
            padding: clamp(2rem, 6vw, 4.5rem);
            border: 1px solid rgba(125, 29, 29, 0.14);
            border-radius: 1.5rem;
            background:
                radial-gradient(ellipse at 90% 10%, rgba(176, 75, 53, 0.17), transparent 34%),
                linear-gradient(140deg, #fffdf9, #f5e8dc);
            box-shadow: 0 18px 50px rgba(75, 38, 29, 0.12);
            text-align: center;
        }
        .splash-kicker {
            color: #8a4334;
            font-size: 0.75rem;
            font-weight: 700;
            letter-spacing: 0.16em;
            text-transform: uppercase;
        }
        .splash-title {
            margin: 0.75rem 0 0.5rem;
            color: #48201f;
            font-size: clamp(2.2rem, 6vw, 3.8rem);
            line-height: 1.08;
        }
        .splash-copy {
            max-width: 600px;
            margin: 0 auto;
            color: #705d56;
            font-size: 1.08rem;
            line-height: 1.7;
        }
        .splash-note {
            max-width: 620px;
            margin: 1.5rem auto 0;
            padding: 0.8rem 1rem;
            border-radius: 0.75rem;
            background: rgba(255, 255, 255, 0.7);
            color: #76534b;
            font-size: 0.85rem;
        }
        div[data-testid="stButton"] > button[kind="primary"] {
            width: 100%;
            border: 0;
            border-radius: 0.75rem;
            background: #7d1d1d;
            color: white;
            font-weight: 700;
            padding: 0.65rem 1.5rem;
        }
        div[data-testid="stButton"] > button[kind="primary"]:hover {
            background: #5e171b;
            border: 0;
            color: white;
        }
        .glass-panel {
            background: rgba(255,255,255,0.75);
            border: 1px solid rgba(123, 74, 55, 0.12);
            border-radius: 1rem;
            padding: 1rem;
            box-shadow: 0 4px 14px rgba(52, 26, 19, 0.08);
        }
        .section-label {
            color: #7d1d1d;
            font-size: 0.8rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }
        .quality-badge {
            display: inline-block;
            padding: 0.4rem 0.8rem;
            border-radius: 999px;
            font-weight: 700;
            background: #f5e4d9;
            color: #7d1d1d;
        }
        .sidebar-card {
            padding: 1rem;
            border: 1px solid rgba(125, 29, 29, 0.14);
            border-radius: 0.9rem;
            background: linear-gradient(160deg, #fffaf5, #f8eee8);
        }
        .sidebar-eyebrow {
            color: #8a4334;
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.1em;
            text-transform: uppercase;
        }
        .sidebar-rule {
            margin: 0.8rem 0;
            padding: 0.75rem;
            border-radius: 0.65rem;
            background: #fff;
            border-left: 3px solid #8a2432;
        }
        .sidebar-rule strong {
            display: block;
            color: #70202a;
            font-size: 0.9rem;
        }
        .sidebar-rule span {
            color: #65504a;
            font-size: 0.8rem;
        }
        div[data-testid="stMetricValue"] {
            font-size: 1.8rem;
            font-weight: 700;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

if "show_dashboard" not in st.session_state:
    st.session_state.show_dashboard = False

if not st.session_state.show_dashboard:
    st.markdown(
        """
        <div class="splash-card">
            <div style="font-size:3rem; margin-bottom:0.5rem;">🍷</div>
            <div class="splash-kicker">Red wine · Quality screening</div>
            <h1 class="splash-title">A clearer look at wine quality</h1>
            <p class="splash-copy">
                Review physicochemical measurements and get a preliminary quality
                classification for your wine samples.
            </p>
            <div class="splash-note">
                A decision-support prototype — not a replacement for laboratory,
                sensory, or food-safety testing.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    _, button_column, _ = st.columns([1, 1, 1])
    with button_column:
        if st.button("Open", type="primary", use_container_width=True):
            st.session_state.show_dashboard = True
            st.rerun()
    st.stop()

model = load_model()
data = load_dataset()

st.markdown(
    """
    <div class="main-header">
        <h1 style="margin:0; font-size:2.5rem;">🍷 Wine Quality Predictor</h1>
        <p style="margin:0.5rem 0 0 0; font-size:1rem; opacity:0.9;">Preliminary screening for red wine batches based on physicochemical composition</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown(
    """
    <div class="sidebar-card">
        <div class="sidebar-eyebrow">🍷 How it works</div>
        <p style="margin:0.35rem 0 0.2rem; font-size:1.15rem; font-weight:700; color:#39231f;">
            Wine quality check
        </p>
        <p style="margin:0; color:#76645e; font-size:0.83rem;">
            A Random Forest model reviews the wine’s chemical measurements.
        </p>
        <div class="sidebar-rule">
            <strong>Quality score ≥ 6</strong>
            <span>GOOD QUALITY</span>
        </div>
        <div class="sidebar-rule">
            <strong>Quality score &lt; 6</strong>
            <span>NEEDS IMPROVEMENT</span>
        </div>
        <p style="margin:0.65rem 0 0; color:#76645e; font-size:0.78rem;">
            Preliminary screening only — confirm results with professional testing.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

col1, col2, col3 = st.columns(3)
col1.metric("Total records", len(data))
col2.metric("Average quality", round(data["quality"].mean(), 2))
col3.metric("Good-quality batches", int((data["quality"] >= 6).sum()))

st.markdown('<div class="section-label">Dataset overview</div>', unsafe_allow_html=True)
with st.container():
    st.dataframe(data.head(10), use_container_width=True)

st.markdown('<div class="section-label">Prediction upload</div>', unsafe_allow_html=True)
st.write("Upload a CSV with the same predictor columns used during training.")

sample_template = data.drop(columns=["quality"]).head(5)
st.download_button(
    label="Download sample input CSV",
    data=sample_template.to_csv(index=False),
    file_name="sample_wine_input.csv",
    mime="text/csv",
)

uploaded = st.file_uploader("Choose wine sample CSV", type=["csv"])

if uploaded is not None:
    new_data = pd.read_csv(uploaded)
    required_columns = list(model.feature_names_in_)
    missing = [column for column in required_columns if column not in new_data.columns]

    if missing:
        st.error("Missing required columns: " + ", ".join(missing))
    else:
        predictions = model.predict(new_data[required_columns])
        results = new_data.copy()
        results["Prediction"] = [
            "GOOD QUALITY" if value == 1 else "NEEDS IMPROVEMENT"
            for value in predictions
        ]

        st.markdown('<div class="section-label">Prediction results</div>', unsafe_allow_html=True)
        st.dataframe(results, use_container_width=True)

        good_count = int((predictions == 1).sum())
        st.metric("Predicted good-quality samples", good_count)

        st.download_button(
            label="Download prediction results",
            data=results.to_csv(index=False),
            file_name="wine_quality_predictions.csv",
            mime="text/csv",
        )
