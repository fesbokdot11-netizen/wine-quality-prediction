import pandas as pd
import streamlit as st
import joblib

st.set_page_config(
    page_title="Wine Quality Predictor",
    page_icon="🍷",
    layout="wide",
)

st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] {
        background:
            linear-gradient(rgba(18, 8, 10, 0.62), rgba(18, 8, 10, 0.72)),
            url("/app/static/wine-background.png") center / cover fixed;
    }
    [data-testid="stHeader"] {
        background: rgba(18, 8, 10, 0.25);
    }
    [data-testid="stMain"] {
        background: transparent;
    }
    [data-testid="stMain"] p,
    [data-testid="stMain"] label,
    [data-testid="stMain"] h2,
    [data-testid="stMain"] h3,
    [data-testid="stMain"] [data-testid="stCaption"],
    [data-testid="stMain"] [data-testid="stMarkdownContainer"] {
        color: #fff8f6;
    }
    [data-testid="stMain"] h1,
    [data-testid="stMain"] h2,
    [data-testid="stMain"] h3 {
        color: #fff4f1;
    }
    [data-testid="stMain"] [data-testid="stFileUploaderDropzone"] {
        background: rgba(48, 18, 24, 0.82);
        border: 1px dashed #c88b91;
        border-radius: 14px;
    }
    [data-testid="stMain"] [data-testid="stFileUploaderDropzone"] * {
        color: #fff4f1 !important;
    }
    [data-testid="stMain"] [data-testid="stFileUploaderDropzone"] button,
    [data-testid="stMain"] [data-testid="stDownloadButton"] button {
        background: #7d3540;
        color: #fff8f6;
        border: 1px solid #ba737b;
        border-radius: 9px;
    }
    [data-testid="stMain"] [data-testid="stFileUploaderDropzone"] button:hover,
    [data-testid="stMain"] [data-testid="stDownloadButton"] button:hover {
        background: #984b56;
        border-color: #e0a4aa;
        color: #ffffff;
    }
    [data-testid="stMain"] [data-testid="stMetric"] {
        background: rgba(48, 18, 24, 0.78);
        border: 1px solid rgba(231, 183, 184, 0.35);
        border-radius: 12px;
        padding: 0.8rem 1rem;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .title-box {
        background: linear-gradient(135deg, #7b1e1e, #a83c3c);
        padding: 1.4rem 1.5rem;
        border-radius: 16px;
        color: white;
        box-shadow: 0 8px 24px rgba(123, 30, 30, 0.2);
    }
    .subtle-text {
        color: #5d5d5d;
        font-size: 1rem;
    }
    .classification-heading {
        color: #fff4f1;
        font-size: 1.05rem;
        font-weight: 650;
        margin-bottom: 0.7rem;
    }
    .classification-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 1rem;
    }
    .classification-card {
        display: flex;
        align-items: center;
        gap: 1rem;
        min-height: 88px;
        padding: 1rem 1.2rem;
        background: rgba(52, 19, 25, 0.78);
        border: 1px solid rgba(231, 183, 184, 0.42);
        border-radius: 14px;
        box-shadow: 0 6px 20px rgba(12, 4, 6, 0.22);
        backdrop-filter: blur(8px);
    }
    .classification-mark {
        display: grid;
        place-items: center;
        width: 42px;
        height: 42px;
        flex: 0 0 42px;
        border-radius: 50%;
        font-size: 1.25rem;
        font-weight: 700;
    }
    .good-mark {
        color: #d7f0df;
        background: rgba(39, 105, 73, 0.58);
    }
    .improve-mark {
        color: #ffe0dc;
        background: rgba(145, 54, 59, 0.6);
    }
    .classification-label {
        color: #fff5f2;
        font-weight: 700;
        letter-spacing: 0.025em;
    }
    .classification-threshold {
        margin-top: 0.2rem;
        color: #e4c9c8;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    return joblib.load("wine_model.pkl")


@st.cache_data
def load_dataset():
    return pd.read_csv("wine_quality.csv")


model = load_model()
data = load_dataset()

st.markdown(
    """
    <div class="title-box">
        <h1 style='margin:0; font-size:2.1rem;'>🍷 Wine Quality Prediction System</h1>
        <p style='margin:0.5rem 0 0 0; font-size:1rem;'>Decision-support prototype for preliminary batch screening</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")
st.caption("")

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(
        """
        <div style='border:1px solid rgba(255,255,255,0.28); border-radius:12px; padding:18px; background:rgba(35,12,16,0.72); min-height:120px; display:flex; flex-direction:column; justify-content:center; align-items:center; text-align:center; backdrop-filter:blur(8px);'>
            <div style='font-size:0.8rem; color:#f1dadd; text-transform:uppercase; letter-spacing:0.08em;'>Total samples</div>
            <div style='font-size:2rem; font-weight:700; color:#ffffff; margin-top:8px;'>1599</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col2:
    st.markdown(
        """
        <div style='border:1px solid rgba(255,255,255,0.28); border-radius:12px; padding:18px; background:rgba(35,12,16,0.72); min-height:120px; display:flex; flex-direction:column; justify-content:center; align-items:center; text-align:center; backdrop-filter:blur(8px);'>
            <div style='font-size:0.8rem; color:#f1dadd; text-transform:uppercase; letter-spacing:0.08em;'>Average quality score</div>
            <div style='font-size:2rem; font-weight:700; color:#ffffff; margin-top:8px;'>5.64</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col3:
    st.markdown(
        """
        <div style='border:1px solid rgba(255,255,255,0.28); border-radius:12px; padding:18px; background:rgba(35,12,16,0.72); min-height:120px; display:flex; flex-direction:column; justify-content:center; align-items:center; text-align:center; backdrop-filter:blur(8px);'>
            <div style='font-size:0.8rem; color:#f1dadd; text-transform:uppercase; letter-spacing:0.08em;'>Good-quality records</div>
            <div style='font-size:2rem; font-weight:700; color:#ffffff; margin-top:8px;'>855</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")
st.markdown(
    """
    <div class="classification-heading">How quality is classified</div>
    <div class="classification-grid">
        <div class="classification-card">
            <div class="classification-mark good-mark">✓</div>
            <div>
                <div class="classification-label">GOOD QUALITY</div>
                <div class="classification-threshold">Quality score 6 or higher</div>
            </div>
        </div>
        <div class="classification-card">
            <div class="classification-mark improve-mark">!</div>
            <div>
                <div class="classification-label">NEEDS IMPROVEMENT</div>
                <div class="classification-threshold">Quality score below 6</div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")

st.subheader("📊 Dataset overview")
st.dataframe(data.head(10), width="stretch")

st.write("")
st.subheader("🧪 Predict uploaded wine samples")
st.write(
    "Upload a CSV file containing the same physicochemical columns used in training. "
    "The system will check that the required inputs are present and return a preliminary quality label."
)

template = data.drop(columns=["quality"]).head(5)
st.download_button(
    "Download sample input CSV",
    template.to_csv(index=False),
    "sample_wine_input.csv",
    "text/csv",
)

uploaded = st.file_uploader("Upload wine sample data", type=["csv"])

if uploaded is not None:
    new_data = pd.read_csv(uploaded)
    required_columns = list(model.feature_names_in_)
    missing = [column for column in required_columns if column not in new_data.columns]

    if missing:
        st.error("Missing required columns: " + ", ".join(missing))
        st.info(
            "Expected columns include: " + ", ".join(required_columns[:5]) + " ... and the remaining wine features."
        )
    else:
        predictions = model.predict(new_data[required_columns])
        results = new_data.copy()
        results["Prediction"] = [
            "GOOD QUALITY" if value == 1 else "NEEDS IMPROVEMENT"
            for value in predictions
        ]

        st.write("")
        st.subheader("✅ Prediction results")
        st.dataframe(results, width="stretch")

        good_count = int((predictions == 1).sum())
        poor_count = int((predictions == 0).sum())
        colA, colB = st.columns(2)
        colA.metric("Predicted good quality", good_count)
        colB.metric("Predicted needs improvement", poor_count)

        st.write("")
        st.warning(
            "This is a preliminary screening result only and is not a substitute for laboratory testing or professional quality control decisions."
        )

        st.download_button(
            "Download prediction results",
            results.to_csv(index=False),
            "wine_quality_predictions.csv",
            "text/csv",
        )
