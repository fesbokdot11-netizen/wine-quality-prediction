import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Wine Quality Predictor",
    page_icon="🍷",
    layout="wide"
)

st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        html, body, [data-testid="stAppViewContainer"] {
            background:
                radial-gradient(ellipse at 8% 8%, rgba(190, 24, 93, 0.13), transparent 34%),
                radial-gradient(ellipse at 92% 18%, rgba(127, 29, 29, 0.12), transparent 32%),
                linear-gradient(135deg, #fff8f4 0%, #f8e9e9 48%, #fdf3e8 100%);
            color: #1f2937;
            font-family: 'Inter', sans-serif;
        }

        .hero {
            background: linear-gradient(135deg, rgba(127, 29, 29, 0.96), rgba(190, 24, 93, 0.85));
            border-radius: 22px;
            padding: 1.5rem 1.75rem;
            box-shadow: 0 18px 38px rgba(127, 29, 29, 0.16);
            margin-bottom: 1rem;
        }

        .hero h1 {
            color: #fff;
            margin: 0;
            font-size: 2.3rem;
            font-weight: 800;
        }

        .hero p {
            color: rgba(255,255,255,0.85);
            margin: 0.45rem 0 0;
            font-size: 1rem;
        }

        .tag-row {
            display: flex;
            gap: 0.6rem;
            flex-wrap: wrap;
            margin-top: 1rem;
        }

        .tag {
            background: rgba(255,255,255,0.14);
            border: 1px solid rgba(255,255,255,0.18);
            color: #fff;
            border-radius: 999px;
            padding: 0.35rem 0.75rem;
            font-size: 0.8rem;
            font-weight: 600;
        }

        .metric-card {
            background: rgba(255,255,255,0.7);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(148, 163, 184, 0.2);
            border-radius: 18px;
            padding: 1rem 1.1rem;
            box-shadow: 0 10px 30px rgba(15, 23, 42, 0.06);
            height: 100%;
        }

        .metric-label {
            color: #64748b;
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.4rem;
            font-weight: 700;
        }

        .metric-value {
            font-size: 2rem;
            font-weight: 800;
            color: #0f172a;
            line-height: 1.1;
        }

        .panel {
            background: rgba(255,255,255,0.72);
            border: 1px solid rgba(148, 163, 184, 0.2);
            border-radius: 20px;
            padding: 1rem 1.1rem;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.05);
            margin-top: 1rem;
        }

        .section-title {
            color: #111827;
            font-weight: 800;
            font-size: 1.25rem;
            margin-bottom: 0.75rem;
        }

        .status-box {
            background: linear-gradient(135deg, #eefdf5, #dcfce7);
            border: 1px solid rgba(34, 197, 94, 0.25);
            border-radius: 14px;
            padding: 0.8rem 1rem;
            color: #166534;
            font-weight: 700;
            margin-bottom: 0.8rem;
        }

        .warning-box {
            background: linear-gradient(135deg, #fff7ed, #ffedd5);
            border: 1px solid rgba(251, 146, 60, 0.3);
            border-radius: 14px;
            padding: 0.8rem 1rem;
            color: #9a4d00;
            font-weight: 700;
            margin-bottom: 0.8rem;
        }

        [data-testid="stFileUploader"] > section {
            background: rgba(255,255,255,0.8);
            border: 1px dashed rgba(147, 51, 234, 0.5);
            border-radius: 16px;
        }

        [data-testid="stRadio"] > div {
            gap: 0.65rem;
        }

        [data-testid="stRadio"] label {
            background: rgba(255,255,255,0.82);
            border: 1px solid rgba(124, 58, 237, 0.2);
            border-radius: 14px;
            padding: 0.8rem 1rem;
            min-height: 3.2rem;
            box-shadow: 0 5px 14px rgba(15, 23, 42, 0.06);
            transition: transform 160ms ease, box-shadow 160ms ease, border-color 160ms ease;
        }

        [data-testid="stRadio"] label:hover {
            transform: translateY(-2px);
            border-color: rgba(124, 58, 237, 0.55);
            box-shadow: 0 9px 20px rgba(124, 58, 237, 0.12);
        }

        [data-testid="stRadio"] label p {
            font-size: 1rem;
            font-weight: 700;
        }

        [data-testid="stRadio"] label:has(input:checked) {
            background: linear-gradient(135deg, #7c3aed, #ec4899);
            border-color: transparent;
        }

        [data-testid="stRadio"] label:has(input:checked) p {
            color: #fff;
        }

        .stDownloadButton > button, .stButton > button {
            width: 100%;
            min-height: 3.25rem;
            background: linear-gradient(135deg, #7c3aed, #ec4899);
            color: white;
            border: none;
            border-radius: 14px;
            font-size: 1rem;
            font-weight: 800;
            padding: 0.8rem 1.2rem;
            box-shadow: 0 8px 18px rgba(124, 58, 237, 0.2);
            transition: transform 160ms ease, box-shadow 160ms ease;
        }

        .stDownloadButton > button:hover, .stButton > button:hover {
            color: white;
            transform: translateY(-2px);
            box-shadow: 0 12px 24px rgba(124, 58, 237, 0.3);
        }

        .footer {
            text-align: center;
            color: #64748b;
            font-size: 0.85rem;
            padding: 2rem 0 0.5rem;
            letter-spacing: 0.03em;
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
    <div class="hero">
        <h1>🍷 Wine Quality Prediction System</h1>
        <div class="tag-row">
            <span class="tag">Random Forest</span>
            <span class="tag">Red Wine Dataset</span>
            <span class="tag">CSV Prediction Flow</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(
        """
        <div class="metric-card">
            <div class="metric-label">Records</div>
            <div class="metric-value">{}</div>
        </div>
        """.format(len(data)),
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        """
        <div class="metric-card">
            <div class="metric-label">Avg. Quality</div>
            <div class="metric-value">{}</div>
        </div>
        """.format(round(data["quality"].mean(), 2)),
        unsafe_allow_html=True,
    )
with c3:
    st.markdown(
        """
        <div class="metric-card">
            <div class="metric-label">Good Quality</div>
            <div class="metric-value">{}</div>
        </div>
        """.format(int((data["quality"] >= 6).sum())),
        unsafe_allow_html=True,
    )

page = st.sidebar.radio("Navigation", ["Dashboard", "Upload & Download"])

if page == "Dashboard":
    st.markdown('<div class="section-title">Dataset preview</div>', unsafe_allow_html=True)
    st.dataframe(data.head(10), use_container_width=True)
    st.caption(
        "Use the Upload & Download page in the sidebar to submit wine measurements "
        "and download prediction results."
    )
else:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-title">Upload & Download Predictions</div>',
        unsafe_allow_html=True,
    )
    st.write(
        "Download a sample CSV template, or upload a CSV containing the wine "
        "measurement columns used by the model."
    )

    template = data.drop(columns=["quality"]).head(5)
    st.download_button(
        "Download sample input CSV",
        template.to_csv(index=False),
        "sample_wine_input.csv",
        "text/csv",
    )

    uploaded = st.file_uploader("Choose wine sample CSV", type=["csv"])
    if uploaded is not None:
        try:
            new_data = pd.read_csv(uploaded)
        except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as error:
            st.error("The uploaded file could not be read as a valid CSV. Please check the file and try again.")
        else:
            required_columns = list(model.feature_names_in_)
            missing = [
                column for column in required_columns
                if column not in new_data.columns
            ]

            if missing:
                st.markdown(
                    "<div class='warning-box'>Missing required columns: {} </div>".format(
                        ", ".join(missing)
                    ),
                    unsafe_allow_html=True,
                )
                st.error(
                    "Please upload a file that includes all required wine measurements."
                )
            elif new_data.empty:
                st.error("The uploaded CSV contains no data rows.")
            else:
                predictions = model.predict(new_data[required_columns])
                results = new_data.copy()
                results["Prediction"] = [
                    "GOOD QUALITY" if value == 1 else "NEEDS IMPROVEMENT"
                    for value in predictions
                ]

                good_count = int((predictions == 1).sum())
                st.markdown(
                    "<div class='status-box'>Prediction summary: {} good-quality "
                    "samples out of {} reviewed.</div>".format(
                        good_count, len(results)
                    ),
                    unsafe_allow_html=True,
                )

                st.markdown(
                    '<div class="section-title">Prediction results</div>',
                    unsafe_allow_html=True,
                )
                st.dataframe(results, use_container_width=True)
                st.metric("Predicted good-quality samples", good_count)
                st.download_button(
                    "Download prediction results",
                    results.to_csv(index=False),
                    "wine_quality_predictions.csv",
                    "text/csv",
                )

    st.markdown('</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="footer">Wine Quality Prediction System™ · @Angel Faith Gomez</div>',
    unsafe_allow_html=True,
)
