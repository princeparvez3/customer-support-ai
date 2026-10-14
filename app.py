from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# -----------------------------------------------------------------------------
# App configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Customer Support AI | Intent Intelligence",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "customer_support_model.pkl"
VALIDATION_DATA_PATH = ROOT / "data" / "processed" / "validation.csv"
RAW_DATA_PATH = ROOT / "data" / "raw" / "train.csv"
ERROR_REPORT_PATH = ROOT / "reports" / "validation_predictions.csv"


# -----------------------------------------------------------------------------
# Visual styling
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
      :root { --ink: #14213d; --muted: #62718b; --blue: #3767f5; --line: #e6ebf3; }
      html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
      .stApp { background: #f5f7fb; color: var(--ink); }
      [data-testid="stHeader"] { background: rgba(245,247,251,.92); }
      [data-testid="stSidebar"] { background: #111c35; border-right: 1px solid #202e4b; }
      [data-testid="stSidebar"] * { color: #e9eefb; }
      [data-testid="stSidebar"] [data-testid="stRadio"] label { padding: 8px 10px; border-radius: 9px; }
      [data-testid="stSidebar"] [data-testid="stRadio"] label:hover { background: #1a2949; }
      [data-testid="stSidebar"] hr { border-color: #2b3957; }
      .brand-row { display:flex; align-items:center; gap:12px; padding: 8px 0 24px; }
      .brand-mark { width:44px; height:44px; border-radius:14px; display:flex; align-items:center; justify-content:center; background:linear-gradient(135deg,#5c83ff,#3151d8); color:white; font-size:23px; box-shadow:0 8px 24px rgba(55,103,245,.3); }
      .brand-name { font-family:'Manrope',sans-serif; font-size:18px; font-weight:800; line-height:1.2; color:#fff; }
      .brand-sub { font-size:11px; color:#9eacc8; margin-top:4px; letter-spacing:.06em; text-transform:uppercase; }
      .side-label { color:#7f91b3; font-size:10px; font-weight:700; text-transform:uppercase; letter-spacing:.14em; margin: 5px 0 8px; }
      .hero { padding: 27px 30px; border-radius:20px; background:linear-gradient(115deg,#182747 0%,#203c72 58%,#315bd4 100%); color:white; position:relative; overflow:hidden; margin-bottom:22px; border:1px solid rgba(255,255,255,.12); }
      .hero:after { content:''; position:absolute; width:250px; height:250px; right:-55px; top:-120px; border-radius:50%; border:1px solid rgba(255,255,255,.15); box-shadow:0 0 0 30px rgba(255,255,255,.035),0 0 0 60px rgba(255,255,255,.025); }
      .eyebrow { font-size:11px; text-transform:uppercase; letter-spacing:.16em; font-weight:700; color:#baccff; margin-bottom:9px; }
      .hero h1 { font-family:'Manrope',sans-serif; font-size:clamp(26px,3.2vw,38px); font-weight:800; color:#fff; letter-spacing:-.04em; line-height:1.16; margin:0 0 10px 0; }
      .hero p { max-width:720px; color:#d9e3ff; font-size:14px; line-height:1.65; margin:0; }
      .section-title { font-family:'Manrope',sans-serif; font-size:21px; font-weight:800; letter-spacing:-.025em; color:#172642; margin:5px 0 4px; }
      .section-subtitle { color:#70809b; font-size:13px; margin-bottom:17px; }
      .metric-card { background:white; border:1px solid #e7ecf5; border-radius:15px; padding:18px 19px; min-height:118px; box-shadow:0 3px 12px rgba(24,43,78,.025); }
      .metric-label { font-size:12px; color:#6c7a93; font-weight:600; margin-bottom:13px; }
      .metric-value { font-family:'Manrope',sans-serif; font-size:27px; color:#182947; font-weight:800; letter-spacing:-.04em; line-height:1.15; }
      .metric-foot { color:#8b97ac; font-size:11px; margin-top:8px; }
      .panel { background:white; border:1px solid #e7ecf5; border-radius:16px; padding:21px; box-shadow:0 3px 12px rgba(24,43,78,.025); margin-bottom:12px; }
      .panel-title { font-family:'Manrope',sans-serif; color:#172642; font-size:15px; font-weight:800; margin-bottom:5px; }
      .panel-desc { color:#75829a; font-size:12px; margin-bottom:15px; line-height:1.55; }
      .pill { display:inline-block; padding:5px 9px; border-radius:99px; background:#edf2ff; color:#315bd1; font-size:10px; font-weight:700; letter-spacing:.035em; }
      .step-row { display:flex; gap:12px; align-items:flex-start; margin:12px 0; }
      .step-num { width:27px; height:27px; flex:0 0 27px; border-radius:9px; background:#edf2ff; color:#315bd1; display:flex; align-items:center; justify-content:center; font-size:12px; font-weight:800; }
      .step-name { color:#263653; font-weight:700; font-size:13px; margin-bottom:3px; }
      .step-desc { color:#79869c; font-size:12px; line-height:1.5; }
      .intent-chip { display:inline-block; border:1px solid #dce5ff; background:#f2f5ff; color:#2f54b6; border-radius:10px; padding:8px 11px; font-size:13px; font-weight:700; }
      .result-box { border:1px solid #dce6ff; background:linear-gradient(120deg,#f6f8ff,#fff); border-radius:15px; padding:21px; }
      .result-label { color:#71809a; font-size:11px; text-transform:uppercase; letter-spacing:.1em; font-weight:700; margin-bottom:8px; }
      .result-intent { font-family:'Manrope',sans-serif; font-size:25px; line-height:1.25; font-weight:800; color:#244cb9; overflow-wrap:anywhere; }
      .small-note { color:#8390a5; font-size:11px; line-height:1.5; }
      div.stButton > button { border-radius:10px; min-height:43px; font-weight:700; border:1px solid #dce4f4; }
      div.stButton > button[kind="primary"] { background:linear-gradient(135deg,#416ff5,#3156db); border:0; box-shadow:0 7px 16px rgba(55,103,245,.17); }
      div.stButton > button[kind="primary"] p { color:white; }
      div[data-testid="stTextArea"] textarea, div[data-testid="stTextInput"] input { border-radius:11px; border-color:#dfe6f1; background:white; }
      div[data-testid="stFileUploader"] { background:white; border:1px dashed #cbd6e9; border-radius:13px; padding:9px; }
      div[data-testid="stDataFrame"] { border:1px solid #e5ebf4; border-radius:12px; overflow:hidden; }
      .footer { text-align:center; padding:24px 0 5px; color:#9aa5b7; font-size:11px; }
      @media (max-width: 720px) { .hero { padding:22px; } .metric-card { padding:14px; } .metric-value { font-size:23px; } }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading the trained classification model…")
def load_model(model_path: str):
    """Load the existing trained model once per Streamlit process."""
    return joblib.load(model_path)


def safe_read_csv(path: Path):
    """Read an optional project CSV safely; return None if it isn't present."""
    try:
        if path.exists():
            return pd.read_csv(path)
    except (OSError, ValueError, pd.errors.ParserError):
        return None
    return None


def pretty_intent(value) -> str:
    return str(value).replace("_", " ").strip().title()


def html_metric(label: str, value: str, foot: str, accent: str = "#3767f5") -> str:
    return f"""
    <div class="metric-card" style="border-top:3px solid {accent}">
      <div class="metric-label">{label}</div>
      <div class="metric-value">{value}</div>
      <div class="metric-foot">{foot}</div>
    </div>
    """


def page_header(kicker: str, title: str, subtitle: str) -> None:
    st.markdown(
        f"""<div class="hero"><div class="eyebrow">{kicker}</div>
        <h1>{title}</h1><p>{subtitle}</p></div>""",
        unsafe_allow_html=True,
    )


def panel_heading(title: str, description: str = "") -> None:
    st.markdown(
        f"<div class='panel-title'>{title}</div><div class='panel-desc'>{description}</div>",
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# Load model and optional project files
# -----------------------------------------------------------------------------
if not MODEL_PATH.exists():
    st.error(
        "Trained model file was not found. Expected: `models/customer_support_model.pkl`. "
        "Keep the app.py file in the project root and make sure the model file is committed to the repository."
    )
    st.stop()

try:
    model = load_model(str(MODEL_PATH))
except Exception as exc:  # display a readable deployment error rather than a blank app
    st.error("The trained model could not be loaded. Check that the model file and scikit-learn versions are compatible.")
    with st.expander("Technical details"):
        st.code(str(exc))
    st.stop()

validation_df = safe_read_csv(VALIDATION_DATA_PATH)
raw_df = safe_read_csv(RAW_DATA_PATH)
error_df = safe_read_csv(ERROR_REPORT_PATH)

# Accuracy/F1 scores are the project's recorded evaluation results (README),
# not metrics re-computed live by this Streamlit interface.
VALIDATION_ACCURACY = 83.90
TEST_ACCURACY = 84.70
VALIDATION_MACRO_F1 = 0.83
TEST_MACRO_F1 = 0.85

try:
    number_of_intents = len(model.classes_)
except (AttributeError, TypeError):
    number_of_intents = 77  # Banking77 target label space described in README


# -----------------------------------------------------------------------------
# Sidebar navigation
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """<div class="brand-row"><div class="brand-mark">✦</div>
        <div><div class="brand-name">SupportIQ</div>
        <div class="brand-sub">Customer Support AI</div></div></div>""",
        unsafe_allow_html=True,
    )
    st.markdown("<div class='side-label'>Workspace</div>", unsafe_allow_html=True)
    page = st.radio(
        "Navigate",
        ["Dashboard", "Live Predictor", "Batch Classifier", "Error Analysis", "Dataset Explorer"],
        label_visibility="collapsed",
    )
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<div class='side-label'>Model snapshot</div>", unsafe_allow_html=True)
    st.markdown(
        f"<div style='font-size:12px;line-height:1.9;color:#c4d0e7'>"
        f"<div>◉ &nbsp; {number_of_intents} intent classes</div>"
        f"<div>◉ &nbsp; {TEST_ACCURACY:.2f}% recorded test accuracy</div>"
        f"<div>◉ &nbsp; Banking77 dataset</div></div>",
        unsafe_allow_html=True,
    )
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size:11px;color:#8999b8;line-height:1.6'>"
        "Intent predictions are suggestions from a closed-set classifier. "
        "Review sensitive or ambiguous cases before taking action.</div>",
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# Dashboard
# -----------------------------------------------------------------------------
if page == "Dashboard":
    page_header(
        "INTENT INTELLIGENCE / OVERVIEW",
        "Understand every customer message.",
        "A machine-learning workspace for classifying banking support requests, exploring quality results, and reviewing common prediction errors.",
    )

    metrics = st.columns(4, gap="medium")
    cards = [
        ("VALIDATION ACCURACY", f"{VALIDATION_ACCURACY:.2f}%", "1,000 validation records", "#3767f5"),
        ("TEST ACCURACY", f"{TEST_ACCURACY:.2f}%", "1,000 unseen test records", "#19a879"),
        ("INTENT CLASSES", str(number_of_intents), "Banking-support categories", "#8b5cf6"),
        ("VALIDATION MACRO F1", f"{VALIDATION_MACRO_F1:.2f}", "Per-class performance balance", "#f59e0b"),
    ]
    for col, (label, value, foot, color) in zip(metrics, cards):
        with col:
            st.markdown(html_metric(label, value, foot, color), unsafe_allow_html=True)

    st.write("")
    left, right = st.columns([1.15, 0.85], gap="large")
    with left:
        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        panel_heading("Model performance", "Recorded evaluation metrics from the project's documented validation and held-out test runs.")
        perf = pd.DataFrame(
            {
                "Evaluation set": ["Validation", "Test"],
                "Accuracy (%)": [VALIDATION_ACCURACY, TEST_ACCURACY],
                "Macro F1 (%)": [VALIDATION_MACRO_F1 * 100, TEST_MACRO_F1 * 100],
            }
        ).set_index("Evaluation set")
        st.bar_chart(perf, height=250)
        st.markdown(
            "<div class='small-note'>These are saved project results, not a live benchmark run when this page opens.</div>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        panel_heading("What you can do", "Use the workspace to test the model and inspect where it gets confused.")
        steps = [
            ("1", "Predict a message", "Enter a single customer message and view the predicted intent."),
            ("2", "Classify a CSV", "Upload a file with a message/text column and download the predictions."),
            ("3", "Review model errors", "Explore actual vs predicted labels from the saved validation error report."),
            ("4", "Explore the dataset", "Filter available validation or raw data by category and search terms."),
        ]
        for num, name, desc in steps:
            st.markdown(
                f"<div class='step-row'><div class='step-num'>{num}</div><div><div class='step-name'>{name}</div><div class='step-desc'>{desc}</div></div></div>",
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Built for practical evaluation</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-subtitle'>A clear interface around the existing trained intent classifier.</div>", unsafe_allow_html=True)
    feature_cols = st.columns(3, gap="medium")
    features = [
        ("⌕", "Intent prediction", "Turn a short banking-support message into one of the model's known intent labels."),
        ("▦", "Batch workflow", "Process many messages at once and export a CSV containing each prediction."),
        ("↗", "Error visibility", "Inspect common actual-to-predicted label pairs to identify confusing intent classes."),
    ]
    for col, (symbol, title, description) in zip(feature_cols, features):
        with col:
            st.markdown(
                f"<div class='panel' style='height:145px'><div style='font-size:23px;color:#416ff5;margin-bottom:10px'>{symbol}</div><div class='panel-title'>{title}</div><div class='panel-desc' style='margin-bottom:0'>{description}</div></div>",
                unsafe_allow_html=True,
            )


# -----------------------------------------------------------------------------
# Single-message predictor
# -----------------------------------------------------------------------------
elif page == "Live Predictor":
    page_header(
        "LIVE INFERENCE / SINGLE MESSAGE",
        "Classify a customer message.",
        "Enter a banking-support query and run it through the existing trained classification model.",
    )
    left, right = st.columns([1.2, 0.8], gap="large")
    with left:
        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        panel_heading("Customer message", "Try a realistic example or paste your own message below.")
        example = st.selectbox(
            "Example messages",
            [
                "Choose an example…",
                "My card hasn't arrived yet.",
                "I want to cancel my transfer.",
                "A payment was taken from my account that I don't recognise.",
                "Why is my cash withdrawal still pending?",
                "I can't top up my account using my card.",
            ],
        )
        default_text = "" if example == "Choose an example…" else example
        customer_text = st.text_area(
            "Message text",
            value=default_text,
            height=160,
            placeholder="Example: My card hasn't arrived yet",
            label_visibility="collapsed",
        )
        predict_clicked = st.button("✦  Predict intent", type="primary", use_container_width=True)
        st.markdown(
            "<div class='small-note'>Tip: include the customer's main issue and any relevant detail. The model always selects from its known intent classes.</div>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        panel_heading("How this works", "A short message goes through the same trained model used by the project's CLI predictor.")
        for num, name, desc in [
            ("01", "Input", "The submitted customer message is passed to the classifier."),
            ("02", "Inference", "The saved model predicts one label from its known categories."),
            ("03", "Result", "The predicted intent is shown for review; no support action is automatically taken."),
        ]:
            st.markdown(
                f"<div class='step-row'><div class='step-num'>{num}</div><div><div class='step-name'>{name}</div><div class='step-desc'>{desc}</div></div></div>",
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

    if predict_clicked:
        if not customer_text.strip():
            st.warning("Please enter a customer message before running a prediction.")
        else:
            try:
                with st.spinner("Analyzing message…"):
                    prediction = str(model.predict([customer_text.strip()])[0])
                st.markdown("<div class='section-title'>Prediction result</div>", unsafe_allow_html=True)
                st.markdown(
                    f"<div class='result-box'><div class='result-label'>Predicted intent</div><div class='result-intent'>{pretty_intent(prediction)}</div><div style='margin-top:10px'><span class='intent-chip'>{prediction}</span></div></div>",
                    unsafe_allow_html=True,
                )
                st.caption("This is a model prediction, not a verified ground-truth label. Review uncertain or high-impact cases.")
            except Exception as exc:
                st.error("Prediction failed. Please check the model file and the input, then try again.")
                with st.expander("Technical details"):
                    st.code(str(exc))


# -----------------------------------------------------------------------------
# Batch classifier
# -----------------------------------------------------------------------------
elif page == "Batch Classifier":
    page_header(
        "BATCH INFERENCE / CSV WORKFLOW",
        "Classify messages in bulk.",
        "Upload a CSV, select the column containing customer messages, and export an annotated file with predictions.",
    )
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    panel_heading("Upload customer messages", "CSV files with a `text`, `message`, `customer_message`, or `query` column are detected automatically.")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    if uploaded_file is not None:
        try:
            upload_df = pd.read_csv(uploaded_file)
            if upload_df.empty:
                st.warning("This CSV has no rows to classify.")
            else:
                st.markdown(f"<span class='pill'>{len(upload_df):,} ROWS LOADED</span>", unsafe_allow_html=True)
                st.write("")
                preferred = ["text", "message", "customer_message", "query", "customer_query", "content"]
                matched = next((c for c in preferred if c in upload_df.columns), None)
                default_index = list(upload_df.columns).index(matched) if matched else 0
                text_column = st.selectbox("Message column", options=list(upload_df.columns), index=default_index)
                st.dataframe(upload_df.head(8), use_container_width=True, hide_index=True)
                run_batch = st.button("Run batch predictions", type="primary", use_container_width=True)
                if run_batch:
                    working_df = upload_df.copy()
                    messages = working_df[text_column].fillna("").astype(str).str.strip()
                    nonempty_mask = messages.ne("")
                    if not nonempty_mask.any():
                        st.warning("The selected column does not contain any non-empty messages.")
                    else:
                        try:
                            with st.spinner(f"Classifying {int(nonempty_mask.sum()):,} messages…"):
                                preds = pd.Series(index=working_df.index, dtype="object")
                                preds.loc[nonempty_mask] = model.predict(messages.loc[nonempty_mask].tolist())
                            working_df["predicted_intent"] = preds
                            working_df["prediction_status"] = nonempty_mask.map({True: "Predicted", False: "Empty message"})
                            st.session_state["batch_result_df"] = working_df
                            st.session_state["batch_result_name"] = uploaded_file.name.rsplit(".", 1)[0]
                        except Exception as exc:
                            st.error("Batch prediction failed. Please verify the uploaded data and model.")
                            st.code(str(exc))
        except (UnicodeDecodeError, pd.errors.ParserError, ValueError) as exc:
            st.error("The CSV could not be read. Save it as a UTF-8 CSV and try again.")
            st.caption(str(exc))
    st.markdown("</div>", unsafe_allow_html=True)

    result_df = st.session_state.get("batch_result_df")
    if isinstance(result_df, pd.DataFrame):
        st.markdown("<div class='section-title'>Batch prediction results</div>", unsafe_allow_html=True)
        predicted_count = int(result_df["predicted_intent"].notna().sum())
        empty_count = len(result_df) - predicted_count
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(html_metric("TOTAL ROWS", f"{len(result_df):,}", "Rows from uploaded CSV"), unsafe_allow_html=True)
        with m2:
            st.markdown(html_metric("PREDICTED", f"{predicted_count:,}", "Non-empty messages", "#19a879"), unsafe_allow_html=True)
        with m3:
            st.markdown(html_metric("SKIPPED", f"{empty_count:,}", "Empty message values", "#f59e0b"), unsafe_allow_html=True)
        st.dataframe(result_df, use_container_width=True, hide_index=True)
        csv_bytes = result_df.to_csv(index=False).encode("utf-8-sig")
        output_name = st.session_state.get("batch_result_name", "customer_support_predictions") + "_predictions.csv"
        st.download_button(
            "Download predictions CSV",
            data=csv_bytes,
            file_name=output_name,
            mime="text/csv",
            type="primary",
        )
        st.caption("Predicted labels are model outputs and should be checked before use in operational decisions.")


# -----------------------------------------------------------------------------
# Error analysis
# -----------------------------------------------------------------------------
elif page == "Error Analysis":
    page_header(
        "MODEL QUALITY / ERROR ANALYSIS",
        "See where the model gets confused.",
        "Inspect incorrect validation predictions and the most frequent actual-to-predicted label pairs.",
    )
    if error_df is None or error_df.empty:
        st.info("No saved error report was found. Run `scripts/error_analysis.py` in the project folder to generate `reports/validation_predictions.csv`.")
    elif not {"text", "category", "predicted_category"}.issubset(error_df.columns):
        st.warning("The saved report does not contain the expected columns: text, category, predicted_category.")
        st.dataframe(error_df.head(50), use_container_width=True, hide_index=True)
    else:
        analysis_df = error_df[["text", "category", "predicted_category"]].copy()
        analysis_df["actual_label"] = analysis_df["category"].astype(str)
        analysis_df["predicted_label"] = analysis_df["predicted_category"].astype(str)
        analysis_df["confusion_pair"] = analysis_df["actual_label"] + "  →  " + analysis_df["predicted_label"]
        unique_actual = analysis_df["actual_label"].nunique()
        unique_pred = analysis_df["predicted_label"].nunique()
        top_pair = analysis_df["confusion_pair"].value_counts().index[0] if len(analysis_df) else "—"

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(html_metric("SAVED ERROR ROWS", f"{len(analysis_df):,}", "Rows in the error-only report", "#e17b38"), unsafe_allow_html=True)
        with c2:
            st.markdown(html_metric("ACTUAL LABELS", f"{unique_actual:,}", "Distinct labels in these errors", "#8b5cf6"), unsafe_allow_html=True)
        with c3:
            st.markdown(html_metric("PREDICTED LABELS", f"{unique_pred:,}", "Distinct predicted labels", "#3767f5"), unsafe_allow_html=True)

        st.write("")
        left, right = st.columns([1, 1], gap="large")
        with left:
            st.markdown("<div class='panel'>", unsafe_allow_html=True)
            panel_heading("Most common confusion pairs", "Counts from the saved report. The current report contains incorrect validation rows only.")
            pairs = analysis_df["confusion_pair"].value_counts().head(10).rename_axis("Actual → Predicted").to_frame("Rows")
            st.bar_chart(pairs, horizontal=True, height=330)
            st.markdown("</div>", unsafe_allow_html=True)
        with right:
            st.markdown("<div class='panel'>", unsafe_allow_html=True)
            panel_heading("Browse incorrect predictions", "Filter the saved report to review representative examples.")
            label_options = ["All actual labels"] + sorted(analysis_df["actual_label"].unique().tolist())
            chosen_actual = st.selectbox("Actual intent", label_options)
            filtered = analysis_df if chosen_actual == "All actual labels" else analysis_df[analysis_df["actual_label"] == chosen_actual]
            search_error = st.text_input("Search message text", placeholder="e.g. card, transfer, top up")
            if search_error.strip():
                filtered = filtered[filtered["text"].astype(str).str.contains(search_error.strip(), case=False, na=False)]
            st.dataframe(
                filtered[["text", "actual_label", "predicted_label"]].rename(
                    columns={"text": "Customer message", "actual_label": "Actual intent", "predicted_label": "Predicted intent"}
                ),
                use_container_width=True,
                hide_index=True,
                height=330,
            )
            st.markdown("</div>", unsafe_allow_html=True)

        st.info("Important: this report is generated by the project's error-analysis script and contains incorrect validation examples only. The error-row count must not be used by itself to calculate accuracy.")
        st.download_button(
            "Download filtered error report",
            data=analysis_df[["text", "category", "predicted_category"]].to_csv(index=False).encode("utf-8-sig"),
            file_name="customer_support_error_analysis.csv",
            mime="text/csv",
        )


# -----------------------------------------------------------------------------
# Dataset explorer
# -----------------------------------------------------------------------------
elif page == "Dataset Explorer":
    page_header(
        "DATA QUALITY / DATA EXPLORER",
        "Explore messages and intent labels.",
        "Search sample customer messages, filter by intent, and inspect the data available in this repository.",
    )
    selected_df = validation_df if validation_df is not None and not validation_df.empty else raw_df
    source_name = "Processed validation data" if selected_df is validation_df and validation_df is not None else "Raw training data"
    if selected_df is None or selected_df.empty:
        st.info("No dataset CSV was found in the expected `data/processed/validation.csv` or `data/raw/train.csv` locations.")
    elif not {"text", "category"}.issubset(selected_df.columns):
        st.warning("The available CSV does not contain both expected columns: `text` and `category`.")
        st.dataframe(selected_df.head(50), use_container_width=True, hide_index=True)
    else:
        data = selected_df[["text", "category"]].copy()
        data["text"] = data["text"].fillna("").astype(str)
        data["category"] = data["category"].fillna("unknown").astype(str)
        st.markdown(f"<span class='pill'>SOURCE: {source_name.upper()}</span>", unsafe_allow_html=True)
        st.write("")
        a, b, c = st.columns(3)
        with a:
            st.markdown(html_metric("ROWS AVAILABLE", f"{len(data):,}", source_name), unsafe_allow_html=True)
        with b:
            st.markdown(html_metric("UNIQUE INTENTS", f"{data['category'].nunique():,}", "Distinct category labels", "#8b5cf6"), unsafe_allow_html=True)
        with c:
            st.markdown(html_metric("EMPTY MESSAGES", f"{int(data['text'].str.strip().eq('').sum()):,}", "Blank text values", "#f59e0b"), unsafe_allow_html=True)

        st.write("")
        left, right = st.columns([0.75, 1.25], gap="large")
        with left:
            st.markdown("<div class='panel'>", unsafe_allow_html=True)
            panel_heading("Most frequent intents", "Top categories in the selected file.")
            counts = data["category"].value_counts().head(12).rename_axis("Intent").to_frame("Rows")
            st.bar_chart(counts, horizontal=True, height=350)
            st.markdown("</div>", unsafe_allow_html=True)
        with right:
            st.markdown("<div class='panel'>", unsafe_allow_html=True)
            panel_heading("Search the dataset", "Filter by category and/or a phrase in the customer message.")
            categories = ["All categories"] + sorted(data["category"].unique().tolist())
            chosen_category = st.selectbox("Category", categories)
            text_query = st.text_input("Search message", placeholder="Try: card, refund, transfer…")
            filtered_data = data
            if chosen_category != "All categories":
                filtered_data = filtered_data[filtered_data["category"] == chosen_category]
            if text_query.strip():
                filtered_data = filtered_data[filtered_data["text"].str.contains(text_query.strip(), case=False, na=False)]
            st.caption(f"Showing {len(filtered_data):,} matching rows")
            st.dataframe(
                filtered_data.rename(columns={"text": "Customer message", "category": "Intent label"}),
                use_container_width=True,
                hide_index=True,
                height=310,
            )
            st.download_button(
                "Download filtered rows",
                data=filtered_data.to_csv(index=False).encode("utf-8-sig"),
                file_name="customer_support_dataset_filtered.csv",
                mime="text/csv",
            )
            st.markdown("</div>", unsafe_allow_html=True)

    st.warning("Model limitation: this is a closed-set classifier. Messages outside the banking-support domain may still be assigned one of the known intents; the current model does not have a dedicated unknown/out-of-domain label.")


st.markdown(
    "<div class='footer'>CUSTOMER SUPPORT AI · INTENT CLASSIFICATION WORKSPACE<br>Predictions support review; they do not replace human judgment.</div>",
    unsafe_allow_html=True,
)
