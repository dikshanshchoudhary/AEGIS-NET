from __future__ import annotations

import hashlib
import html
import io

import httpx
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="IMB Threat Monitor",
    page_icon="🛡️",
    layout="wide",
)

st.markdown(
    """
    <style>
    :root {
        --imb-bg: #071018;
        --imb-panel: rgba(15, 29, 40, 0.88);
        --imb-border: rgba(133, 170, 190, 0.16);
        --imb-text: #e8f1f5;
        --imb-muted: #91a6b1;
        --imb-mint: #48e1b0;
        --imb-blue: #70b7ff;
    }

    .stApp {
        background:
            radial-gradient(ellipse at 78% -10%, rgba(35, 122, 114, 0.18), transparent 38%),
            radial-gradient(ellipse at 5% 25%, rgba(49, 104, 151, 0.13), transparent 34%),
            var(--imb-bg);
        color: var(--imb-text);
    }

    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stSidebar"] {
        background: rgba(7, 16, 24, 0.96);
        border-right: 1px solid var(--imb-border);
    }
    .block-container {
        max-width: 1480px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    .imb-hero {
        position: relative;
        overflow: hidden;
        padding: 30px 34px;
        margin: 0 0 24px;
        border: 1px solid rgba(72, 225, 176, 0.24);
        border-radius: 22px;
        background:
            linear-gradient(112deg, rgba(16, 42, 48, 0.96), rgba(11, 25, 38, 0.94) 56%, rgba(16, 35, 50, 0.96));
        box-shadow: 0 22px 60px rgba(0, 0, 0, 0.22);
    }
    .imb-hero::after {
        content: "";
        position: absolute;
        top: -110px;
        right: -40px;
        width: 330px;
        height: 330px;
        border: 1px solid rgba(72, 225, 176, 0.12);
        border-radius: 50%;
        box-shadow: 0 0 0 28px rgba(72, 225, 176, 0.025), 0 0 0 58px rgba(112, 183, 255, 0.025);
        pointer-events: none;
    }
    .imb-eyebrow {
        display: flex;
        align-items: center;
        gap: 9px;
        color: var(--imb-mint);
        font: 700 0.72rem/1.2 ui-monospace, SFMono-Regular, Consolas, monospace;
        letter-spacing: 0.14em;
        text-transform: uppercase;
    }
    .imb-live-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: var(--imb-mint);
        box-shadow: 0 0 14px rgba(72, 225, 176, 0.8);
    }
    .imb-hero h1 {
        margin: 12px 0 6px;
        color: #f0f7fa;
        font-size: clamp(2.2rem, 5vw, 3.5rem);
        font-weight: 800;
        letter-spacing: -0.055em;
    }
    .imb-hero h1 span { color: var(--imb-mint); }
    .imb-hero p {
        max-width: 740px;
        margin: 0;
        color: #b1c2cb;
        font-size: 1rem;
    }
    .imb-badges {
        display: flex;
        flex-wrap: wrap;
        gap: 9px;
        margin-top: 22px;
    }
    .imb-badge {
        padding: 7px 11px;
        border: 1px solid var(--imb-border);
        border-radius: 999px;
        background: rgba(5, 14, 21, 0.48);
        color: #d3e2e8;
        font: 600 0.73rem/1 ui-monospace, SFMono-Regular, Consolas, monospace;
        letter-spacing: 0.035em;
    }
    .imb-badge.good { border-color: rgba(72, 225, 176, 0.28); color: var(--imb-mint); }
    .imb-badge.neutral { border-color: rgba(112, 183, 255, 0.25); color: var(--imb-blue); }

    [data-testid="stMetric"] {
        min-height: 122px;
        padding: 17px 18px;
        border: 1px solid var(--imb-border);
        border-radius: 15px;
        background: linear-gradient(145deg, rgba(18, 35, 47, 0.93), rgba(11, 23, 33, 0.9));
        box-shadow: 0 10px 24px rgba(0, 0, 0, 0.12);
    }
    [data-testid="stMetricLabel"] p {
        color: var(--imb-muted);
        font-size: 0.79rem;
        font-weight: 650;
        letter-spacing: 0.035em;
    }
    [data-testid="stMetricValue"] {
        color: var(--imb-text);
        font-weight: 760;
    }
    [data-testid="stMetricDelta"] { font-size: 0.76rem; }
    [data-testid="stTabs"] button {
        font-weight: 650;
    }
    [data-testid="stTabs"] button[aria-selected="true"] {
        color: var(--imb-mint);
    }
    [data-testid="stDataFrame"], [data-testid="stAlert"] {
        border-radius: 13px;
    }

    .imb-model-card {
        height: 100%;
        padding: 18px;
        border: 1px solid var(--imb-border);
        border-radius: 16px;
        background: linear-gradient(150deg, rgba(18, 36, 48, 0.96), rgba(10, 21, 31, 0.96));
    }
    .imb-model-card.best {
        border-color: rgba(72, 225, 176, 0.5);
        box-shadow: inset 0 0 26px rgba(72, 225, 176, 0.045), 0 10px 30px rgba(0, 0, 0, 0.14);
    }
    .imb-model-top {
        display: flex;
        align-items: center;
        justify-content: space-between;
        color: var(--imb-muted);
        font: 700 0.67rem/1 ui-monospace, SFMono-Regular, Consolas, monospace;
        letter-spacing: 0.1em;
        text-transform: uppercase;
    }
    .imb-rank {
        color: var(--imb-mint);
        font-size: 0.7rem;
    }
    .imb-model-card h3 {
        margin: 13px 0 19px;
        color: var(--imb-text);
        font-size: 1.35rem;
        letter-spacing: -0.025em;
    }
    .imb-score-row {
        display: flex;
        justify-content: space-between;
        margin: 11px 0 6px;
        color: #a9bac3;
        font-size: 0.8rem;
    }
    .imb-score-row strong { color: #edf5f7; font-size: 0.9rem; }
    .imb-score-track {
        height: 6px;
        overflow: hidden;
        border-radius: 999px;
        background: rgba(143, 166, 179, 0.14);
    }
    .imb-score-track span {
        display: block;
        height: 100%;
        border-radius: inherit;
        background: linear-gradient(90deg, #32c99b, #70d9c0);
    }
    .imb-score-track.f1 span { background: linear-gradient(90deg, #4a9cf2, #8ecaff); }
    .imb-model-note {
        margin-top: 14px;
        color: var(--imb-muted);
        font-size: 0.72rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def _get_json(base_url: str, path: str) -> dict:
    response = httpx.get(f"{base_url}{path}", timeout=30.0)
    response.raise_for_status()
    return response.json()


def _post_json(base_url: str, path: str, payload: dict) -> dict:
    response = httpx.post(f"{base_url}{path}", json=payload, timeout=60.0)
    response.raise_for_status()
    return response.json()


def _show_request_error(error: httpx.HTTPError) -> None:
    if isinstance(error, httpx.HTTPStatusError):
        st.error(f"API error {error.response.status_code}: {error.response.text}")
    else:
        st.error(f"Could not reach the IMB API: {error}")


def _as_numeric_features(row: pd.Series) -> dict[str, float | None]:
    features: dict[str, float | None] = {}
    for name, value in row.items():
        if str(name).strip().casefold() == "label":
            continue
        numeric_value = pd.to_numeric(value, errors="coerce")
        features[str(name)] = (
            float(numeric_value) if pd.notna(numeric_value) else None
        )
    return features


def _render_model_card(
    name: str,
    metrics: dict[str, float],
    rank: int,
    *,
    best: bool,
) -> None:
    accuracy = max(0.0, min(1.0, float(metrics["accuracy"])))
    macro_f1 = max(0.0, min(1.0, float(metrics["macro_f1"])))
    card_class = "imb-model-card best" if best else "imb-model-card"
    badge = "TOP MACRO-F1" if best else f"RANK {rank}"
    st.markdown(
        f"""
        <div class="{card_class}">
          <div class="imb-model-top">
            <span>Deep learning</span><span class="imb-rank">{badge}</span>
          </div>
          <h3>{html.escape(name.upper())}</h3>
          <div class="imb-score-row"><span>Accuracy</span><strong>{accuracy:.2%}</strong></div>
          <div class="imb-score-track"><span style="width:{accuracy:.2%}"></span></div>
          <div class="imb-score-row"><span>Macro-F1</span><strong>{macro_f1:.2%}</strong></div>
          <div class="imb-score-track f1"><span style="width:{macro_f1:.2%}"></span></div>
          <div class="imb-model-note">Standalone holdout evaluation</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _model_vote_frame(result: dict) -> pd.DataFrame:
    predicted_category = result["predicted_category"]
    votes = [
        {
            "Model": "XGBoost",
            "Family": "Boosted trees",
            "Score (%)": 100 * result["xgboost_scores"].get(predicted_category, 0.0),
        }
    ]
    for model_name, scores in result["neural_scores"].items():
        votes.append(
            {
                "Model": model_name.upper(),
                "Family": "Deep learning",
                "Score (%)": 100 * scores.get(predicted_category, 0.0),
            }
        )
    return pd.DataFrame(votes).sort_values("Score (%)", ascending=False)


api_url = st.sidebar.text_input(
    "API base URL",
    value="http://127.0.0.1:8000",
).rstrip("/")
if st.sidebar.button("Refresh dashboard"):
    st.rerun()

try:
    health = _get_json(api_url, "/health")
    metrics = _get_json(api_url, "/metrics")
except httpx.HTTPError as error:
    _show_request_error(error)
    st.info("Start the local API with `uvicorn api.main:app --host 127.0.0.1 --port 8000`.")
    st.stop()

if not health.get("model_loaded"):
    st.error("The API is running, but no model bundle is loaded.")
    st.stop()

st.sidebar.success(
    f"Model ready · {health['feature_count']} features · {health['device']}"
)
device_name = str(health["device"])
device_badge = (
    "CUDA ACCELERATION ACTIVE"
    if device_name.casefold().startswith("cuda")
    else f"{device_name.upper()} INFERENCE · CUDA UNAVAILABLE"
)
model_version = html.escape(str(health["model_version"]))
st.markdown(
    f"""
    <section class="imb-hero">
      <div class="imb-eyebrow"><span class="imb-live-dot"></span>
        Defensive network intelligence · offline analysis
      </div>
      <h1>AEGIS<span>-NET</span></h1>
      <p>One flow at a time. Multiple model perspectives. Clear, explainable risk signals.</p>
      <div class="imb-badges">
        <span class="imb-badge good">DETECTION ENGINE ONLINE</span>
        <span class="imb-badge neutral">{html.escape(device_badge)}</span>
        <span class="imb-badge">{int(health['feature_count'])} FEATURES</span>
        <span class="imb-badge">MODEL {model_version[:10]}</span>
        <span class="imb-badge">NO TRAFFIC BLOCKING</span>
      </div>
    </section>
    """,
    unsafe_allow_html=True,
)

performance = metrics["model_performance"]
classifier_metrics = performance["classifiers"]
neural_metrics = {
    name: values
    for name, values in classifier_metrics.items()
    if name.casefold() != "xgboost"
}
ranked_neural_metrics = sorted(
    neural_metrics.items(),
    key=lambda item: float(item[1]["macro_f1"]),
    reverse=True,
)
best_neural_model, best_neural_scores = ranked_neural_metrics[0]
overview_tab, deep_learning_tab, analysis_tab = st.tabs(
    ["Operations overview", "Deep learning lab", "Analyze flows"]
)

with overview_tab:
    total_flows = int(metrics["total_flows"])
    suspicious_flows = int(metrics["suspicious_flows"])
    suspicious_rate = suspicious_flows / total_flows if total_flows else 0.0
    summary_columns = st.columns(4)
    summary_columns[0].metric("Flows analyzed", total_flows)
    summary_columns[1].metric("Suspicious flows", suspicious_flows)
    summary_columns[2].metric(
        "Suspicious rate",
        f"{suspicious_rate:.1%}",
        delta=f"{suspicious_flows} flagged",
        delta_color="off",
    )
    summary_columns[3].metric(
        "Leading DL model",
        best_neural_model,
        delta=f"{float(best_neural_scores['macro_f1']):.1%} macro-F1",
        delta_color="off",
    )

    chart_columns = st.columns(2)
    categories = metrics["category_distribution"]
    with chart_columns[0]:
        st.subheader("Predicted categories")
        if categories:
            category_frame = pd.DataFrame(
                [{"Category": name, "Flows": count} for name, count in categories.items()]
            ).sort_values("Flows", ascending=False)
            st.bar_chart(category_frame, x="Category", y="Flows")
        else:
            st.info("No flow records yet. Open **Analyze flows** to start a local CSV analysis.")

    with chart_columns[1]:
        st.subheader("Risk levels")
        if total_flows:
            risk_frame = pd.DataFrame(
                [
                    {"Risk level": name.title(), "Flows": count}
                    for name, count in metrics["risk_distribution"].items()
                ]
            )
            st.bar_chart(risk_frame, x="Risk level", y="Flows")
        else:
            st.info("Risk distribution will appear after the first analysis.")

    st.subheader("Recent alerts")
    try:
        alerts_response = _get_json(api_url, "/alerts?limit=20")
    except httpx.HTTPError as error:
        _show_request_error(error)
    else:
        alert_rows = alerts_response["alerts"]
        if alert_rows:
            st.dataframe(
                pd.DataFrame(alert_rows),
                hide_index=True,
                use_container_width=True,
            )
        else:
            st.info("No suspicious flows have been recorded.")

with deep_learning_tab:
    st.subheader("Neural architecture leaderboard")
    st.caption(
        "Standalone benchmark scores from the exported evaluation artifact. "
        "Macro-F1 gives each attack class more equal weight than accuracy."
    )
    for start in range(0, len(ranked_neural_metrics), 3):
        model_row = ranked_neural_metrics[start:start + 3]
        model_columns = st.columns(len(model_row), gap="medium")
        for offset, (model_name, model_scores) in enumerate(model_row):
            with model_columns[offset]:
                _render_model_card(
                    model_name,
                    model_scores,
                    start + offset + 1,
                    best=(start + offset == 0),
                )

    st.subheader("Deep learning vs. boosted-tree reference")
    benchmark_rows = [
        {
            "Model": model_name,
            "Accuracy (%)": 100 * float(model_scores["accuracy"]),
            "Macro-F1 (%)": 100 * float(model_scores["macro_f1"]),
        }
        for model_name, model_scores in classifier_metrics.items()
    ]
    benchmark_frame = pd.DataFrame(benchmark_rows).sort_values(
        "Macro-F1 (%)",
        ascending=False,
    )
    st.bar_chart(
        benchmark_frame.set_index("Model"),
        horizontal=True,
        stack=False,
        height=360,
        y_label="Score (%)",
    )
    st.dataframe(
        benchmark_frame.round({"Accuracy (%)": 2, "Macro-F1 (%)": 2}),
        hide_index=True,
        use_container_width=True,
    )
    st.caption(
        "CNN, RNN, LSTM, and Transformer inputs are fixed-width flow features "
        "ordered by column—not chronological packet or flow sequences."
    )
    st.warning(
        performance.get(
            "evaluation_note",
            "These preliminary metrics do not establish performance on another capture or network.",
        )
    )

    anomaly_metrics = performance.get("anomaly_detector", {})
    if anomaly_metrics:
        st.subheader("Autoencoder anomaly detector")
        anomaly_columns = st.columns(3)
        anomaly_columns[0].metric(
            "Accuracy",
            f"{float(anomaly_metrics['accuracy']):.2%}",
        )
        anomaly_columns[1].metric(
            "Macro-F1",
            f"{float(anomaly_metrics['macro_f1']):.2%}",
        )
        anomaly_columns[2].metric(
            "Reconstruction threshold",
            f"{float(anomaly_metrics['threshold']):.4f}",
        )
        st.warning(
            "The current autoencoder benchmark is weak. Treat its anomaly flags "
            "as experimental signals, not as a reliable standalone detector."
        )

with analysis_tab:
    st.subheader("Analyze a flow CSV")
    uploaded_file = st.file_uploader("Upload flow records", type=["csv"])
    if uploaded_file is not None:
        content = uploaded_file.getvalue()
        content_digest = hashlib.sha256(content).hexdigest()
        try:
            preview = pd.read_csv(
                io.BytesIO(content),
                encoding="cp1252",
                nrows=10_000,
            )
        except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as error:
            st.error(f"Could not read the CSV: {error}")
        else:
            if preview.empty:
                st.error("The CSV contains no flow records.")
            else:
                row_number = st.number_input(
                    "Row to inspect (zero-based)",
                    min_value=0,
                    max_value=len(preview) - 1,
                    value=0,
                    step=1,
                )
                selected_row = preview.iloc[int(row_number)]
                with st.expander("Selected input row"):
                    st.dataframe(selected_row.to_frame("Value"), use_container_width=True)

                prediction_column, file_column = st.columns(2)
                with prediction_column:
                    if st.button("Predict selected row", key="predict-row"):
                        try:
                            prediction = _post_json(
                                api_url,
                                "/predict",
                                {"features": _as_numeric_features(selected_row)},
                            )
                        except httpx.HTTPError as error:
                            _show_request_error(error)
                        else:
                            st.session_state["last_flow_prediction"] = {
                                "content_digest": content_digest,
                                "row_number": int(row_number),
                                "response": prediction,
                            }
                            st.rerun()

                saved_prediction = st.session_state.get("last_flow_prediction")
                if (
                    isinstance(saved_prediction, dict)
                    and saved_prediction.get("content_digest") == content_digest
                    and saved_prediction.get("row_number") == int(row_number)
                ):
                    result = saved_prediction["response"]["result"]
                    st.subheader("Ensemble decision")
                    result_columns = st.columns(2)
                    result_columns[0].metric(
                        "Predicted category",
                        result["predicted_category"],
                    )
                    result_columns[1].metric(
                        "Risk level",
                        f"{result['risk_level'].title()} · {result['risk_score']:.3f}",
                    )
                    st.caption(result["confidence_note"])
                    st.write("Why it was flagged:", " ".join(result["reasons"]))
                    st.markdown("**Model votes for the predicted category**")
                    vote_frame = _model_vote_frame(result)
                    st.bar_chart(
                        vote_frame.set_index("Model")[["Score (%)"]],
                        height=260,
                        y_label="Model score (%)",
                    )
                    with st.expander("Full model response"):
                        st.dataframe(
                            vote_frame,
                            hide_index=True,
                            use_container_width=True,
                        )
                        st.json(result)

                with file_column:
                    if st.button("Analyze entire CSV", key="analyze-csv"):
                        try:
                            response = httpx.post(
                                f"{api_url}/analyze-file",
                                files={
                                    "file": (
                                        uploaded_file.name,
                                        content,
                                        uploaded_file.type or "text/csv",
                                    )
                                },
                                timeout=600.0,
                            )
                            response.raise_for_status()
                            analysis = response.json()
                        except httpx.HTTPError as error:
                            _show_request_error(error)
                        else:
                            st.session_state["last_file_analysis"] = {
                                "content_digest": content_digest,
                                "response": analysis,
                            }
                            st.rerun()

                saved_analysis = st.session_state.get("last_file_analysis")
                if (
                    isinstance(saved_analysis, dict)
                    and saved_analysis.get("content_digest") == content_digest
                ):
                    analysis = saved_analysis["response"]
                    st.success(
                        f"Analyzed {analysis['rows_analyzed']} flows; "
                        f"recorded {analysis['alerts_created']} suspicious flows."
                    )
                    summary_columns = st.columns(2)
                    summary_columns[0].write("Category distribution")
                    summary_columns[0].dataframe(
                        pd.DataFrame(
                            [
                                {"Category": name, "Flows": count}
                                for name, count in analysis["category_distribution"].items()
                            ]
                        ),
                        hide_index=True,
                        use_container_width=True,
                    )
                    summary_columns[1].write("Risk distribution")
                    summary_columns[1].dataframe(
                        pd.DataFrame(
                            [
                                {"Risk level": name.title(), "Flows": count}
                                for name, count in analysis["risk_distribution"].items()
                            ]
                        ),
                        hide_index=True,
                        use_container_width=True,
                    )
                    if analysis["alert_examples"]:
                        st.write("Suspicious examples")
                        st.dataframe(
                            pd.DataFrame(analysis["alert_examples"]),
                            hide_index=True,
                            use_container_width=True,
                        )
