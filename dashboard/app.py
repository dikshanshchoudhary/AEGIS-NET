from __future__ import annotations

import hashlib
import io

import httpx
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="AEGIS-NET Threat Monitor",
    page_icon="🛡️",
    layout="wide",
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
        st.error(f"Could not reach the AEGIS-NET API: {error}")


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


st.title("AEGIS-NET")
st.caption("Offline network-flow threat detection · analysis only · no traffic blocking")
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
overview_tab, analysis_tab = st.tabs(["Overview", "Analyze flows"])

with overview_tab:
    summary_columns = st.columns(4)
    summary_columns[0].metric("Flows analyzed", metrics["total_flows"])
    summary_columns[1].metric("Suspicious flows", metrics["suspicious_flows"])
    summary_columns[2].metric(
        "Predicted categories",
        len(metrics["category_distribution"]),
    )
    summary_columns[3].metric("Model version", health["model_version"][:10])

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
            st.info("No flows analyzed yet.")

    with chart_columns[1]:
        st.subheader("Risk levels")
        risk_frame = pd.DataFrame(
            [
                {"Risk level": name.title(), "Flows": count}
                for name, count in metrics["risk_distribution"].items()
            ]
        )
        st.bar_chart(risk_frame, x="Risk level", y="Flows")

    st.subheader("Model evaluation")
    performance = metrics["model_performance"]
    classifier_metrics = performance["classifiers"]
    if classifier_metrics:
        performance_frame = pd.DataFrame(
            [
                {
                    "Model": name,
                    "Accuracy (%)": round(values["accuracy"] * 100, 2),
                    "Macro-F1 (%)": round(values["macro_f1"] * 100, 2),
                }
                for name, values in classifier_metrics.items()
            ]
        )
        st.dataframe(performance_frame, hide_index=True, use_container_width=True)
    anomaly_metrics = performance.get("anomaly_detector", {})
    if anomaly_metrics:
        st.caption(
            "Autoencoder: "
            f"accuracy {anomaly_metrics['accuracy'] * 100:.2f}%, "
            f"macro-F1 {anomaly_metrics['macro_f1'] * 100:.2f}%, "
            f"threshold {anomaly_metrics['threshold']:.4f}."
        )
    st.warning(performance.get("evaluation_note", "Model metrics are preliminary."))

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

                saved_prediction = st.session_state.get("last_flow_prediction")
                if (
                    isinstance(saved_prediction, dict)
                    and saved_prediction.get("content_digest") == content_digest
                    and saved_prediction.get("row_number") == int(row_number)
                ):
                    result = saved_prediction["response"]["result"]
                    st.metric("Predicted category", result["predicted_category"])
                    st.metric(
                        "Risk",
                        f"{result['risk_level'].title()} ({result['risk_score']:.3f})",
                    )
                    st.caption(result["confidence_note"])
                    st.write("Reasons:", ", ".join(result["reasons"]))
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
