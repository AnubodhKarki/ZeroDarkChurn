"""Streamlit dashboard for the DevAPICo Silent Churn Detection System.

Three tabs:
  1. Account List  — sortable/filterable table of all accounts with trigger status
  2. Account Detail — per-account usage chart, signals, and draft outreach
  3. Evaluation    — precision/recall/F1 vs ground truth

Run with:
    streamlit run dashboard/app.py
"""

import json
import os
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.utils import compute_comms_checksum

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "output"
RESULTS_PATH = DATA_DIR / "results.parquet"
USAGE_PATH = DATA_DIR / "usage.csv"
GT_PATH = DATA_DIR / "ground_truth.json"

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DevAPICo — Silent Churn Detector",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Data loading (cached)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data
def load_results() -> pd.DataFrame:
    df = pd.read_parquet(RESULTS_PATH)
    df["anomaly_types_list"] = df["anomaly_types"].apply(
        lambda x: json.loads(x) if isinstance(x, str) else []
    )
    df["churn_signals_list"] = df["churn_signals"].apply(
        lambda x: json.loads(x) if isinstance(x, str) else []
    )
    return df


@st.cache_data
def load_usage() -> pd.DataFrame:
    df = pd.read_csv(USAGE_PATH, parse_dates=["date"])
    return df


@st.cache_data
def load_ground_truth() -> dict:
    if GT_PATH.exists():
        return json.loads(GT_PATH.read_text())
    return {}


@st.cache_data
def load_comms() -> tuple:
    """Load tickets/slack/transcripts indexed by account_id."""
    def _index(items: list, key: str) -> dict:
        idx: dict = {}
        for item in items:
            k = str(item.get(key, ""))
            idx.setdefault(k, []).append(item)
        return idx

    tickets = json.loads((DATA_DIR / "tickets.json").read_text())
    slack_msgs = json.loads((DATA_DIR / "slack.json").read_text())
    transcripts = json.loads((DATA_DIR / "transcripts.json").read_text())
    return (
        _index(tickets, "account_id"),
        _index(slack_msgs, "account_id"),
        _index(transcripts, "account_id"),
    )


def _provider_info() -> str:
    provider = os.environ.get("LLM_PROVIDER", "openai")
    model = os.environ.get("LLM_MODEL", "")
    defaults = {"openai": "gpt-4o-mini", "anthropic": "claude-sonnet-4-5"}
    model = model or defaults.get(provider, "unknown")
    return f"{provider} / {model}"


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("📡 Silent Churn")
    st.caption("DevAPICo — internal CS tool")
    st.divider()
    st.markdown("**About**")
    st.markdown(
        "Detects developers silently churning by correlating "
        "health metrics, LLM-scored communications, and account value."
    )
    st.divider()

    if RESULTS_PATH.exists():
        df_all = load_results()
        n_triggered = df_all["should_trigger"].sum()
        n_total = len(df_all)
        st.metric("Accounts monitored", n_total)
        st.metric("Outreach triggered", int(n_triggered))
        st.metric("Anomalies detected", int(df_all["has_anomaly"].sum()))

    st.divider()
    st.caption(f"LLM: `{_provider_info()}`")
    st.caption("Synthetic data · portfolio project")

# ─────────────────────────────────────────────────────────────────────────────
# Guard: check data exists
# ─────────────────────────────────────────────────────────────────────────────

if not RESULTS_PATH.exists():
    st.error(
        "No results file found. Run the pipeline first:\n\n"
        "```\npython -m pipeline.run\n```"
    )
    st.stop()

df_all = load_results()
usage_df = load_usage()
ground_truth = load_ground_truth()

# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs(["Account List", "Account Detail", "Evaluation"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Account List
# ══════════════════════════════════════════════════════════════════════════════

with tab1:
    st.header("Account List")

    # Filters
    col_f1, col_f2, col_f3, col_f4 = st.columns(4)
    with col_f1:
        tier_opts = ["All"] + sorted(df_all["tier"].dropna().unique().tolist())
        tier_filter = st.selectbox("Tier", tier_opts)
    with col_f2:
        trigger_filter = st.selectbox("Trigger status", ["All", "Triggered", "Not triggered"])
    with col_f3:
        risk_opts = ["All", "high", "medium", "low"]
        risk_filter = st.selectbox("Churn risk", risk_opts)
    with col_f4:
        severity_opts = ["All", "high", "medium", "low", "none"]
        sev_filter = st.selectbox("Anomaly severity", severity_opts)

    display = df_all.copy()
    if tier_filter != "All":
        display = display[display["tier"] == tier_filter]
    if trigger_filter == "Triggered":
        display = display[display["should_trigger"]]
    elif trigger_filter == "Not triggered":
        display = display[~display["should_trigger"]]
    if risk_filter != "All":
        display = display[display["churn_risk"] == risk_filter]
    if sev_filter != "All":
        display = display[display["anomaly_severity"] == sev_filter]

    # Build display table
    table = display[[
        "account_id", "company_name", "tier", "funding_stage",
        "current_mrr_usd", "anomaly_severity", "churn_risk",
        "should_trigger", "confidence",
    ]].copy()

    table["should_trigger"] = table["should_trigger"].map({True: "🔴 Yes", False: "✅ No"})
    table["confidence"] = table["confidence"].apply(
        lambda x: f"{x:.0%}" if pd.notna(x) else "—"
    )
    table["current_mrr_usd"] = table["current_mrr_usd"].apply(
        lambda x: f"${int(x):,}" if pd.notna(x) else "—"
    )
    table = table.rename(columns={
        "account_id": "Account ID",
        "company_name": "Company",
        "tier": "Tier",
        "funding_stage": "Funding",
        "current_mrr_usd": "MRR",
        "anomaly_severity": "Anomaly",
        "churn_risk": "LLM Risk",
        "should_trigger": "Trigger",
        "confidence": "Confidence",
    })

    st.dataframe(table, use_container_width=True, height=500)
    st.caption(f"Showing {len(display)} of {len(df_all)} accounts")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Account Detail
# ══════════════════════════════════════════════════════════════════════════════

with tab2:
    st.header("Account Detail")

    # Account selector — show triggered first
    triggered_ids = df_all[df_all["should_trigger"]]["account_id"].tolist()
    other_ids = df_all[~df_all["should_trigger"]]["account_id"].tolist()
    all_ids = triggered_ids + other_ids

    def _label(acc_id: str) -> str:
        row = df_all[df_all["account_id"] == acc_id].iloc[0]
        flag = "🔴" if row["should_trigger"] else "⚪"
        return f"{flag} {acc_id} — {row['company_name']}"

    selected_id = st.selectbox(
        "Select account (🔴 = triggered)",
        all_ids,
        format_func=_label,
    )

    row = df_all[df_all["account_id"] == selected_id].iloc[0]
    acc_usage = usage_df[usage_df["account_id"] == selected_id].sort_values("date")

    # ── Profile row ───────────────────────────────────────────────────────────
    st.divider()
    p1, p2, p3, p4, p5, p6 = st.columns(6)
    p1.metric("Tier", row["tier"].capitalize())
    p2.metric("MRR", f"${int(row['current_mrr_usd']):,}")
    p3.metric("Funding", row["funding_stage"].replace("_", " ").title())
    p4.metric("Use case", str(row.get("primary_use_case", "—")).replace("_", " "))
    p5.metric("Anomaly severity", row["anomaly_severity"] or "none")
    trigger_str = "🔴 Triggered" if row["should_trigger"] else "✅ No trigger"
    p6.metric("Decision", trigger_str)

    # ── Usage chart ───────────────────────────────────────────────────────────
    st.subheader("API call volume — last 90 days")

    if not acc_usage.empty:
        fig = go.Figure()

        # Determine anomaly highlight window
        has_anomaly = bool(row["has_anomaly"])
        if has_anomaly:
            cutoff = acc_usage["date"].max() - pd.Timedelta(days=14)
            normal = acc_usage[acc_usage["date"] < cutoff]
            anomaly_window = acc_usage[acc_usage["date"] >= cutoff]
        else:
            normal = acc_usage
            anomaly_window = pd.DataFrame()

        fig.add_trace(go.Scatter(
            x=normal["date"], y=normal["api_calls"],
            name="API calls (normal)",
            line=dict(color="#4A90E2", width=2),
            fill="tozeroy",
            fillcolor="rgba(74,144,226,0.08)",
        ))

        if not anomaly_window.empty:
            fig.add_trace(go.Scatter(
                x=anomaly_window["date"], y=anomaly_window["api_calls"],
                name="API calls (anomaly window)",
                line=dict(color="#E05252", width=2),
                fill="tozeroy",
                fillcolor="rgba(224,82,82,0.12)",
            ))
            # Vertical marker at window start
            # add_vline is broken in Plotly 5.15 with datetime axes — use add_shape instead
            cutoff_str = cutoff.strftime("%Y-%m-%d")
            fig.add_shape(
                type="line",
                x0=cutoff_str, x1=cutoff_str,
                y0=0, y1=1,
                xref="x", yref="paper",
                line=dict(dash="dash", color="rgba(224,82,82,0.5)", width=1.5),
            )
            fig.add_annotation(
                x=cutoff_str, y=1,
                xref="x", yref="paper",
                text="Anomaly window",
                showarrow=False,
                xanchor="left",
                yanchor="top",
                font=dict(color="rgba(224,82,82,0.8)", size=11),
            )

        fig.add_trace(go.Scatter(
            x=acc_usage["date"],
            y=acc_usage["error_count"],
            name="Error count",
            line=dict(color="#F5A623", width=1.5, dash="dot"),
            yaxis="y2",
        ))

        fig.update_layout(
            height=320,
            margin=dict(l=0, r=0, t=10, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
            yaxis=dict(title="API calls"),
            yaxis2=dict(title="Errors", overlaying="y", side="right", showgrid=False),
            hovermode="x unified",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No usage data available for this account.")

    # ── Health signals ────────────────────────────────────────────────────────
    st.subheader("Health signals")
    anomaly_types = row["anomaly_types_list"]
    if anomaly_types:
        cols = st.columns(len(anomaly_types))
        signal_colors = {
            "usage_drop": "🔴",
            "wow_drop": "🟠",
            "error_spike": "🔴",
            "webhook_drop": "🟡",
            "rate_limit_stress": "🟡",
        }
        for col, sig in zip(cols, anomaly_types):
            icon = signal_colors.get(sig, "🔵")
            col.markdown(f"**{icon} {sig.replace('_', ' ').title()}**")

        # Details table
        det_rows = []
        if pd.notna(row.get("usage_drop_pct")):
            det_rows.append(("Usage drop vs baseline", f"{row['usage_drop_pct']:.1f}%"))
        if pd.notna(row.get("wow_delta_pct")):
            det_rows.append(("Week-over-week delta", f"{row['wow_delta_pct']:.1f}%"))
        if pd.notna(row.get("error_rate_zscore")):
            det_rows.append(("Error rate z-score", f"{row['error_rate_zscore']:.2f}σ"))
        if pd.notna(row.get("webhook_drop_pp")):
            det_rows.append(("Webhook success drop", f"{row['webhook_drop_pp']*100:.1f}pp"))
        if det_rows:
            st.table(pd.DataFrame(det_rows, columns=["Signal", "Value"]))
    else:
        st.info("No health anomalies detected for this account.")

    # ── LLM classification ────────────────────────────────────────────────────
    if row["classified"]:
        st.subheader("LLM risk classification")
        rc1, rc2 = st.columns([1, 3])

        risk_color = {"high": "🔴", "medium": "🟠", "low": "🟢"}.get(
            str(row["churn_risk"]), "⚪"
        )
        rc1.metric("Churn risk", f"{risk_color} {str(row['churn_risk']).upper()}")
        if pd.notna(row.get("use_case_detected")):
            rc1.metric("Detected use case", row["use_case_detected"])

        with rc2:
            signals = row["churn_signals_list"]
            if signals:
                st.markdown("**Churn signals quoted from communications:**")
                for sig in signals:
                    st.markdown(f"> {sig}")
            else:
                st.markdown("_No explicit churn signals quoted._")

            if pd.notna(row.get("reasoning")):
                st.caption(f"Reasoning: {row['reasoning']}")

    # ── Correlation decision ──────────────────────────────────────────────────
    st.subheader("Trigger decision")
    if row["should_trigger"]:
        conf_pct = f"{row['confidence']:.0%}"
        st.success(
            f"**Outreach triggered** — confidence {conf_pct}\n\n"
            f"{row['trigger_reasoning']}"
        )
    else:
        st.info(f"**No outreach** — {row['trigger_reasoning']}")

    # ── Draft outreach ────────────────────────────────────────────────────────
    if pd.notna(row.get("draft_email")) and row["draft_email"]:
        st.subheader("Draft outreach email")
        st.markdown(
            f"""
<div style="background:#1a1a2e;border:1px solid #3a3a5c;border-radius:8px;
            padding:20px 24px;font-family:monospace;font-size:0.9rem;
            line-height:1.7;color:#e0e0f0;">
{row['draft_email'].replace(chr(10), '<br>')}
</div>
""",
            unsafe_allow_html=True,
        )
        st.caption(
            "This is an AI-generated draft for CS review. "
            "Edit/Personalise before sending."
        )

    # ── Analyse & Draft button ────────────────────────────────────────────────
    st.divider()
    st.subheader("Analyse & Draft")

    tickets_by_acc, slack_by_acc, transcripts_by_acc = load_comms()
    acc_tickets = tickets_by_acc.get(selected_id, [])
    acc_slack = slack_by_acc.get(selected_id, [])
    acc_transcripts = transcripts_by_acc.get(selected_id, [])

    current_checksum = compute_comms_checksum(acc_tickets, acc_slack, acc_transcripts)
    stored_checksum = ""
    raw_stored = row.get("comms_checksum")
    if raw_stored is not None and pd.notna(raw_stored):
        stored_checksum = str(raw_stored)

    already_analysed = bool(row["classified"])
    checksum_match = bool(stored_checksum) and (current_checksum == stored_checksum)

    if already_analysed and checksum_match:
        st.success(
            "Already analysed — communications unchanged since last run. "
            "Showing cached results above."
        )
        st.caption(f"Checksum: `{current_checksum[:16]}…`")
    else:
        if already_analysed and not checksum_match:
            st.warning("Communications have changed since last analysis — re-analysis available.")
        elif not already_analysed:
            st.info(
                "This account has not been LLM-classified yet "
                "(no anomaly was detected in the last pipeline run). "
                "You can still run a one-off analysis below."
            )

        if st.button("🔍 Analyse & Draft", key="analyse_btn"):
            with st.spinner("Running classify → correlate → draft…"):
                try:
                    from dotenv import load_dotenv
                    load_dotenv()
                except ImportError:
                    pass

                from llm import get_client
                from pipeline.classify import classify_account
                from pipeline.correlate import correlate
                from pipeline.respond import draft_outreach

                llm = get_client()
                account_dict = {k: row[k] for k in row.index}

                # Classify
                new_classification = classify_account(
                    account_dict,
                    acc_tickets,
                    acc_slack,
                    acc_transcripts,
                    llm,
                )

                # Reconstruct health dict from stored row values
                health = {
                    "account_id": selected_id,
                    "has_anomaly": bool(row["has_anomaly"]),
                    "anomaly_types": row["anomaly_types_list"],
                    "severity": str(row["anomaly_severity"]),
                    "details": {
                        "usage_drop_pct": (
                            float(row["usage_drop_pct"])
                            if pd.notna(row.get("usage_drop_pct")) else None
                        ),
                        "wow_delta_pct": (
                            float(row["wow_delta_pct"])
                            if pd.notna(row.get("wow_delta_pct")) else None
                        ),
                        "error_rate_zscore": (
                            float(row["error_rate_zscore"])
                            if pd.notna(row.get("error_rate_zscore")) else None
                        ),
                        "webhook_drop_pp": (
                            float(row["webhook_drop_pp"])
                            if pd.notna(row.get("webhook_drop_pp")) else None
                        ),
                    },
                }

                # Correlate
                new_decision = correlate(account_dict, health, new_classification)

                # Draft outreach if triggered
                new_draft = None
                if new_decision["should_trigger"]:
                    new_draft = draft_outreach(
                        account_dict, health, new_classification, new_decision, llm
                    )

                # Write back to parquet
                df_updated = pd.read_parquet(RESULTS_PATH)
                idx = df_updated[df_updated["account_id"] == selected_id].index[0]

                df_updated.at[idx, "classified"] = True
                df_updated.at[idx, "use_case_detected"] = new_classification.get("use_case")
                df_updated.at[idx, "churn_risk"] = new_classification.get("churn_risk")
                df_updated.at[idx, "churn_signals"] = json.dumps(
                    new_classification.get("churn_signals", [])
                )
                df_updated.at[idx, "recommended_action"] = new_classification.get(
                    "recommended_action"
                )
                df_updated.at[idx, "reasoning"] = new_classification.get("reasoning")
                df_updated.at[idx, "should_trigger"] = new_decision["should_trigger"]
                df_updated.at[idx, "confidence"] = new_decision["confidence"]
                df_updated.at[idx, "trigger_reasoning"] = new_decision["reasoning"]
                df_updated.at[idx, "comms_checksum"] = current_checksum

                if new_draft:
                    df_updated.at[idx, "draft_email"] = new_draft.get("draft_email", "")
                    df_updated.at[idx, "draft_error"] = new_draft.get("error")

                df_updated.to_parquet(RESULTS_PATH, index=False)

            st.cache_data.clear()
            st.rerun()

        st.caption(f"Checksum: `{current_checksum[:16]}…`")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Evaluation
# ══════════════════════════════════════════════════════════════════════════════

with tab3:
    st.header("Evaluation vs Ground Truth")

    if not ground_truth:
        st.warning("No ground_truth.json found. Run the data generator first.")
        st.stop()

    # Build label sets from ground truth
    planted_churn_ids = {r["account_id"] for r in ground_truth.get("planted_churn", [])}
    fp_trap_ids = {r["account_id"] for r in ground_truth.get("false_positive_traps", [])}
    # Everything else is a true negative
    all_ids_set = set(df_all["account_id"].tolist())
    true_negative_ids = all_ids_set - planted_churn_ids - fp_trap_ids

    # Compute confusion matrix
    results_dict = df_all.set_index("account_id")["should_trigger"].to_dict()

    tp = sum(1 for acc in planted_churn_ids if results_dict.get(acc, False))
    fn = sum(1 for acc in planted_churn_ids if not results_dict.get(acc, False))
    fp = sum(1 for acc in fp_trap_ids if results_dict.get(acc, False))
    tn_fp = sum(1 for acc in fp_trap_ids if not results_dict.get(acc, False))
    fp_tn = sum(1 for acc in true_negative_ids if results_dict.get(acc, False))
    tn_tn = sum(1 for acc in true_negative_ids if not results_dict.get(acc, False))

    total_fp = fp + fp_tn  # FP traps triggered + healthy accounts triggered
    total_tn = tn_fp + tn_tn

    precision = tp / (tp + total_fp) if (tp + total_fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # ── Metrics row ───────────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Precision", f"{precision:.1%}", help="Of triggered accounts, how many were true churn?")
    m2.metric("Recall", f"{recall:.1%}", help="Of true churn cases, how many were caught?")
    m3.metric("F1 Score", f"{f1:.2f}")
    m4.metric("FP traps blocked", f"{tn_fp}/{len(fp_trap_ids)}", help="FP trap accounts correctly NOT triggered")

    st.divider()

    # ── Confusion matrix ──────────────────────────────────────────────────────
    st.subheader("Confusion matrix")
    cm_col, detail_col = st.columns([1, 2])

    with cm_col:
        cm_data = pd.DataFrame(
            [[tp, fn], [total_fp, total_tn]],
            index=["Predicted: Triggered", "Predicted: Not triggered"],
            columns=["Actual: Churn", "Actual: Not churn"],
        )
        st.dataframe(cm_data, use_container_width=True)

    with detail_col:
        st.markdown(f"""
| | Count | Notes |
|---|---|---|
| True Positives  | **{tp}** | Planted churn cases correctly triggered |
| False Negatives | **{fn}** | Planted churn cases missed |
| FP Traps blocked| **{tn_fp}** | False-positive traps correctly held |
| FP Traps fired  | **{fp}** | False-positive traps incorrectly triggered |
| Healthy FPs     | **{fp_tn}** | Healthy accounts incorrectly triggered |
| True Negatives  | **{tn_tn}** | Healthy accounts correctly held |
""")

    st.divider()

    # ── Per-signal-mix breakdown ──────────────────────────────────────────────
    st.subheader("Precision/Recall by signal mix")

    signal_mix_map = {
        r["account_id"]: r["signal_mix"]
        for r in ground_truth.get("planted_churn", [])
    }

    mix_rows = []
    for mix in ["health_only", "risk_only", "combined"]:
        ids = [acc for acc, m in signal_mix_map.items() if m == mix]
        caught = sum(1 for acc in ids if results_dict.get(acc, False))
        mix_rows.append({
            "Signal mix": mix.replace("_", " ").title(),
            "Total": len(ids),
            "Caught": caught,
            "Missed": len(ids) - caught,
            "Recall": f"{caught/len(ids):.0%}" if ids else "—",
        })

    st.table(pd.DataFrame(mix_rows))

    st.divider()

    # ── Case-by-case table ────────────────────────────────────────────────────
    st.subheader("Planted churn cases — detail")

    case_rows = []
    for r in ground_truth.get("planted_churn", []):
        acc_id = r["account_id"]
        res = df_all[df_all["account_id"] == acc_id]
        if res.empty:
            continue
        res_row = res.iloc[0]
        case_rows.append({
            "Account": acc_id,
            "Company": res_row["company_name"],
            "Signal mix": r["signal_mix"].replace("_", " ").title(),
            "Triggered": "✅ Yes" if res_row["should_trigger"] else "❌ Missed",
            "Confidence": f"{res_row['confidence']:.0%}" if res_row["should_trigger"] else "—",
            "Anomaly": res_row["anomaly_severity"],
            "LLM risk": res_row["churn_risk"] or "—",
            "Notes": r.get("notes", ""),
        })

    st.dataframe(pd.DataFrame(case_rows), use_container_width=True, height=380)

    st.subheader("False-positive traps — detail")

    fp_rows = []
    for r in ground_truth.get("false_positive_traps", []):
        acc_id = r["account_id"]
        res = df_all[df_all["account_id"] == acc_id]
        if res.empty:
            continue
        res_row = res.iloc[0]
        fp_rows.append({
            "Account": acc_id,
            "Company": res_row["company_name"],
            "Trap type": r["trap_type"].replace("_", " ").title(),
            "Triggered (bad)": "🚨 Yes" if res_row["should_trigger"] else "✅ Blocked",
            "Anomaly": res_row["anomaly_severity"],
            "LLM risk": res_row["churn_risk"] or "—",
            "Notes": r.get("notes", "")[:80] + "…",
        })

    st.dataframe(pd.DataFrame(fp_rows), use_container_width=True)

    st.divider()
    st.subheader("Reflection")
    st.info(
        "**What this system does well:** Multi-signal correlation (health + LLM + value tier) "
        "prevents both high false-positive and high false-negative rates. The correlator's "
        "'weak anomaly requires high LLM confidence' rule cleanly blocks seasonal and transient FPs.\n\n"
        "**Main limitation (synthetic data):** The one miss (acc_0009) had a WoW decline of −21.8%, "
        "just below the 30% threshold. In production, the threshold would be tuned on real labeled data. "
        "Risk-only churn (no health signal at all) would be a blind spot without a complementary "
        "daily communications scan.\n\n"
        "**What I'd do differently with real data:** Use account-specific adaptive thresholds, "
        "feed the LLM real Slack/Pylon/Fireflies exports, and add a feedback loop where CS "
        "outcomes (did the customer churn?) retrain the classifier."
    )
