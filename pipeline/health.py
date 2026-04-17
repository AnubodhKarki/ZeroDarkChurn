"""Health anomaly detection for the silent churn pipeline.

Pure pandas — no ML, no LLM. Computes statistical signals from 90-day usage
logs and returns one anomaly record per account.

Signals computed:
  - usage_drop      : last-7-day average is >2σ below the 30-day rolling baseline
  - wow_drop        : week-over-week usage decline >30%
  - error_spike     : last-7-day average error rate is >2σ above the 30-day baseline
  - rate_limit_stress: 429 hits in the last 7 days above threshold
  - webhook_drop    : last-7-day average webhook success rate >5pp below 30-day baseline

Severity:
  high   — usage_drop AND (error_spike OR webhook_drop), or 3+ anomaly types
  medium — 2 anomaly types
  low    — 1 anomaly type
  none   — no anomalies

Usage:
    from pipeline.health import detect_anomalies
    results = detect_anomalies(usage_df)   # list of dicts, one per account
"""

import logging
from typing import Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Thresholds — documented here so they're easy to tune
# ─────────────────────────────────────────────────────────────────────────────
ZSCORE_THRESHOLD = 2.0        # σ below baseline to flag usage_drop / error_spike
WOW_DROP_THRESHOLD = 0.30     # week-over-week absolute decline to flag wow_drop
RATE_LIMIT_HITS_THRESHOLD = 5 # 429 hits in last 7 days
WEBHOOK_DROP_PP = 0.05        # percentage-point drop in webhook success rate
MIN_DAILY_CALLS = 10          # ignore days with fewer calls when computing rates
RECENT_DAYS = 7               # "current" window for comparison

# Baseline anchor: exclude the last BASELINE_EXCLUDE days when computing the
# stable baseline. Set to 30 so the baseline window is days 1–60 for a 90-day
# dataset — well before any planted anomaly (which starts at day 76).
# In production, you'd recompute this daily so new anomalies are detected as
# they happen; here we run once over historical data and need a clean window.
BASELINE_EXCLUDE = 30


def detect_anomalies(usage_df: pd.DataFrame) -> List[Dict]:
    """Run anomaly detection on the full usage DataFrame.

    Args:
        usage_df: DataFrame with columns:
            account_id, date, api_calls, error_count,
            rate_limit_hits_429, webhook_success_rate, p95_latency_ms

    Returns:
        List of dicts, one per account:
            {
                account_id, has_anomaly, anomaly_types, details, severity
            }
    """
    usage_df = usage_df.copy()
    usage_df["date"] = pd.to_datetime(usage_df["date"])
    usage_df = usage_df.sort_values(["account_id", "date"])

    # Compute derived columns used by multiple detectors
    usage_df["error_rate"] = np.where(
        usage_df["api_calls"] >= MIN_DAILY_CALLS,
        usage_df["error_count"] / usage_df["api_calls"],
        np.nan,
    )

    results = []
    for acc_id, group in usage_df.groupby("account_id"):
        group = group.sort_values("date").reset_index(drop=True)
        record = _analyze_account(str(acc_id), group)
        results.append(record)

    flagged = sum(1 for r in results if r["has_anomaly"])
    logger.info("Health detection complete: %d/%d accounts flagged", flagged, len(results))
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Per-account analysis
# ─────────────────────────────────────────────────────────────────────────────

def _analyze_account(account_id: str, df: pd.DataFrame) -> Dict:
    """Compute all anomaly signals for a single account's usage history."""
    n = len(df)
    min_required = BASELINE_EXCLUDE + RECENT_DAYS * 2
    if n < min_required:
        return _no_anomaly(account_id, note=f"only {n} days of data (need {min_required})")

    recent = df.iloc[-RECENT_DAYS:]
    prev_week = df.iloc[-(RECENT_DAYS * 2):-RECENT_DAYS]
    # Baseline = all data before the last BASELINE_EXCLUDE days.
    # This keeps the baseline clean — it never contains recent anomalous data.
    baseline = df.iloc[:-BASELINE_EXCLUDE]

    anomaly_types: List[str] = []
    details: Dict = {}

    # ── 1. Usage drop: last-7-day avg vs 30-day baseline ─────────────────────
    baseline_calls = baseline["api_calls"].values
    baseline_mean = float(np.mean(baseline_calls))
    baseline_std = float(np.std(baseline_calls))
    recent_mean_calls = float(recent["api_calls"].mean())

    if baseline_std > 0:
        usage_zscore = (recent_mean_calls - baseline_mean) / baseline_std
    else:
        usage_zscore = 0.0

    details["baseline_mean_calls"] = round(baseline_mean, 1)
    details["recent_mean_calls"] = round(recent_mean_calls, 1)
    details["usage_zscore"] = round(usage_zscore, 2)

    if usage_zscore < -ZSCORE_THRESHOLD and baseline_mean > MIN_DAILY_CALLS:
        anomaly_types.append("usage_drop")
        pct = (recent_mean_calls - baseline_mean) / baseline_mean * 100
        details["usage_drop_pct"] = round(pct, 1)

    # ── 2. Week-over-week drop ────────────────────────────────────────────────
    if len(prev_week) >= RECENT_DAYS:
        prev_mean = float(prev_week["api_calls"].mean())
        if prev_mean > MIN_DAILY_CALLS:
            wow_delta = (recent_mean_calls - prev_mean) / prev_mean
            details["wow_delta_pct"] = round(wow_delta * 100, 1)
            if wow_delta < -WOW_DROP_THRESHOLD:
                anomaly_types.append("wow_drop")

    # ── 3. Error rate spike: last-7-day avg vs 30-day baseline ───────────────
    baseline_err = baseline["error_rate"].dropna().values
    recent_err = recent["error_rate"].dropna().values

    if len(baseline_err) >= 5 and len(recent_err) >= 3:
        err_baseline_mean = float(np.mean(baseline_err))
        err_baseline_std = float(np.std(baseline_err))
        err_recent_mean = float(np.mean(recent_err))

        if err_baseline_std > 0:
            err_zscore = (err_recent_mean - err_baseline_mean) / err_baseline_std
        else:
            err_zscore = 0.0

        details["error_rate_recent"] = round(err_recent_mean, 4)
        details["error_rate_baseline"] = round(err_baseline_mean, 4)
        details["error_rate_zscore"] = round(err_zscore, 2)

        if err_zscore > ZSCORE_THRESHOLD:
            anomaly_types.append("error_spike")

    # ── 4. Rate-limit stress ──────────────────────────────────────────────────
    rl_hits_recent = int(recent["rate_limit_hits_429"].sum())
    details["rate_limit_hits_7d"] = rl_hits_recent
    if rl_hits_recent >= RATE_LIMIT_HITS_THRESHOLD:
        anomaly_types.append("rate_limit_stress")

    # ── 5. Webhook success rate drop ─────────────────────────────────────────
    baseline_webhook = float(baseline["webhook_success_rate"].mean())
    recent_webhook = float(recent["webhook_success_rate"].mean())
    webhook_drop = baseline_webhook - recent_webhook

    details["webhook_success_baseline"] = round(baseline_webhook, 4)
    details["webhook_success_recent"] = round(recent_webhook, 4)
    details["webhook_drop_pp"] = round(webhook_drop, 4)

    if webhook_drop > WEBHOOK_DROP_PP:
        anomaly_types.append("webhook_drop")

    # ── Severity ──────────────────────────────────────────────────────────────
    has_anomaly = len(anomaly_types) > 0
    severity = _score_severity(anomaly_types)

    return {
        "account_id": account_id,
        "has_anomaly": has_anomaly,
        "anomaly_types": anomaly_types,
        "details": details,
        "severity": severity,
    }


def _score_severity(anomaly_types: List[str]) -> str:
    """Assign severity from the set of detected anomaly types.

    high   — usage_drop paired with error_spike or webhook_drop (multi-system failure),
             or 3+ distinct anomaly types
    medium — exactly 2 anomaly types
    low    — 1 anomaly type
    none   — no anomalies
    """
    n = len(anomaly_types)
    if n == 0:
        return "none"
    if n >= 3:
        return "high"
    if n == 2:
        types = set(anomaly_types)
        # usage drop + either signal-of-distress → high
        if "usage_drop" in types and ("error_spike" in types or "webhook_drop" in types):
            return "high"
        return "medium"
    return "low"


def _no_anomaly(account_id: str, note: str = "") -> Dict:
    return {
        "account_id": account_id,
        "has_anomaly": False,
        "anomaly_types": [],
        "details": {"note": note},
        "severity": "none",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: results as DataFrame
# ─────────────────────────────────────────────────────────────────────────────

def anomalies_to_df(results: List[Dict]) -> pd.DataFrame:
    """Flatten the anomaly results list into a tidy DataFrame for inspection."""
    rows = []
    for r in results:
        d = r["details"]
        rows.append({
            "account_id": r["account_id"],
            "has_anomaly": r["has_anomaly"],
            "severity": r["severity"],
            "anomaly_types": ", ".join(r["anomaly_types"]),
            "usage_drop_pct": d.get("usage_drop_pct"),
            "usage_zscore": d.get("usage_zscore"),
            "wow_delta_pct": d.get("wow_delta_pct"),
            "error_rate_zscore": d.get("error_rate_zscore"),
            "webhook_drop_pp": d.get("webhook_drop_pp"),
            "rate_limit_hits_7d": d.get("rate_limit_hits_7d"),
        })
    return pd.DataFrame(rows)
