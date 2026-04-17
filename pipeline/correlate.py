"""Correlation engine for the silent churn pipeline.

Combines health anomaly data, LLM classification output, and account profile
into a single trigger decision. No LLM calls here — pure business rules.

Business decisions documented inline:
  - Free-tier accounts are skipped even when churning (low revenue impact;
    CS bandwidth is better spent on paying accounts).
  - Risk-only anomalies (wow_drop without error_spike/webhook_drop) require
    high churn_risk from the LLM to trigger — avoiding noise from healthy
    accounts that happen to have a slow week.
  - Confidence is a weighted composite of anomaly severity, LLM churn risk,
    and account value tier.

Usage:
    from pipeline.correlate import correlate
    decision = correlate(account_row, health_result, classification_result)
"""

import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Tier weights for the confidence score.
# Business decision: enterprise/growth are higher priority for outreach.
TIER_WEIGHT = {
    "free":       0.0,   # never triggered — separate business rule
    "starter":    0.4,
    "growth":     0.7,
    "enterprise": 1.0,
}

SEVERITY_WEIGHT = {
    "none":   0.0,
    "low":    0.3,
    "medium": 0.6,
    "high":   1.0,
}

RISK_WEIGHT = {
    "low":    0.0,
    "medium": 0.5,
    "high":   1.0,
}


def correlate(
    account: Dict,
    health: Dict,
    classification: Optional[Dict],
) -> Dict:
    """Combine signals and decide whether to trigger outreach for one account.

    Args:
        account:        One row from accounts.csv as a dict.
        health:         Output from pipeline.health.detect_anomalies (one account).
        classification: Output from pipeline.classify.classify_account, or None
                        if classification was not run (no health anomaly).

    Returns:
        Dict with keys:
            account_id, should_trigger, confidence, reasoning,
            tier, churn_risk, anomaly_severity, anomaly_types
    """
    acc_id = str(account.get("account_id", ""))
    tier = str(account.get("tier", "free"))
    has_anomaly = bool(health.get("has_anomaly", False))
    severity = str(health.get("severity", "none"))
    anomaly_types: list = health.get("anomaly_types", [])

    churn_risk = "low"
    if classification:
        churn_risk = str(classification.get("churn_risk", "low"))

    # ── Business rule: skip free tier entirely ────────────────────────────────
    # Free accounts generate no revenue; CS time is finite.
    if tier == "free":
        return _decision(
            acc_id, False, 0.0,
            "Free tier — not prioritised for proactive outreach.",
            tier, churn_risk, severity, anomaly_types,
        )

    # ── No anomaly at all — no trigger ───────────────────────────────────────
    if not has_anomaly:
        return _decision(
            acc_id, False, 0.0,
            "No health anomaly detected.",
            tier, churn_risk, severity, anomaly_types,
        )

    # ── Weak anomaly (single signal) needs high LLM confidence to trigger ────
    # A lone wow_drop or error_spike could be seasonal, transient, or noise.
    # Only trigger if the LLM also sees a strong risk signal.
    weak_anomaly = (severity == "low") or (
        len(anomaly_types) == 1 and anomaly_types[0] in ("wow_drop", "error_spike")
    )
    if weak_anomaly and churn_risk != "high":
        return _decision(
            acc_id, False,
            _confidence(severity, churn_risk, tier),
            (
                f"Weak anomaly ({', '.join(anomaly_types)}) with only {churn_risk} LLM risk — "
                "likely seasonal, transient, or noise. No outreach triggered."
            ),
            tier, churn_risk, severity, anomaly_types,
        )

    # ── Multi-signal or high-confidence: trigger ──────────────────────────────
    if churn_risk in ("medium", "high"):
        conf = _confidence(severity, churn_risk, tier)
        signals_str = ", ".join(anomaly_types) if anomaly_types else "none"
        llm_signals = (
            classification.get("churn_signals", []) if classification else []
        )
        signal_quote = f' ("{llm_signals[0]}")' if llm_signals else ""
        reasoning = (
            f"{severity.capitalize()}-severity anomaly ({signals_str}) + "
            f"{churn_risk} churn risk{signal_quote} + {tier} tier."
        )
        return _decision(acc_id, True, conf, reasoning, tier, churn_risk, severity, anomaly_types)

    # ── Anomaly present but LLM says low risk — hold ─────────────────────────
    return _decision(
        acc_id, False,
        _confidence(severity, churn_risk, tier),
        (
            f"Anomaly detected ({', '.join(anomaly_types)}) but LLM classified risk as low. "
            "Account likely healthy — no outreach triggered."
        ),
        tier, churn_risk, severity, anomaly_types,
    )


def _confidence(severity: str, churn_risk: str, tier: str) -> float:
    """Composite 0–1 confidence score from the three signal axes."""
    s = SEVERITY_WEIGHT.get(severity, 0.0)
    r = RISK_WEIGHT.get(churn_risk, 0.0)
    t = TIER_WEIGHT.get(tier, 0.0)
    # Weighted average: health signals (40%), LLM risk (40%), tier value (20%)
    raw = 0.40 * s + 0.40 * r + 0.20 * t
    return round(min(max(raw, 0.0), 1.0), 3)


def _decision(
    account_id: str,
    should_trigger: bool,
    confidence: float,
    reasoning: str,
    tier: str,
    churn_risk: str,
    anomaly_severity: str,
    anomaly_types: list,
) -> Dict:
    return {
        "account_id": account_id,
        "should_trigger": should_trigger,
        "confidence": confidence,
        "reasoning": reasoning,
        "tier": tier,
        "churn_risk": churn_risk,
        "anomaly_severity": anomaly_severity,
        "anomaly_types": anomaly_types,
    }
