"""Unit tests for pipeline components — no LLM calls, no file I/O."""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ─────────────────────────────────────────────────────────────────────────────
# Health anomaly detection
# ─────────────────────────────────────────────────────────────────────────────

from pipeline.health import detect_anomalies


def _make_usage_df(
    account_id: str,
    n_days: int = 90,
    baseline_calls: int = 1000,
    drop_start: int = 76,
    drop_multiplier: float = 0.2,
    error_rate: float = 0.01,
    error_rate_drop: float = 0.01,  # error rate during normal period
    error_rate_spike: float = 0.01,  # error rate during drop period
    webhook_success: float = 0.99,
) -> pd.DataFrame:
    """Build a synthetic daily usage DataFrame for one account."""
    import random
    random.seed(42)

    rows = []
    base_date = pd.Timestamp("2024-01-01")
    for day in range(n_days):
        date = base_date + pd.Timedelta(days=day)
        in_drop = day >= drop_start
        calls = int(baseline_calls * (drop_multiplier if in_drop else 1.0))
        calls += random.randint(-20, 20)
        err_rate = error_rate_spike if in_drop else error_rate_drop
        err_count = max(0, int(calls * err_rate) + random.randint(-2, 2))
        wh_attempts = 50
        wh_success = int(wh_attempts * webhook_success)
        rows.append({
            "account_id": account_id,
            "date": date,
            "api_calls": max(0, calls),
            "error_count": err_count,
            "error_rate": err_rate,
            "webhook_attempts": wh_attempts,
            "webhook_success_count": wh_success,
            "webhook_success_rate": wh_success / wh_attempts,
            "rate_limit_hits_429": 0,
            "unique_endpoints": 3,
        })
    return pd.DataFrame(rows)


class TestHealthDetection:
    def test_healthy_account_no_anomaly(self):
        df = _make_usage_df("acc_healthy", drop_multiplier=1.0)
        results = detect_anomalies(df)
        assert len(results) == 1
        r = results[0]
        assert r["account_id"] == "acc_healthy"
        assert r["has_anomaly"] is False
        assert r["anomaly_types"] == []
        assert r["severity"] == "none"

    def test_usage_drop_detected(self):
        df = _make_usage_df("acc_drop", drop_multiplier=0.15)
        results = detect_anomalies(df)
        r = results[0]
        assert r["has_anomaly"] is True
        assert "usage_drop" in r["anomaly_types"]

    def test_high_severity_requires_multiple_signals(self):
        # usage_drop + error_spike → high severity
        df = _make_usage_df(
            "acc_high",
            drop_multiplier=0.15,
            error_rate_drop=0.02,
            error_rate_spike=0.35,
        )
        results = detect_anomalies(df)
        r = results[0]
        assert r["has_anomaly"] is True
        assert r["severity"] in ("high", "medium")

    def test_multiple_accounts(self):
        df_a = _make_usage_df("acc_a", drop_multiplier=0.15)
        df_b = _make_usage_df("acc_b", drop_multiplier=1.0)
        combined = pd.concat([df_a, df_b], ignore_index=True)
        results = detect_anomalies(combined)
        assert len(results) == 2
        by_id = {r["account_id"]: r for r in results}
        assert by_id["acc_a"]["has_anomaly"] is True
        assert by_id["acc_b"]["has_anomaly"] is False

    def test_result_keys_present(self):
        df = _make_usage_df("acc_keys")
        results = detect_anomalies(df)
        r = results[0]
        for key in ("account_id", "has_anomaly", "anomaly_types", "severity", "details"):
            assert key in r, f"Missing key: {key}"


# ─────────────────────────────────────────────────────────────────────────────
# Correlator business rules
# ─────────────────────────────────────────────────────────────────────────────

from pipeline.correlate import correlate


def _account(tier: str = "growth", mrr: int = 2000) -> dict:
    return {"account_id": "test_acc", "tier": tier, "current_mrr_usd": mrr}


def _health(has_anomaly: bool = True, severity: str = "high", types: list = None) -> dict:
    return {
        "account_id": "test_acc",
        "has_anomaly": has_anomaly,
        "severity": severity,
        "anomaly_types": types or ["usage_drop", "error_spike"],
    }


def _classification(risk: str = "high", signals: list = None) -> dict:
    return {
        "churn_risk": risk,
        "churn_signals": signals or ["we are switching"],
        "use_case": "transcription",
        "recommended_action": "Schedule a call.",
        "reasoning": "High risk detected.",
    }


class TestCorrelate:
    def test_free_tier_never_triggers(self):
        d = correlate(_account(tier="free"), _health(), _classification())
        assert d["should_trigger"] is False
        assert "Free tier" in d["reasoning"]

    def test_no_anomaly_no_trigger(self):
        d = correlate(_account(), _health(has_anomaly=False), None)
        assert d["should_trigger"] is False
        assert "No health anomaly" in d["reasoning"]

    def test_strong_multi_signal_triggers(self):
        d = correlate(_account(), _health(severity="high"), _classification(risk="high"))
        assert d["should_trigger"] is True
        assert d["confidence"] > 0.5

    def test_weak_single_signal_low_risk_blocked(self):
        d = correlate(
            _account(),
            _health(severity="low", types=["wow_drop"]),
            _classification(risk="low"),
        )
        assert d["should_trigger"] is False

    def test_weak_single_signal_high_risk_triggers(self):
        # Single weak signal but LLM says high risk → should trigger
        d = correlate(
            _account(),
            _health(severity="low", types=["wow_drop"]),
            _classification(risk="high"),
        )
        assert d["should_trigger"] is True

    def test_anomaly_low_risk_no_trigger(self):
        d = correlate(
            _account(),
            _health(severity="high", types=["usage_drop", "error_spike"]),
            _classification(risk="low"),
        )
        assert d["should_trigger"] is False

    def test_no_classification_no_trigger(self):
        # classification is None (no anomaly at health stage) → no trigger
        d = correlate(_account(), _health(has_anomaly=False), None)
        assert d["should_trigger"] is False

    def test_confidence_in_range(self):
        d = correlate(_account(), _health(), _classification())
        assert 0.0 <= d["confidence"] <= 1.0

    def test_enterprise_tier_boosts_confidence(self):
        d_ent = correlate(_account(tier="enterprise"), _health(), _classification(risk="medium"))
        d_str = correlate(_account(tier="starter"), _health(), _classification(risk="medium"))
        assert d_ent["confidence"] >= d_str["confidence"]

    def test_result_keys_present(self):
        d = correlate(_account(), _health(), _classification())
        for key in ("account_id", "should_trigger", "confidence", "reasoning",
                    "tier", "churn_risk", "anomaly_severity", "anomaly_types"):
            assert key in d, f"Missing key: {key}"


# ─────────────────────────────────────────────────────────────────────────────
# Checksum utility
# ─────────────────────────────────────────────────────────────────────────────

from pipeline.utils import compute_comms_checksum


class TestChecksum:
    def test_empty_inputs_stable(self):
        c1 = compute_comms_checksum([], [], [])
        c2 = compute_comms_checksum([], [], [])
        assert c1 == c2

    def test_content_change_changes_checksum(self):
        t1 = [{"ticket_id": "t1", "body": "hello", "subject": "s"}]
        t2 = [{"ticket_id": "t1", "body": "goodbye", "subject": "s"}]
        assert compute_comms_checksum(t1, [], []) != compute_comms_checksum(t2, [], [])

    def test_order_invariant_within_source(self):
        tickets = [
            {"ticket_id": "t1", "body": "aaa", "subject": ""},
            {"ticket_id": "t2", "body": "bbb", "subject": ""},
        ]
        tickets_rev = list(reversed(tickets))
        assert compute_comms_checksum(tickets, [], []) == compute_comms_checksum(
            tickets_rev, [], []
        )

    def test_metadata_change_ignored(self):
        # Changing timestamp or status should NOT change checksum
        t1 = [{"ticket_id": "t1", "body": "body", "subject": "sub", "status": "open"}]
        t2 = [{"ticket_id": "t1", "body": "body", "subject": "sub", "status": "closed"}]
        assert compute_comms_checksum(t1, [], []) == compute_comms_checksum(t2, [], [])

    def test_hex_digest_format(self):
        c = compute_comms_checksum([], [], [])
        assert len(c) == 64
        assert all(ch in "0123456789abcdef" for ch in c)
