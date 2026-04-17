"""Pipeline orchestrator for the silent churn detection system.

Runs the full pipeline end to end:
  health → (if anomaly) classify → correlate → (if triggered) respond

Writes all intermediate signals and draft emails to a Parquet file.

Usage:
    python -m pipeline.run
    python -m pipeline.run --data-dir data/output --out results.parquet
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_pipeline(data_dir: Path, out_path: Path) -> List[Dict]:
    """Execute the full silent-churn detection pipeline.

    Args:
        data_dir: Directory containing accounts.csv, usage.csv,
                  tickets.json, slack.json, transcripts.json.
        out_path: Path to write the results Parquet file.

    Returns:
        List of per-account result dicts.
    """
    t0 = time.monotonic()

    # ── Load data ─────────────────────────────────────────────────────────────
    logger.info("Loading data from %s", data_dir)
    accounts_df = pd.read_csv(data_dir / "accounts.csv")
    usage_df = pd.read_csv(data_dir / "usage.csv")
    tickets: List[Dict] = json.loads((data_dir / "tickets.json").read_text())
    slack: List[Dict] = json.loads((data_dir / "slack.json").read_text())
    transcripts: List[Dict] = json.loads((data_dir / "transcripts.json").read_text())

    # Index communications by account_id for fast lookup
    tickets_by_acc = _index_by(tickets, "account_id")
    slack_by_acc = _index_by(slack, "account_id")
    transcripts_by_acc = _index_by(transcripts, "account_id")

    # ── Step 1: Health anomaly detection (pure pandas, no LLM) ───────────────
    from pipeline.health import detect_anomalies
    logger.info("Running health anomaly detection...")
    health_results = detect_anomalies(usage_df)
    health_by_acc = {r["account_id"]: r for r in health_results}

    n_anomalies = sum(1 for r in health_results if r["has_anomaly"])
    logger.info("  %d/%d accounts flagged with anomalies", n_anomalies, len(accounts_df))

    # ── LLM client (lazy — only instantiated if there are anomalies) ──────────
    llm: Optional[Any] = None

    if n_anomalies > 0:
        from llm import get_client
        llm = get_client()
        logger.info("  LLM: %s / %s", llm._provider_name(), llm._model_name())

    # ── Steps 2–4: Classify → Correlate → Respond ────────────────────────────
    from pipeline.classify import classify_account
    from pipeline.correlate import correlate
    from pipeline.respond import draft_outreach
    from pipeline.utils import compute_comms_checksum

    results: List[Dict] = []
    n_classified = 0
    n_triggered = 0
    n_drafted = 0

    for _, account_row in accounts_df.iterrows():
        acc_id = str(account_row["account_id"])
        account = dict(account_row)

        health = health_by_acc.get(acc_id, {
            "account_id": acc_id,
            "has_anomaly": False,
            "anomaly_types": [],
            "details": {},
            "severity": "none",
        })

        # ── Step 2: LLM classification (only if anomaly detected) ─────────────
        classification: Optional[Dict] = None
        if health["has_anomaly"] and llm is not None:
            classification = classify_account(
                account,
                tickets_by_acc.get(acc_id, []),
                slack_by_acc.get(acc_id, []),
                transcripts_by_acc.get(acc_id, []),
                llm,
            )
            n_classified += 1
            logger.debug(
                "  %s classified: churn_risk=%s signals=%d",
                acc_id,
                classification.get("churn_risk"),
                len(classification.get("churn_signals", [])),
            )

        # ── Step 3: Correlation — combine signals into trigger decision ────────
        decision = correlate(account, health, classification)

        # ── Step 4: Respond — draft outreach if triggered ─────────────────────
        draft: Optional[Dict] = None
        if decision["should_trigger"] and llm is not None:
            draft = draft_outreach(account, health, classification or {}, decision, llm)
            n_triggered += 1
            if not draft.get("error"):
                n_drafted += 1

        # Checksum of this account's communications — used by the dashboard
        # to detect whether re-analysis is needed when the button is clicked.
        acc_tickets = tickets_by_acc.get(acc_id, [])
        acc_slack = slack_by_acc.get(acc_id, [])
        acc_transcripts = transcripts_by_acc.get(acc_id, [])
        checksum = compute_comms_checksum(acc_tickets, acc_slack, acc_transcripts)

        results.append(_flatten_result(account, health, classification, decision, draft, checksum))

    # ── Write output ──────────────────────────────────────────────────────────
    out_df = pd.DataFrame(results)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path, index=False)
    logger.info("Results written to %s (%d rows)", out_path, len(out_df))

    elapsed = time.monotonic() - t0
    _print_summary(
        accounts_df, n_anomalies, n_classified, n_triggered, n_drafted,
        llm, elapsed, out_path,
    )
    return results


def _flatten_result(
    account: Dict,
    health: Dict,
    classification: Optional[Dict],
    decision: Dict,
    draft: Optional[Dict],
    comms_checksum: str = "",
) -> Dict:
    """Merge all signal layers into a single flat row for the Parquet output."""
    row: Dict = {
        # Account profile
        "account_id": account.get("account_id"),
        "company_name": account.get("company_name"),
        "tier": account.get("tier"),
        "funding_stage": account.get("funding_stage"),
        "current_mrr_usd": account.get("current_mrr_usd"),
        "primary_use_case": account.get("primary_use_case"),
        # Health signals
        "has_anomaly": health.get("has_anomaly", False),
        "anomaly_severity": health.get("severity", "none"),
        "anomaly_types": json.dumps(health.get("anomaly_types", [])),
        "usage_drop_pct": health.get("details", {}).get("usage_drop_pct"),
        "wow_delta_pct": health.get("details", {}).get("wow_delta_pct"),
        "error_rate_zscore": health.get("details", {}).get("error_rate_zscore"),
        "webhook_drop_pp": health.get("details", {}).get("webhook_drop_pp"),
        # LLM classification
        "classified": classification is not None,
        "use_case_detected": classification.get("use_case") if classification else None,
        "churn_risk": classification.get("churn_risk") if classification else None,
        "churn_signals": json.dumps(classification.get("churn_signals", []) if classification else []),
        "recommended_action": classification.get("recommended_action") if classification else None,
        "reasoning": classification.get("reasoning") if classification else None,
        # Correlation decision
        "should_trigger": decision.get("should_trigger", False),
        "confidence": decision.get("confidence", 0.0),
        "trigger_reasoning": decision.get("reasoning"),
        # Draft outreach
        "draft_email": draft.get("draft_email") if draft else None,
        "draft_error": draft.get("error") if draft else None,
        # Communications checksum — used by dashboard to gate re-analysis
        "comms_checksum": comms_checksum,
    }
    return row


def _index_by(items: List[Dict], key: str) -> Dict[str, List[Dict]]:
    """Group a list of dicts by a key field."""
    index: Dict[str, List[Dict]] = {}
    for item in items:
        k = str(item.get(key, ""))
        index.setdefault(k, []).append(item)
    return index


def _print_summary(
    accounts_df: pd.DataFrame,
    n_anomalies: int,
    n_classified: int,
    n_triggered: int,
    n_drafted: int,
    llm: Optional[Any],
    elapsed: float,
    out_path: Path,
) -> None:
    sep = "═" * 56
    print(f"\n{sep}")
    print(" Silent Churn Pipeline — Run Complete")
    print(sep)
    print(f"  Accounts processed  : {len(accounts_df):>6,}")
    print(f"  Anomalies detected  : {n_anomalies:>6,}")
    print(f"  Accounts classified : {n_classified:>6,}")
    print(f"  Outreach triggered  : {n_triggered:>6,}")
    print(f"  Drafts generated    : {n_drafted:>6,}")
    print(f"  Elapsed             : {elapsed:>6.1f}s")
    if llm is not None:
        print(f"  Provider / model    :  {llm._provider_name()} / {llm._model_name()}")
    print(f"  Output              :  {out_path}")
    print(sep)


# ── CLI entry point ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Run the DevAPICo silent churn pipeline.")
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data/output"),
        help="Directory containing the generated dataset (default: data/output)",
    )
    parser.add_argument(
        "--out", type=Path, default=Path("data/output/results.parquet"),
        help="Output Parquet file path (default: data/output/results.parquet)",
    )
    args = parser.parse_args()
    run_pipeline(args.data_dir, args.out)


if __name__ == "__main__":
    main()
