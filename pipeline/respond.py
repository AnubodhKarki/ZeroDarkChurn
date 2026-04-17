"""Draft outreach message generator for the silent churn pipeline.

Only runs on accounts where the correlator set should_trigger=True.
Takes the full signal bundle and generates a personalised draft email
for the CS team to review and send.

Usage:
    from pipeline.respond import draft_outreach
    result = draft_outreach(account, health, classification, correlation, llm)
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a customer success manager at DevAPICo, a speech-to-text API company. "
    "You write concise, technical, and genuinely helpful outreach emails to developers. "
    "You never use marketing speak or generic check-ins."
)

OUTREACH_PROMPT = """Draft a short outreach email to a developer at {company_name}.

ACCOUNT CONTEXT:
- Company: {company_name} ({tier} tier, ${mrr}/mo MRR)
- Use case: {use_case}
- Funding stage: {funding_stage}

WHAT WE DETECTED:
- Health signals: {anomaly_types}
- Usage change: {usage_detail}
- LLM risk assessment: {churn_risk} risk
- Specific signals from their communications: {churn_signals}

RECOMMENDED ACTION from analysis: {recommended_action}

Write the email body only (no subject line, no "Dear X", just the body starting directly).
Requirements:
- Address the specific technical issue detected, not generic "how are you"
- Reference the usage pattern or error rate specifically if relevant
- Offer one concrete next step (a call, a doc link, a config review)
- Under 120 words
- Casual but technical tone — developer to developer
- End with a soft, non-pushy call to action
- Do not mention competitor names or internal risk scores"""

MAX_SIGNALS_TO_INCLUDE = 3


def draft_outreach(
    account: Dict,
    health: Dict,
    classification: Dict,
    correlation: Dict,
    llm: Any,
) -> Dict:
    """Generate a draft outreach email for a triggered account.

    Args:
        account:     One row from accounts.csv as a dict.
        health:      Output from pipeline.health (one account).
        classification: Output from pipeline.classify (one account).
        correlation: Output from pipeline.correlate (one account).
        llm:         Any object with a .complete(prompt, **kwargs) method.

    Returns:
        Dict with keys: account_id, draft_email, error (None on success).
    """
    acc_id = str(account.get("account_id", ""))

    prompt = _build_prompt(account, health, classification, correlation)

    try:
        response = llm.complete(
            prompt,
            system=SYSTEM_PROMPT,
            max_tokens=300,
            temperature=0.5,   # slightly higher for natural prose
            json_mode=False,
        )
        return {
            "account_id": acc_id,
            "draft_email": response.text.strip(),
            "error": None,
        }
    except Exception as exc:
        logger.error("Outreach generation failed for %s: %s", acc_id, exc)
        return {
            "account_id": acc_id,
            "draft_email": "",
            "error": str(exc),
        }


def _build_prompt(
    account: Dict,
    health: Dict,
    classification: Dict,
    correlation: Dict,
) -> str:
    company = str(account.get("company_name", "your company"))
    tier = str(account.get("tier", "starter"))
    mrr = int(account.get("current_mrr_usd", 0))
    funding = str(account.get("funding_stage", "unknown"))
    use_case = str(classification.get("use_case", str(account.get("primary_use_case", "unknown"))))

    anomaly_types = health.get("anomaly_types", [])
    details = health.get("details", {})

    # Build a human-readable usage detail string
    usage_detail = _usage_detail_str(anomaly_types, details)

    churn_risk = str(classification.get("churn_risk", "medium"))
    signals: List[str] = classification.get("churn_signals", [])
    signals_str = (
        "; ".join(f'"{s}"' for s in signals[:MAX_SIGNALS_TO_INCLUDE])
        if signals else "none identified"
    )
    recommended = str(classification.get("recommended_action", "Schedule a check-in call."))

    return OUTREACH_PROMPT.format(
        company_name=company,
        tier=tier,
        mrr=mrr,
        funding_stage=funding,
        use_case=use_case,
        anomaly_types=", ".join(anomaly_types) if anomaly_types else "usage anomaly",
        usage_detail=usage_detail,
        churn_risk=churn_risk,
        churn_signals=signals_str,
        recommended_action=recommended,
    )


def _usage_detail_str(anomaly_types: List[str], details: Dict) -> str:
    """Convert health details into a readable sentence for the prompt."""
    parts: List[str] = []

    if "usage_drop" in anomaly_types:
        pct = details.get("usage_drop_pct")
        if pct is not None:
            parts.append(f"API call volume down {abs(pct):.0f}% vs baseline")

    if "wow_drop" in anomaly_types:
        pct = details.get("wow_delta_pct")
        if pct is not None:
            parts.append(f"week-over-week volume drop of {abs(pct):.0f}%")

    if "error_spike" in anomaly_types:
        z = details.get("error_rate_zscore")
        rate = details.get("error_rate_recent")
        if rate is not None:
            parts.append(f"error rate elevated to {rate*100:.1f}%")
        elif z is not None:
            parts.append(f"error rate z-score {z:.1f}σ above baseline")

    if "webhook_drop" in anomaly_types:
        drop = details.get("webhook_drop_pp")
        recent = details.get("webhook_success_recent")
        if drop is not None and recent is not None:
            parts.append(f"webhook success rate dropped to {recent*100:.1f}%")

    return "; ".join(parts) if parts else "anomalous usage pattern detected"
