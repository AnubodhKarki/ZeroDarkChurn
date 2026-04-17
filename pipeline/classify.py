"""LLM risk classification for the silent churn pipeline.

Only runs on accounts that the health detector flagged. Assembles the account's
tickets, Slack messages, and call transcripts into a single context block and
asks the LLM to extract churn signals and assign a risk level.

The output is a structured dict that feeds directly into the correlator.

Usage:
    from pipeline.classify import classify_account
    result = classify_account(account_row, tickets, slack, transcripts, llm_client)
"""

import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a customer success analyst for DevAPICo, a speech-to-text API company. "
    "You analyze developer communications to identify churn risk. Be concise and specific. "
    "Always respond with valid JSON only."
)

CLASSIFY_PROMPT = """You are analyzing customer communications for DevAPICo, a speech-to-text API company.

Given the following tickets, Slack messages, and call transcripts for {company_name} (account {account_id}):

{context}

Extract and return valid JSON with exactly these fields:
1. "use_case": What are they building? (one short phrase, e.g. "meeting transcription SaaS")
2. "churn_risk": one of "low", "medium", "high"
3. "churn_signals": array of strings — quote specific phrases that indicate competitor interest, frustration, pricing pressure, or timeline urgency. Empty array if none.
4. "recommended_action": What should the CS team do? (one sentence, specific to this account)
5. "reasoning": one sentence explaining your churn_risk rating

Respond with only the JSON object, no preamble or commentary."""

# How much context to include per category (chars). Keeps prompts within token budgets.
MAX_TICKET_CHARS = 2000
MAX_SLACK_CHARS = 1500
MAX_TRANSCRIPT_CHARS = 3000


def classify_account(
    account: Dict,
    tickets: List[Dict],
    slack: List[Dict],
    transcripts: List[Dict],
    llm: Any,
) -> Dict:
    """Run LLM risk classification for a single account.

    Args:
        account:     One row from accounts.csv as a dict.
        tickets:     All tickets belonging to this account.
        slack:       All Slack messages belonging to this account.
        transcripts: All transcripts belonging to this account.
        llm:         Any object with a .complete(prompt, **kwargs) method.

    Returns:
        Dict with keys: account_id, use_case, churn_risk, churn_signals,
        recommended_action, reasoning, raw_response, error (if any).
    """
    acc_id = str(account.get("account_id", ""))
    company = str(account.get("company_name", "Unknown"))

    context = _build_context(tickets, slack, transcripts)

    if not context.strip():
        logger.warning("No communications found for %s — defaulting to low risk", acc_id)
        return _fallback(acc_id, reason="no communications available")

    prompt = CLASSIFY_PROMPT.format(
        company_name=company,
        account_id=acc_id,
        context=context,
    )

    try:
        response = llm.complete(
            prompt,
            system=SYSTEM_PROMPT,
            max_tokens=512,
            temperature=0.2,   # low temp — we want consistent structured output
            json_mode=True,
        )
        parsed = _parse_response(response.text, acc_id)
        parsed["account_id"] = acc_id
        parsed["raw_response"] = response.text
        parsed["error"] = None
        return parsed

    except Exception as exc:
        logger.error("Classification failed for %s: %s", acc_id, exc)
        return _fallback(acc_id, reason=str(exc))


def _build_context(
    tickets: List[Dict],
    slack: List[Dict],
    transcripts: List[Dict],
) -> str:
    """Assemble communications into a single context string with section headers."""
    parts: List[str] = []

    if tickets:
        section = "=== SUPPORT TICKETS ===\n"
        for t in tickets:
            section += f"[{t.get('created_at', '')} | {t.get('status', '')}] {t.get('subject', '')}\n"
            section += f"{t.get('body', '')}\n\n"
        parts.append(section[:MAX_TICKET_CHARS])

    if slack:
        section = "=== SLACK MESSAGES ===\n"
        for m in slack:
            section += f"[{m.get('timestamp', '')[:10]}] {m.get('author', 'dev')}: {m.get('text', '')}\n"
        parts.append(section[:MAX_SLACK_CHARS])

    if transcripts:
        section = "=== CALL TRANSCRIPTS ===\n"
        for tr in transcripts:
            section += f"[{tr.get('date', '')} | {tr.get('duration_minutes', '?')} min]\n"
            section += f"{tr.get('transcript', '')}\n\n"
        parts.append(section[:MAX_TRANSCRIPT_CHARS])

    return "\n".join(parts)


def _parse_response(text: str, acc_id: str) -> Dict:
    """Parse and validate the LLM JSON response. Returns a safe default on any failure."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        logger.warning("JSON decode failed for %s: %s", acc_id, exc)
        return _default_classification()

    # Validate and normalise required fields
    churn_risk = str(data.get("churn_risk", "low")).lower()
    if churn_risk not in ("low", "medium", "high"):
        churn_risk = "low"

    signals = data.get("churn_signals", [])
    if not isinstance(signals, list):
        signals = []

    return {
        "use_case": str(data.get("use_case", "unknown")),
        "churn_risk": churn_risk,
        "churn_signals": [str(s) for s in signals],
        "recommended_action": str(data.get("recommended_action", "Schedule a check-in call.")),
        "reasoning": str(data.get("reasoning", "")),
    }


def _default_classification() -> Dict:
    return {
        "use_case": "unknown",
        "churn_risk": "low",
        "churn_signals": [],
        "recommended_action": "Review account manually — classification failed.",
        "reasoning": "Could not parse LLM response.",
    }


def _fallback(acc_id: str, reason: str) -> Dict:
    result = _default_classification()
    result["account_id"] = acc_id
    result["raw_response"] = ""
    result["error"] = reason
    return result
