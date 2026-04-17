"""Shared utilities for the silent churn pipeline."""

import hashlib
import json
from typing import Dict, List


def compute_comms_checksum(
    tickets: List[Dict],
    slack: List[Dict],
    transcripts: List[Dict],
) -> str:
    """Stable SHA-256 of an account's communications content.

    Used by the dashboard to detect whether communications have changed since
    the last analysis run. If the checksum matches the stored value, the cached
    classification is still valid and no LLM call is needed.

    The hash covers ticket bodies, Slack message texts, and transcript content —
    the parts that actually feed into the LLM prompt. Metadata fields (timestamps,
    status, IDs) are excluded so minor bookkeeping changes don't trigger a re-run.
    """
    parts: List[str] = []
    for t in sorted(tickets, key=lambda x: x.get("ticket_id", "")):
        parts.append(t.get("body", "") + t.get("subject", ""))
    for m in sorted(slack, key=lambda x: x.get("message_id", "")):
        parts.append(m.get("text", ""))
    for tr in sorted(transcripts, key=lambda x: x.get("transcript_id", "")):
        parts.append(tr.get("transcript", ""))

    payload = "\n---\n".join(parts)
    return hashlib.sha256(payload.encode()).hexdigest()
