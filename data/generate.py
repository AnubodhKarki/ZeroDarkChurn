#!/usr/bin/env python3
"""Synthetic data generator for the DevAPICo silent churn detection system.

Generates 100 fake developer API accounts, 90 days of usage data, support
tickets, Slack messages, and call transcripts. Plants known churn cases and
false-positive traps whose ground truth is written to ground_truth.json.

Fictional competitors: VoxCore, TranscriBit, SpeechLayer, AudioNest.
No real company or product names are used anywhere in this dataset.

Usage:
    python -m data.generate
    python -m data.generate --out-dir data/output --skip-llm
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from faker import Faker

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

# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────
SEED = 42
NUM_ACCOUNTS = 100
USAGE_DAYS = 90
# Fixed reference date — all synthetic dates anchor to this.
TODAY = datetime(2024, 11, 15)

# ─────────────────────────────────────────────────────────────────────────────
# Domain constants
# ─────────────────────────────────────────────────────────────────────────────
COMPETITORS = ["VoxCore", "TranscriBit", "SpeechLayer", "AudioNest"]

FUNDING_STAGES = ["bootstrapped", "seed", "series_a", "series_b", "series_c_plus"]
FUNDING_WEIGHTS = [0.30, 0.30, 0.25, 0.10, 0.05]
FUNDING_RANGES = {
    "bootstrapped": (0, 0),
    "seed": (500_000, 3_000_000),
    "series_a": (3_000_000, 15_000_000),
    "series_b": (15_000_000, 60_000_000),
    "series_c_plus": (60_000_000, 300_000_000),
}

TIERS = ["free", "starter", "growth", "enterprise"]
TIER_WEIGHTS_DEFAULT = [0.40, 0.30, 0.20, 0.10]
TIER_LIMITS = {
    "free": 10_000,
    "starter": 100_000,
    "growth": 1_000_000,
    "enterprise": 10_000_000,
}
TIER_MRR_RANGE = {
    "free": (0, 0),
    "starter": (49, 249),
    "growth": (499, 1999),
    "enterprise": (2000, 15000),
}

USE_CASES = [
    "meeting_assistant",
    "call_center_analytics",
    "podcast_transcription",
    "voice_agent",
    "accessibility",
    "other",
]
B2B_USE_CASES = {"meeting_assistant", "call_center_analytics", "voice_agent"}

COST_PER_1M = {
    "openai":    {"input": 0.150, "output": 0.600},
    "anthropic": {"input": 3.000, "output": 15.000},
}

# ─────────────────────────────────────────────────────────────────────────────
# Planted case configuration — deterministic IDs and tiers
# ─────────────────────────────────────────────────────────────────────────────
PLANTED_HEALTH_IDS   = [f"acc_{i:04d}" for i in range(1, 6)]    # acc_0001–acc_0005
PLANTED_RISK_IDS     = [f"acc_{i:04d}" for i in range(6, 11)]   # acc_0006–acc_0010
PLANTED_COMBINED_IDS = [f"acc_{i:04d}" for i in range(11, 16)]  # acc_0011–acc_0015
FP_SEASONAL_IDS      = [f"acc_{i:04d}" for i in range(16, 18)]  # acc_0016–acc_0017
FP_TRANSIENT_IDS     = [f"acc_{i:04d}" for i in range(18, 20)]  # acc_0018–acc_0019
FP_LOYAL_IDS         = ["acc_0020"]

ALL_CHURN = set(PLANTED_HEALTH_IDS + PLANTED_RISK_IDS + PLANTED_COMBINED_IDS)
ALL_FP    = set(FP_SEASONAL_IDS + FP_TRANSIENT_IDS + FP_LOYAL_IDS)
ALL_HEALTH_AFFECTED = set(PLANTED_HEALTH_IDS + PLANTED_COMBINED_IDS)

# Tier overrides — combined cases must be high-value so the correlator fires.
# Health/risk just need to be non-free so they're evaluated at all.
PLANTED_TIER_MAP: Dict[str, str] = {
    "acc_0001": "starter",    "acc_0002": "growth",
    "acc_0003": "starter",    "acc_0004": "growth",    "acc_0005": "starter",
    "acc_0006": "growth",     "acc_0007": "starter",
    "acc_0008": "growth",     "acc_0009": "starter",   "acc_0010": "growth",
    "acc_0011": "enterprise", "acc_0012": "growth",
    "acc_0013": "enterprise", "acc_0014": "growth",    "acc_0015": "enterprise",
    "acc_0016": "starter",    "acc_0017": "growth",
    "acc_0018": "starter",    "acc_0019": "growth",
    "acc_0020": "starter",
}

# Competitor assigned to each risk/combined account (no real names)
PLANTED_COMPETITOR_MAP: Dict[str, str] = {
    "acc_0006": "VoxCore",      "acc_0007": "TranscriBit",
    "acc_0008": "SpeechLayer",  "acc_0009": "AudioNest",   "acc_0010": "VoxCore",
    "acc_0011": "TranscriBit",  "acc_0012": "SpeechLayer",
    "acc_0013": "AudioNest",    "acc_0014": "VoxCore",     "acc_0015": "TranscriBit",
    # FP: this account mentions a competitor but is actually loyal
    "acc_0020": "SpeechLayer",
}

# FP trap type — drives seasonal/transient context in generated content
FP_TYPE_MAP: Dict[str, str] = {
    "acc_0016": "seasonal",
    "acc_0017": "seasonal",
    "acc_0018": "transient_error",
    "acc_0019": "transient_error",
}


# ─────────────────────────────────────────────────────────────────────────────
# Tracking wrapper — counts live calls vs cache hits for the summary
# ─────────────────────────────────────────────────────────────────────────────
class _TrackingWrapper:
    """Wraps LLMClient to count live calls vs cache hits without modifying the client."""

    def __init__(self, client: Any) -> None:
        self._client = client
        self.live_calls = 0
        self.cache_hits = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def complete(self, prompt: str, **kwargs: Any) -> Any:
        key = self._client._cache_key(
            prompt,
            kwargs.get("system"),
            kwargs.get("max_tokens", 1024),
            kwargs.get("temperature", 0.3),
            kwargs.get("json_mode", False),
        )
        pre_cached = self._client._cache_path(key).exists()
        response = self._client.complete(prompt, **kwargs)
        if pre_cached:
            self.cache_hits += 1
        else:
            self.live_calls += 1
            self.total_input_tokens += response.input_tokens
            self.total_output_tokens += response.output_tokens
        return response

    def _provider_name(self) -> str:
        return self._client._provider_name()

    def _model_name(self) -> str:
        return self._client._model_name()


# ─────────────────────────────────────────────────────────────────────────────
# Account generation
# ─────────────────────────────────────────────────────────────────────────────
def generate_accounts(rng: np.random.Generator, fake: Faker) -> pd.DataFrame:
    """Generate 100 account profiles. Planted cases get deterministic tier overrides."""
    rows = []
    for i in range(1, NUM_ACCOUNTS + 1):
        acc_id = f"acc_{i:04d}"

        tier = PLANTED_TIER_MAP.get(acc_id) or str(
            rng.choice(TIERS, p=TIER_WEIGHTS_DEFAULT)
        )
        funding_stage = str(rng.choice(FUNDING_STAGES, p=FUNDING_WEIGHTS))
        lo, hi = FUNDING_RANGES[funding_stage]
        funding_amount = int(rng.uniform(lo, hi)) if hi > 0 else 0

        days_ago = int(rng.uniform(30, 540))
        signup_date = TODAY - timedelta(days=days_ago)

        mrr_lo, mrr_hi = TIER_MRR_RANGE[tier]
        mrr = int(rng.uniform(mrr_lo, mrr_hi)) if mrr_hi > 0 else 0

        use_case = str(rng.choice(USE_CASES))

        rows.append({
            "account_id": acc_id,
            "company_name": fake.company(),
            "funding_stage": funding_stage,
            "funding_amount_usd": funding_amount,
            "signup_date": signup_date.strftime("%Y-%m-%d"),
            "tier": tier,
            "monthly_tier_limit": TIER_LIMITS[tier],
            "current_mrr_usd": mrr,
            "primary_use_case": use_case,
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Usage generation
# ─────────────────────────────────────────────────────────────────────────────
def generate_usage(accounts_df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Generate 90 days of API usage logs. Applies churn and FP modifications in-place."""
    dates = [TODAY - timedelta(days=USAGE_DAYS - i - 1) for i in range(USAGE_DAYS)]
    date_strs = [d.strftime("%Y-%m-%d") for d in dates]

    # day_idx 76–89 = last 14 days (health churn drop window)
    # day_idx 83–89 = last 7 days  (error spike window)
    # day_idx 10–17 = seasonal FP trap window (~Aug 27 – Sep 3)
    # day_idx 80–81 = transient error FP window (~Nov 13–14)

    rows = []
    for _, acct in accounts_df.iterrows():
        acc_id = str(acct["account_id"])
        tier = str(acct["tier"])
        use_case = str(acct["primary_use_case"])

        daily_limit = TIER_LIMITS[tier] / 30
        usage_pct = float(rng.uniform(0.25, 0.65))
        base_daily = daily_limit * usage_pct
        b2b = use_case in B2B_USE_CASES
        base_error_rate = float(rng.uniform(0.005, 0.025))

        for day_idx, (date, date_str) in enumerate(zip(dates, date_strs)):
            weekend_factor = 0.20 if (b2b and date.weekday() >= 5) else 1.0
            noise = float(np.clip(rng.normal(1.0, 0.15), 0.5, 1.5))
            calls = max(0, int(base_daily * weekend_factor * noise))

            # ── Risk-only churn: modest 7-day usage dip ────────────────────
            # Just enough to trigger the anomaly detector via WoW comparison.
            # No error spike and no webhook drop — that's what separates risk-only
            # from health churn. The LLM then catches the competitor signal in comms.
            if acc_id in set(PLANTED_RISK_IDS) and day_idx >= 83:
                calls = int(calls * float(rng.uniform(0.58, 0.70)))

            # ── Health churn: strong drop in last 14 days ──────────────────
            if acc_id in ALL_HEALTH_AFFECTED and day_idx >= 76:
                calls = int(calls * float(rng.uniform(0.35, 0.55)))

            # ── Error rate ──────────────────────────────────────────────────
            if acc_id in ALL_HEALTH_AFFECTED and day_idx >= 83:
                # Elevated errors in last 7 days for health/combined cases
                err_rate = float(rng.uniform(0.08, 0.18))
            else:
                err_rate = base_error_rate * float(rng.uniform(0.5, 2.0))

            error_count = min(int(calls * err_rate), calls)

            # ── FP: seasonal dip (last 7 days only, day_idx 83–89) ───────────
            # A sharp, recent usage drop with NO error spike and NO webhook drop.
            # Applied only to the current week so WoW comparison flags it.
            # The previous week is normal — making this look like sudden churn.
            # The LLM sees "team was at a conference / slow period" in comms
            # and classifies it as low risk, saving it from triggering outreach.
            if acc_id in FP_SEASONAL_IDS and day_idx >= 83:
                calls = int(calls * float(rng.uniform(0.30, 0.50)))
                error_count = min(int(calls * err_rate), calls)  # no elevated errors

            # ── FP: transient error spike (days 83–89, no usage drop) ───────
            # Error rate is elevated but call volume is normal. Caused by a
            # temporary misconfiguration on the customer's side, now resolving.
            if acc_id in FP_TRANSIENT_IDS and day_idx >= 83:
                error_count = int(calls * float(rng.uniform(0.12, 0.22)))

            # ── Webhook success rate ────────────────────────────────────────
            # Only health/combined churn cases get webhook degradation.
            # Seasonal and transient FP cases keep normal webhook rates —
            # this is a deliberate distinguishing signal for the correlator.
            if acc_id in ALL_HEALTH_AFFECTED and day_idx >= 76:
                webhook_success = float(rng.uniform(0.72, 0.88))
            else:
                webhook_success = float(rng.uniform(0.95, 0.999))

            # ── 429 hits: only when near limit ─────────────────────────────
            rl_hits = int(rng.exponential(0.3)) if (calls > base_daily * 0.8) else 0

            p95_latency = max(100, int(rng.normal(450, 80)))

            rows.append({
                "account_id": acc_id,
                "date": date_str,
                "api_calls": calls,
                "error_count": error_count,
                "rate_limit_hits_429": rl_hits,
                "webhook_success_rate": round(webhook_success, 4),
                "p95_latency_ms": p95_latency,
            })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# LLM content — prompts
# ─────────────────────────────────────────────────────────────────────────────
def _ticket_prompt(acct: Dict, n: int, competitor: Optional[str], is_loyal: bool,
                   fp_type: Optional[str] = None) -> str:
    use_case = acct["primary_use_case"].replace("_", " ")
    if competitor and is_loyal:
        signal = (
            f"In one ticket, the developer should mention they evaluated {competitor} "
            f"as an alternative but ultimately decided to stay with DevAPICo."
        )
    elif competitor:
        signal = (
            f"In at least one ticket, the developer should mention they are actively "
            f"evaluating {competitor} as an alternative due to frustration with "
            f"reliability or pricing."
        )
    elif fp_type == "seasonal":
        signal = (
            "In at least one ticket, the developer casually mentions their usage is "
            "lower this month because their team has been at conferences and the "
            "business is in a slower period. They expect volume to pick back up soon."
        )
    elif fp_type == "transient_error":
        signal = (
            "In at least one ticket, the developer mentions they noticed elevated "
            "error rates recently but have since traced it to a misconfiguration "
            "on their own infrastructure (not DevAPICo's fault). Consider it resolved."
        )
    else:
        signal = ""

    return (
        f"Generate {n} realistic support tickets from a developer at {acct['company_name']} "
        f"building a {use_case} app on DevAPICo (speech-to-text API). {signal}\n\n"
        f"Return a JSON array of exactly {n} objects, each with:\n"
        f'- "subject": short ticket subject (under 12 words)\n'
        f'- "body": 2–4 sentence ticket body written by the developer\n'
        f'- "created_at": YYYY-MM-DD between 2024-08-17 and 2024-11-15\n'
        f'- "status": one of "open", "closed", "pending"\n\n'
        f"Return only the JSON array."
    )


def _slack_prompt(acct: Dict, n: int, competitor: Optional[str], is_loyal: bool,
                  fp_type: Optional[str] = None) -> str:
    use_case = acct["primary_use_case"].replace("_", " ")
    if competitor and is_loyal:
        signal = (
            f"In one message, the developer mentions they looked at {competitor} "
            f"but decided DevAPICo was the right call."
        )
    elif competitor:
        signal = (
            f"In at least one message, the developer expresses frustration and "
            f"mentions they are considering switching to {competitor}."
        )
    elif fp_type == "seasonal":
        signal = (
            "In one message, the developer mentions usage is down this month "
            "because the team has been traveling and it's a slower business period. "
            "They sound relaxed about it."
        )
    elif fp_type == "transient_error":
        signal = (
            "In one message, the developer mentions they saw a spike in errors "
            "recently but it turned out to be a DNS misconfiguration on their end. "
            "They've since fixed it."
        )
    else:
        signal = ""

    return (
        f"Generate {n} realistic Slack messages from a developer at {acct['company_name']} "
        f"in a shared DevAPICo customer Slack workspace. They are building a {use_case} app. "
        f"Messages should vary: questions, issue reports, quick updates. {signal}\n\n"
        f"Return a JSON array of exactly {n} objects, each with:\n"
        f'- "timestamp": ISO datetime between 2024-08-17T00:00:00 and 2024-11-15T23:59:59\n'
        f'- "author": a plausible developer first name\n'
        f'- "text": the message (1–3 sentences, casual developer tone)\n\n'
        f"Return only the JSON array."
    )


def _transcript_prompt(acct: Dict, n: int, competitor: Optional[str], is_loyal: bool,
                       fp_type: Optional[str] = None) -> str:
    use_case = acct["primary_use_case"].replace("_", " ")
    if competitor and is_loyal:
        signal = (
            f"In one call, the developer mentions they evaluated {competitor} as an "
            f"alternative but explicitly says they decided to stay with DevAPICo and why."
        )
    elif competitor:
        signal = (
            f"In at least one call, the developer mentions they are actively evaluating "
            f"{competitor} as an alternative and sounds like they might churn."
        )
    elif fp_type == "seasonal":
        signal = (
            "In the call, the developer explains their usage is lower because the company "
            "had a slow October — team offsite plus a quiet period before their product launch. "
            "They expect volume to ramp up significantly in December. No churn risk."
        )
    elif fp_type == "transient_error":
        signal = (
            "In the call, the developer mentions they saw elevated errors in their monitoring "
            "dashboard last week. They've since traced it to a misconfiguration in their own "
            "load balancer. The issue is fully resolved and they're happy with DevAPICo."
        )
    else:
        signal = ""

    return (
        f"Generate {n} realistic customer success call transcript(s) between a CS agent "
        f"at DevAPICo and a developer at {acct['company_name']} building a {use_case} app. "
        f"Each transcript should be ~400–550 words. {signal}\n\n"
        f"Return a JSON array of exactly {n} objects, each with:\n"
        f'- "date": YYYY-MM-DD between 2024-08-17 and 2024-11-15\n'
        f'- "duration_minutes": integer 15–45\n'
        f'- "participants": ["CS Agent", "Developer"]\n'
        f'- "transcript": full text using "CS:" and "Dev:" speaker labels\n\n'
        f"Return only the JSON array."
    )


# ─────────────────────────────────────────────────────────────────────────────
# LLM content — generation helpers
# ─────────────────────────────────────────────────────────────────────────────
def _parse_llm_json(text: str, acc_id: str, label: str) -> Optional[List[Dict]]:
    """Parse JSON from LLM response. Returns None on failure (caller falls back to template).

    Validates that the result is a list of dicts — rejects lists of strings that
    some models occasionally return when they misread the instruction.
    """
    try:
        data = json.loads(text)
        if isinstance(data, list):
            # Guard: every item must be a dict
            if all(isinstance(item, dict) for item in data):
                return data
            logger.warning("LLM returned list of non-dicts for %s %s — falling back", label, acc_id)
            return None
        # Some models wrap the array in a key
        if isinstance(data, dict):
            for v in data.values():
                if isinstance(v, list) and all(isinstance(item, dict) for item in v):
                    return v
    except (json.JSONDecodeError, AttributeError) as exc:
        logger.warning("JSON parse failed for %s %s: %s", label, acc_id, exc)
    return None


def _template_tickets(acct: Dict, n: int, rng: np.random.Generator,
                       competitor: Optional[str], is_loyal: bool,
                       fp_type: Optional[str] = None) -> List[Dict]:
    use_case = acct["primary_use_case"].replace("_", " ")
    results = []
    subjects = [
        f"Accuracy degradation in {use_case} pipeline",
        "Rate limit questions for production workload",
        "Webhook delivery failures — missing events",
        "P95 latency spikes affecting our SLA",
        "Billing question on overage charges",
    ]
    bodies = [
        f"We've seen a notable drop in transcription accuracy for our {use_case} app over the past two weeks. Can your team take a look at the recent error logs?",
        f"We're approaching our monthly limit and wanted to understand upgrade options before we hit the cap in production.",
        f"Several webhooks have been failing silently over the last few days, causing data gaps in our {use_case} pipeline.",
        f"Our P95 response times have jumped to over 1.2 seconds. This is above our SLA threshold and users are noticing.",
        f"The invoice for October looks higher than expected. Can someone walk me through the overage calculation?",
    ]
    for j in range(n):
        subj = subjects[j % len(subjects)]
        body = bodies[j % len(bodies)]
        if j == 0 and competitor and not is_loyal:
            subj = "Considering alternatives — need to resolve issues soon"
            body = (
                f"We've been dealing with ongoing reliability issues with our {use_case} integration. "
                f"At this point we've started evaluating {competitor} as a potential alternative. "
                f"Would love to get on a call to see if these issues can be resolved quickly."
            )
        elif j == 0 and competitor and is_loyal:
            subj = f"Checked out {competitor} — sticking with DevAPICo"
            body = (
                f"Wanted to let you know we did a short evaluation of {competitor} last month. "
                f"Their accuracy wasn't as good for our {use_case} use case, so we're staying. "
                f"That said, we'd still love to chat about roadmap."
            )
        elif j == 0 and fp_type == "seasonal":
            subj = "Lower usage this month — just a slow period"
            body = (
                f"Heads up that our API usage has been down this month. Our team was at a "
                f"conference and we had an internal product review week. Volume should be "
                f"back to normal in December when we launch our next feature."
            )
        elif j == 0 and fp_type == "transient_error":
            subj = "Error spike last week — issue was on our side"
            body = (
                f"Wanted to flag that we saw elevated error rates in our monitoring last week. "
                f"After digging in, it turned out to be a DNS misconfiguration in our load "
                f"balancer — nothing on DevAPICo's end. We've resolved it and things look clean now."
            )
        days_ago = max(1, 80 - j * 10)
        results.append({
            "subject": subj,
            "body": body,
            "created_at": (TODAY - timedelta(days=days_ago)).strftime("%Y-%m-%d"),
            "status": str(rng.choice(["open", "closed", "pending"])),
        })
    return results


def _template_slack(acct: Dict, n: int, rng: np.random.Generator,
                     competitor: Optional[str], is_loyal: bool,
                     fp_type: Optional[str] = None) -> List[Dict]:
    use_case = acct["primary_use_case"].replace("_", " ")
    messages = [
        f"hey team — seeing some weird latency on the {use_case} endpoint today, anyone else?",
        f"quick q: is there a way to get webhook retries configured per-endpoint?",
        f"transcription accuracy has been solid this week, nice work 👍",
        f"we went live last night — {use_case} pipeline is running in prod!",
        f"getting 429s more than expected this morning, is there an incident?",
        f"does the streaming API support speaker diarization yet?",
        f"our error rate ticked up over the weekend — filed a ticket but flagging here too",
        f"can someone share the docs for batch processing large audio files?",
        f"the new SDK version fixed our webhook issue, thanks for the quick turnaround",
        f"just hit our monthly limit two weeks early — need to talk about upgrading",
    ]
    if competitor and not is_loyal:
        messages[0] = (
            f"honestly been frustrated with the reliability lately — we've been doing a quick eval of {competitor}, "
            f"nothing decided yet but wanted to flag it"
        )
    elif competitor and is_loyal:
        messages[0] = (
            f"we poked at {competitor} for a few days — their latency was worse for our {use_case} case, "
            f"so we're staying put"
        )
    elif fp_type == "seasonal":
        messages[0] = (
            "usage is down a bit this month, we've had a lot of the team traveling and it's just a slower "
            "period — should be back to normal volume by December"
        )
    elif fp_type == "transient_error":
        messages[0] = (
            "hey, noticed some error rate spikes in our alerts last week — traced it back to a DNS config "
            "issue on our end, totally resolved now, no issues on DevAPICo's side"
        )
    results = []
    for j in range(n):
        days_ago = max(1, 85 - j * 8)
        results.append({
            "timestamp": (TODAY - timedelta(days=days_ago)).strftime("%Y-%m-%dT%H:%M:%S"),
            "author": "dev",
            "text": messages[j % len(messages)],
        })
    return results


def _template_transcripts(acct: Dict, n: int, rng: np.random.Generator,
                            competitor: Optional[str], is_loyal: bool,
                            fp_type: Optional[str] = None) -> List[Dict]:
    use_case = acct["primary_use_case"].replace("_", " ")
    results = []
    for j in range(n):
        days_ago = max(1, 60 - j * 20)
        if competitor and not is_loyal:
            body = (
                f"CS: Thanks for joining today. How are things going with the {use_case} integration?\n"
                f"Dev: Honestly, it's been a rough few weeks. We've had repeated reliability issues "
                f"and our error rates have been climbing. It's starting to affect our users.\n"
                f"CS: I'm sorry to hear that. Can you tell me more about what you're seeing?\n"
                f"Dev: It's a mix of things — higher latency, some accuracy regression, and a couple "
                f"of webhook failures that took us hours to debug. We've actually been looking at "
                f"{competitor} as a backup option, just to be safe.\n"
                f"CS: I understand. We definitely want to resolve this. Let me pull up your account "
                f"and get our engineering team looped in today.\n"
                f"Dev: I appreciate that. I want to stay with DevAPICo but we need to see improvement soon."
            )
        elif competitor and is_loyal:
            body = (
                f"CS: Hey, glad we could connect. Anything on your mind?\n"
                f"Dev: Yeah, I wanted to give you a heads up — we spent a couple of days evaluating "
                f"{competitor} last month just to benchmark. Their transcription accuracy was noticeably "
                f"worse for our {use_case} use case, so we decided to stay with DevAPICo.\n"
                f"CS: Really glad to hear that. Was there anything specific that made the difference?\n"
                f"Dev: Accuracy mostly. And honestly the support has been better here. We'd still love "
                f"to talk about getting on an annual plan if that unlocks better pricing.\n"
                f"CS: Absolutely, I'll get our account team to send over some options."
            )
        elif fp_type == "seasonal":
            body = (
                f"CS: Hey, great to connect. How are things going?\n"
                f"Dev: Pretty good overall. I noticed you reached out — probably because our "
                f"usage has been down lately, right?\n"
                f"CS: We did notice a dip, yeah. Is everything okay on your end?\n"
                f"Dev: Oh totally, nothing to worry about. October and November are just "
                f"slow months for us. Half the team was at an industry conference, and we "
                f"had an internal product review sprint where we weren't pushing new features. "
                f"We're actually gearing up for a big launch in December that should push our "
                f"volume up significantly.\n"
                f"CS: That's great to hear! What's the launch?\n"
                f"Dev: We're adding real-time {use_case} features to our enterprise tier. "
                f"We're expecting to double our API call volume within 60 days of launch. "
                f"Might need to talk about a volume plan before then.\n"
                f"CS: Definitely — I'll have our account team reach out closer to launch "
                f"to make sure you have the right plan in place.\n"
                f"Dev: Perfect. And honestly, everything has been rock solid on the API side. "
                f"No complaints at all."
            )
        elif fp_type == "transient_error":
            body = (
                f"CS: Thanks for joining the call today. We noticed some error rate anomalies "
                f"in your account last week and wanted to check in.\n"
                f"Dev: Yeah, I was going to reach out about that actually. We dug into it "
                f"and it turns out it was completely on our end — a misconfiguration in our "
                f"load balancer was causing some requests to hit a stale DNS entry. We "
                f"were routing about 15 percent of traffic to a deprecated endpoint.\n"
                f"CS: Ah, okay — so the errors originated before they even hit our API?\n"
                f"Dev: Exactly. We've fixed the config and rotated the DNS. Everything "
                f"looks clean now. I actually wanted to apologize for the noise in your "
                f"monitoring if it caused any alerts on your side.\n"
                f"CS: No worries at all, we appreciate the transparency. Is there anything "
                f"we can do to help — any logging you'd want us to surface to make debugging "
                f"easier next time?\n"
                f"Dev: Actually, more granular error codes on the response headers would be "
                f"helpful. Something that distinguishes a client-side routing issue from a "
                f"real API error. But overall, super happy with the service."
            )
        else:
            body = (
                f"CS: Thanks for taking the time today. How is the {use_case} integration going?\n"
                f"Dev: Pretty well overall. We're in production now and throughput has been good. "
                f"I had a couple of questions about the streaming API for an upcoming feature.\n"
                f"CS: Of course — what are you looking to build?\n"
                f"Dev: We want to add real-time transcription to our product. I've read the docs but "
                f"wanted to understand the latency guarantees before we commit to the architecture.\n"
                f"CS: Great use case. Let me walk you through what's realistic at your current tier "
                f"and what we'd recommend for production-scale streaming.\n"
                f"Dev: That would be super helpful, thanks."
            )
        results.append({
            "date": (TODAY - timedelta(days=days_ago)).strftime("%Y-%m-%d"),
            "duration_minutes": int(rng.integers(15, 45)),
            "participants": ["CS Agent", "Developer"],
            "transcript": body,
        })
    return results


# ─────────────────────────────────────────────────────────────────────────────
# LLM content — per-account generation
# ─────────────────────────────────────────────────────────────────────────────
def _gen_content(
    acct: Dict,
    n: int,
    prompt_fn,
    template_fn,
    llm: Optional[Any],
    rng: np.random.Generator,
    competitor: Optional[str],
    is_loyal: bool,
    label: str,
    max_tokens: int = 1500,
    fp_type: Optional[str] = None,
) -> List[Dict]:
    """Generate n items for one account. Uses LLM if available, falls back to templates."""
    if n == 0:
        return []
    if llm is None:
        return template_fn(acct, n, rng, competitor, is_loyal, fp_type)

    prompt = prompt_fn(acct, n, competitor, is_loyal, fp_type)
    try:
        resp = llm.complete(prompt, json_mode=True, max_tokens=max_tokens, temperature=0.8)
        parsed = _parse_llm_json(resp.text, str(acct["account_id"]), label)
        if parsed:
            return parsed[:n]
    except Exception as exc:
        logger.warning("LLM call failed for %s %s: %s — using template", label, acct["account_id"], exc)

    return template_fn(acct, n, rng, competitor, is_loyal, fp_type)


def generate_tickets(
    accounts_df: pd.DataFrame, llm: Optional[Any], rng: np.random.Generator
) -> List[Dict]:
    """Generate 0–5 support tickets per account."""
    all_items: List[Dict] = []
    for _, acct in accounts_df.iterrows():
        acct_dict = dict(acct)
        acc_id = str(acct["account_id"])
        tier = str(acct["tier"])

        # Planted risk/combined must have at least 2 tickets so signals are present
        if acc_id in (set(PLANTED_RISK_IDS) | set(PLANTED_COMBINED_IDS)):
            n = max(2, int(rng.integers(2, 5)))
        elif tier == "free":
            n = int(rng.integers(0, 3))
        else:
            n = int(rng.integers(0, 6))

        competitor = PLANTED_COMPETITOR_MAP.get(acc_id)
        is_loyal = acc_id in FP_LOYAL_IDS
        fp_type = FP_TYPE_MAP.get(acc_id)
        items = _gen_content(
            acct_dict, n, _ticket_prompt, _template_tickets,
            llm, rng, competitor, is_loyal, "ticket", fp_type=fp_type,
        )
        for j, item in enumerate(items):
            item["ticket_id"] = f"tkt_{acc_id}_{j + 1:02d}"
            item["account_id"] = acc_id
        all_items.extend(items)
    return all_items


def generate_slack(
    accounts_df: pd.DataFrame, llm: Optional[Any], rng: np.random.Generator
) -> List[Dict]:
    """Generate 0–10 Slack messages per account."""
    all_items: List[Dict] = []
    for _, acct in accounts_df.iterrows():
        acct_dict = dict(acct)
        acc_id = str(acct["account_id"])

        if acc_id in (set(PLANTED_RISK_IDS) | set(PLANTED_COMBINED_IDS)):
            n = max(2, int(rng.integers(2, 8)))
        else:
            n = int(rng.integers(0, 11))

        competitor = PLANTED_COMPETITOR_MAP.get(acc_id)
        is_loyal = acc_id in FP_LOYAL_IDS
        fp_type = FP_TYPE_MAP.get(acc_id)
        items = _gen_content(
            acct_dict, n, _slack_prompt, _template_slack,
            llm, rng, competitor, is_loyal, "slack", fp_type=fp_type,
        )
        for j, item in enumerate(items):
            item["message_id"] = f"msg_{acc_id}_{j + 1:02d}"
            item["account_id"] = acc_id
        all_items.extend(items)
    return all_items


def generate_transcripts(
    accounts_df: pd.DataFrame, llm: Optional[Any], rng: np.random.Generator
) -> List[Dict]:
    """Generate 0–3 call transcripts per account."""
    all_items: List[Dict] = []
    for _, acct in accounts_df.iterrows():
        acct_dict = dict(acct)
        acc_id = str(acct["account_id"])

        # Risk/combined cases must have at least 1 transcript with competitor signal
        if acc_id in (set(PLANTED_RISK_IDS) | set(PLANTED_COMBINED_IDS) | set(FP_LOYAL_IDS)):
            n = max(1, int(rng.integers(1, 3)))
        else:
            n = int(rng.integers(0, 4))

        competitor = PLANTED_COMPETITOR_MAP.get(acc_id)
        is_loyal = acc_id in FP_LOYAL_IDS
        fp_type = FP_TYPE_MAP.get(acc_id)
        items = _gen_content(
            acct_dict, n, _transcript_prompt, _template_transcripts,
            llm, rng, competitor, is_loyal, "transcript", max_tokens=2500, fp_type=fp_type,
        )
        for j, item in enumerate(items):
            item["transcript_id"] = f"trx_{acc_id}_{j + 1:02d}"
            item["account_id"] = acc_id
        all_items.extend(items)
    return all_items


# ─────────────────────────────────────────────────────────────────────────────
# Ground truth
# ─────────────────────────────────────────────────────────────────────────────
def build_ground_truth(accounts_df: pd.DataFrame) -> Dict:
    """Build ground_truth.json from the planted case config."""
    acc_lookup = accounts_df.set_index("account_id")["company_name"].to_dict()

    planted_churn = []
    for acc_id in PLANTED_HEALTH_IDS:
        planted_churn.append({
            "account_id": acc_id,
            "company_name": acc_lookup.get(acc_id, ""),
            "signal_mix": "health_only",
            "notes": "usage drop >45% over last 14 days + error rate spike in last 7 days",
        })
    for acc_id in PLANTED_RISK_IDS:
        comp = PLANTED_COMPETITOR_MAP.get(acc_id, "")
        planted_churn.append({
            "account_id": acc_id,
            "company_name": acc_lookup.get(acc_id, ""),
            "signal_mix": "risk_only",
            "notes": f"competitor {comp} mentioned in tickets/transcripts; modest or normal health metrics",
        })
    for acc_id in PLANTED_COMBINED_IDS:
        comp = PLANTED_COMPETITOR_MAP.get(acc_id, "")
        tier = PLANTED_TIER_MAP.get(acc_id, "")
        planted_churn.append({
            "account_id": acc_id,
            "company_name": acc_lookup.get(acc_id, ""),
            "signal_mix": "combined",
            "notes": (
                f"usage drop >45% last 14 days + competitor {comp} mentioned "
                f"+ {tier} tier (high value) — clearest catches"
            ),
        })

    false_positive_traps = []
    for acc_id in FP_SEASONAL_IDS:
        false_positive_traps.append({
            "account_id": acc_id,
            "company_name": acc_lookup.get(acc_id, ""),
            "trap_type": "seasonal_dip",
            "notes": (
                "usage dropped ~40% in last 14 days — looks like health churn but is a slow "
                "business period (team offsite + pre-launch lull). No error spike, no webhook "
                "degradation. Communications explain the context. Should NOT trigger outreach."
            ),
        })
    for acc_id in FP_TRANSIENT_IDS:
        false_positive_traps.append({
            "account_id": acc_id,
            "company_name": acc_lookup.get(acc_id, ""),
            "trap_type": "transient_error_spike",
            "notes": (
                "error rate elevated in last 7 days but NO usage drop. Caused by a DNS "
                "misconfiguration on the customer's own infrastructure — they explain this "
                "in their ticket and transcript. Should NOT trigger outreach."
            ),
        })
    for acc_id in FP_LOYAL_IDS:
        comp = PLANTED_COMPETITOR_MAP.get(acc_id, "")
        false_positive_traps.append({
            "account_id": acc_id,
            "company_name": acc_lookup.get(acc_id, ""),
            "trap_type": "loyal_competitor_mention",
            "notes": (
                f"developer mentions evaluating {comp} but explicitly states "
                f"they decided to stay with DevAPICo"
            ),
        })

    return {
        "generated_at": TODAY.strftime("%Y-%m-%d"),
        "total_planted_churn": len(planted_churn),
        "total_fp_traps": len(false_positive_traps),
        "planted_churn": planted_churn,
        "false_positive_traps": false_positive_traps,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
def print_summary(
    accounts_df: pd.DataFrame,
    usage_df: pd.DataFrame,
    tickets: List[Dict],
    slack: List[Dict],
    transcripts: List[Dict],
    tracker: Optional["_TrackingWrapper"],
) -> None:
    sep = "═" * 56
    print(f"\n{sep}")
    print(" DevAPICo Synthetic Data Generator — Complete")
    print(sep)
    print(f"  Accounts generated  : {len(accounts_df):>6,}")
    print(f"  Usage rows          : {len(usage_df):>6,}  ({len(accounts_df)} accounts × {USAGE_DAYS} days)")
    print(f"  Tickets             : {len(tickets):>6,}")
    print(f"  Slack messages      : {len(slack):>6,}")
    print(f"  Call transcripts    : {len(transcripts):>6,}")
    print()
    print(f"  Planted churn cases : {15:>6}")
    print(f"    Health-only       : {len(PLANTED_HEALTH_IDS):>6}  (acc_0001–acc_0005)")
    print(f"    Risk-only         : {len(PLANTED_RISK_IDS):>6}  (acc_0006–acc_0010)")
    print(f"    Combined          : {len(PLANTED_COMBINED_IDS):>6}  (acc_0011–acc_0015)")
    print()
    print(f"  False-positive traps: {5:>6}")
    print(f"    Seasonal dip      : {len(FP_SEASONAL_IDS):>6}  (acc_0016–acc_0017)")
    print(f"    Transient error   : {len(FP_TRANSIENT_IDS):>6}  (acc_0018–acc_0019)")
    print(f"    Loyal competitor  : {len(FP_LOYAL_IDS):>6}  (acc_0020)")
    print()

    if tracker is not None:
        provider = tracker._provider_name()
        model = tracker._model_name()
        rates = COST_PER_1M.get(provider, COST_PER_1M["openai"])
        cost = (
            tracker.total_input_tokens / 1_000_000 * rates["input"]
            + tracker.total_output_tokens / 1_000_000 * rates["output"]
        )
        print(f"  LLM calls (live)    : {tracker.live_calls:>6,}")
        print(f"  Cache hits          : {tracker.cache_hits:>6,}")
        print(f"  Provider / model    :  {provider} / {model}")
        print(f"  Input tokens        : {tracker.total_input_tokens:>6,}")
        print(f"  Output tokens       : {tracker.total_output_tokens:>6,}")
        print(f"  Est. cost (live)    :  ${cost:.3f}")
    else:
        print("  LLM calls (live)    :      0  (--skip-llm mode)")

    print(sep)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main(out_dir: Path, skip_llm: bool = False) -> None:
    rng = np.random.default_rng(SEED)
    fake = Faker()
    Faker.seed(SEED)

    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Accounts ──────────────────────────────────────────────────────────────
    logger.info("Generating %d accounts...", NUM_ACCOUNTS)
    accounts_df = generate_accounts(rng, fake)
    accounts_df.to_csv(out_dir / "accounts.csv", index=False)
    logger.info("  → accounts.csv (%d rows)", len(accounts_df))

    # ── Usage ─────────────────────────────────────────────────────────────────
    logger.info("Generating usage data (%d accounts × %d days)...", NUM_ACCOUNTS, USAGE_DAYS)
    usage_df = generate_usage(accounts_df, rng)
    usage_df.to_csv(out_dir / "usage.csv", index=False)
    logger.info("  → usage.csv (%d rows)", len(usage_df))

    # ── LLM content ───────────────────────────────────────────────────────────
    tracker: Optional[_TrackingWrapper] = None
    if not skip_llm:
        from llm import get_client
        raw_client = get_client()
        tracker = _TrackingWrapper(raw_client)
        llm: Optional[Any] = tracker
        logger.info("LLM provider: %s / %s", tracker._provider_name(), tracker._model_name())
    else:
        llm = None
        logger.info("--skip-llm: using template content")

    logger.info("Generating tickets...")
    tickets = generate_tickets(accounts_df, llm, rng)
    (out_dir / "tickets.json").write_text(json.dumps(tickets, indent=2))
    logger.info("  → tickets.json (%d tickets)", len(tickets))

    logger.info("Generating Slack messages...")
    slack_msgs = generate_slack(accounts_df, llm, rng)
    (out_dir / "slack.json").write_text(json.dumps(slack_msgs, indent=2))
    logger.info("  → slack.json (%d messages)", len(slack_msgs))

    logger.info("Generating call transcripts...")
    transcripts = generate_transcripts(accounts_df, llm, rng)
    (out_dir / "transcripts.json").write_text(json.dumps(transcripts, indent=2))
    logger.info("  → transcripts.json (%d transcripts)", len(transcripts))

    # ── Ground truth ──────────────────────────────────────────────────────────
    gt = build_ground_truth(accounts_df)
    (out_dir / "ground_truth.json").write_text(json.dumps(gt, indent=2))
    logger.info("  → ground_truth.json")

    print_summary(accounts_df, usage_df, tickets, slack_msgs, transcripts, tracker)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic DevAPICo churn data.")
    parser.add_argument("--out-dir", type=Path, default=Path("data/output"))
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Use template text instead of LLM calls. Fast and free — useful for pipeline testing.",
    )
    args = parser.parse_args()
    main(args.out_dir, skip_llm=args.skip_llm)
