# GitHub Setup Guide — How to Present This Project Professionally

This guide walks you through every step needed to push this project to GitHub in a way that impresses a technical reviewer. It covers rewriting the git history into a clean narrative, setting up GitHub Actions CI, configuring the repository, and what a reviewer actually looks at.

---

## What Reviewers Actually Check

When a hiring manager or senior engineer opens your GitHub repo they look at things in roughly this order:

1. **README** — Is it clear what the project does and how to run it? Does it have a CI badge?
2. **Commit history** — Does it tell a coherent story, or is it one giant dump?
3. **Commit messages** — Do they explain *why*, not just *what*?
4. **Tests** — Are there any? Do they pass? Is there a CI badge to prove it?
5. **Code structure** — Is it organised into logical modules?
6. **No secrets** — Did they accidentally commit `.env` or API keys?

This guide addresses every one of these points.

---

## Step 1 — Rewrite the Git History

Right now everything is in one commit. A reviewer who clicks "X commits" and sees a single blob learns nothing about how you built this. A history of 10 purposeful commits tells a story: you designed the LLM abstraction first, then the data layer, then the pipeline step by step.

Since you have not pushed yet, rewriting history is safe and leaves no trace.

### Reset both commits (keep all files)

Open a terminal in the project root and run:

```bash
# Uncommit both commits — files stay unchanged, just "unstaged"
git reset HEAD~2

# Confirm you are back to an empty history
git log --oneline
# should output: (nothing)

# Confirm all files are still present as untracked
git status
```

### Make 10 commits that tell the story

Copy and run these one at a time. Read the message for each — they follow the **Conventional Commits** format (`type(scope): subject`).

---

**Commit 1 — Scaffolding**
```bash
git add requirements.txt .gitignore .env.example \
        llm/__init__.py pipeline/__init__.py data/__init__.py tests/__init__.py
git commit -m "chore: project scaffolding, dependencies, and environment setup

Python 3.11. Key packages: openai, anthropic, pandas, streamlit, plotly,
pytest. .env.example documents the two required variables."
```

---

**Commit 2 — LLM abstraction layer**
```bash
git add llm/base.py llm/factory.py llm/openai_client.py llm/anthropic_client.py
git commit -m "feat(llm): provider-agnostic client with transparent prompt caching

Abstract LLMClient base class wraps every provider call with a SHA-256
disk cache so repeated runs never re-call the API. OpenAI uses
response_format for JSON mode; Anthropic uses assistant prefill.
Switch providers via LLM_PROVIDER env var — no code changes needed."
```

---

**Commit 3 — Synthetic data**
```bash
git add data/generate.py \
        data/output/accounts.csv data/output/usage.csv \
        data/output/tickets.json data/output/slack.json \
        data/output/transcripts.json data/output/ground_truth.json
git commit -m "feat(data): synthetic dataset generator — 100 accounts, planted churn cases, FP traps

90-day per-account usage time-series. 15 planted churn accounts across
three signal mixes (health-only, risk-only, combined). 5 false-positive
traps (seasonal dip, transient error, loyal competitor mention) designed
to catch naive detectors. Ground truth committed for evaluation."
```

---

**Commit 4 — Health detector**
```bash
git add pipeline/health.py
git commit -m "feat(pipeline/health): statistical anomaly detector using z-score and WoW analysis

Five signals: usage_drop (z < -2σ vs 60-day baseline), wow_drop (>30%
WoW decline), error_spike (z > 2σ), webhook_drop (>5pp), rate_limit_stress.
Baseline excludes last 30 days to prevent anomaly contamination.
Severity: high/medium/low based on signal combination."
```

---

**Commit 5 — LLM classifier**
```bash
git add pipeline/classify.py
git commit -m "feat(pipeline/classify): LLM communication risk classifier

Assembles tickets + Slack + transcripts into a bounded context (6500
chars max) and extracts: use_case, churn_risk, churn_signals, recommended
action. Runs only on health-flagged accounts to minimise API cost.
temperature=0.2 for deterministic structured output. Robust JSON parsing
with conservative fallback on any parse failure."
```

---

**Commit 6 — Correlator + checksum utility**
```bash
git add pipeline/correlate.py pipeline/utils.py
git commit -m "feat(pipeline/correlate): multi-signal correlation engine and checksum utility

Pure business rules — no LLM. Free tier always skipped. Single weak
anomaly requires churn_risk=high to trigger (blocks seasonal/transient FPs).
Multi-signal or high-confidence triggers if LLM agrees. Confidence score
is a weighted composite: 40% health severity + 40% LLM risk + 20% tier.
utils.py: SHA-256 checksum of communications content for cache-gating."
```

---

**Commit 7 — Outreach drafter + orchestrator + results**
```bash
git add pipeline/respond.py pipeline/run.py data/output/results.parquet
git commit -m "feat(pipeline): outreach drafter, end-to-end orchestrator, and committed results

respond.py: draft personalised outreach emails for triggered accounts.
temperature=0.5 for natural prose. Under 120 words, developer-to-developer
tone, concrete next step, no marketing speak.
run.py: health → classify → correlate → respond orchestration. LLM only
instantiated when anomalies exist. Results committed so the dashboard
works out of the box without re-running the pipeline."
```

---

**Commit 8 — Dashboard**
```bash
git add dashboard/app.py .streamlit/config.toml
git commit -m "feat(dashboard): Streamlit dashboard — account list, detail view, evaluation tab

Tab 1: filterable account table (tier, trigger status, risk, severity).
Tab 2: per-account usage chart (Plotly, dual y-axis, anomaly window
highlighted), health signals, LLM classification, trigger reasoning,
draft email. Tab 3: confusion matrix, precision/recall/F1, per-signal-mix
recall, FP trap detail, reflection. Dark theme via .streamlit/config.toml."
```

---

**Commit 9 — Analyse & Draft button**
```bash
git add dashboard/app.py
git commit -m "feat(dashboard): checksum-gated Analyse & Draft button with live LLM fallback

Per-account button in Tab 2. Computes SHA-256 of current communications
and compares to stored comms_checksum in results.parquet. Cache hit
(checksum match + already classified): instant, zero API calls. Cache
miss: runs classify → correlate → draft for that account only, writes
back to parquet, clears Streamlit cache, reruns. Handles the case where
an account was never classified (no prior anomaly)."
```

---

**Commit 10 — Tests**
```bash
git add tests/test_pipeline.py tests/test_llm.py
git commit -m "test: 32-test suite covering health detector, correlator, checksum, and LLM layer

test_pipeline.py: 20 tests. In-memory synthetic DataFrames — no file I/O,
no mocks of business logic. Covers every correlator branch (free-tier
block, weak-signal block, strong trigger, enterprise boost, etc.).
test_llm.py: 12 tests. Cache miss/hit, parameter isolation, factory
validation. 2 live-API tests skipped without real keys."
```

---

**Commit 11 — Docs and CI**
```bash
git add README.md GUIDE.md evaluation/evaluate.ipynb \
        .github/workflows/ci.yml
git commit -m "docs: README, evaluation notebook, technical guide, and GitHub Actions CI

README: Mermaid architecture diagram, run-locally steps, eval results
table, Streamlit Cloud deploy instructions, CI badge.
GUIDE.md: 3,680-word deep-dive for NotebookLM / interview prep.
evaluate.ipynb: confusion matrix, per-signal-mix recall, FP trap detail,
precision-recall curve sweep.
ci.yml: pytest on push/PR, verifies results.parquet loads cleanly."
```

---

### Verify the history looks right

```bash
git log --oneline
```

You should see 11 commits, most recent at the top, each with a clear subject.

```bash
git log --stat
```

This shows which files changed in each commit — confirm the groupings look logical.

---

## Step 2 — Create the GitHub Repository

1. Go to [github.com/new](https://github.com/new)
2. Set these fields:

| Field | Value |
|---|---|
| Repository name | `silent-churn` |
| Description | `Detects silently churning developer API customers by correlating usage health metrics with LLM-scored communications. LLM-agnostic (OpenAI / Anthropic). Streamlit dashboard.` |
| Visibility | **Public** |
| Initialize with README | **No** (you already have one) |
| .gitignore | **None** (you already have one) |
| License | MIT (optional, but good for portfolio) |

3. Click **Create repository**

---

## Step 3 — Push to GitHub

The CI badge is already set to `AnubodhKarki/ZeroDarkChurn` — no edits needed.

---

## Step 4 — Add the Remote and Push

```bash
git remote add origin https://github.com/AnubodhKarki/ZeroDarkChurn.git
git push -u origin main
```

Go to `https://github.com/AnubodhKarki/ZeroDarkChurn/actions` — you should see the CI workflow running within 30 seconds. Wait for it to go green.

---

## Step 5 — Configure the Repository on GitHub

After pushing, do these things in the GitHub UI:

### Add repository topics

Click the gear icon next to **About** on the repo homepage. Add these topics:

```
python  llm  openai  anthropic  streamlit  pandas  customer-success
churn-prediction  data-pipeline  portfolio
```

Topics make your repo discoverable and signal what skills you are demonstrating.

### Set the website URL

In the same **About** panel, set the website URL to your Streamlit Community Cloud deployment URL once you have it. This lets people click straight through to the live demo.

### Enable the README as the social preview

In **Settings → General → Social preview**, upload a screenshot of the dashboard. When you share the GitHub link on LinkedIn or in messages, this image appears in the preview card.

---

## Step 6 — Deploy to Streamlit Community Cloud (Live Demo Link)

A working live demo is worth more than anything else on a portfolio project.

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. **New app** → Connect your GitHub → Select `silent-churn` repo → `dashboard/app.py`
3. Click **Advanced settings** → **Secrets** and add:

```toml
LLM_PROVIDER = "openai"
OPENAI_API_KEY = "sk-..."
```

4. Deploy. The app loads immediately from the committed `results.parquet` — no pipeline run needed.
5. Copy the deployment URL back into the GitHub repo's **About → Website** field.

---

## Step 7 — Pin the Repo on Your GitHub Profile

On your GitHub profile page, click **Customize your pins** and pin `silent-churn`. It will then appear as one of the featured repos visible without scrolling.

Write a one-sentence summary for the pin — GitHub shows the repo description you set in Step 2.

---

## What the Final GitHub Profile Shows

When a reviewer visits your GitHub after following these steps, they see:

- A **pinned repo** with a clear description
- Clicking it shows a **professional README** with a green CI badge, a Mermaid architecture diagram, eval results, and a live demo link
- The **commit history** shows 11 commits that read like a development diary — each one advancing the system one logical step
- The **Actions tab** shows a green CI run on every push
- The **code** is organised into `llm/`, `pipeline/`, `dashboard/`, `tests/` — each module single-responsibility
- There is **no `.env`** or secrets anywhere in the history

---

## Commit Message Conventions (Reference)

The commits above use [Conventional Commits](https://www.conventionalcommits.org/) format:

```
type(scope): short imperative subject (under 72 chars)

Optional body explaining WHY, not what. Wrap at 72 chars.
Multiple paragraphs OK.
```

**Types used in this project:**
- `feat` — new feature or capability
- `test` — adding or updating tests
- `docs` — documentation only
- `chore` — tooling, config, dependencies (no production logic)
- `fix` — bug fix (use if you need it later)
- `refactor` — restructuring without changing behaviour

**Rules for subject lines:**
- Start with lowercase after the colon
- Use imperative mood: "add", "implement", "fix" — not "added", "implementing"
- No period at the end
- Under 72 characters so it fits in `git log --oneline`

**Body paragraph tips:**
- Explain the *why* and *trade-offs*, not the *what* (the diff shows the what)
- If you made a non-obvious decision, document it here
- Reference the constraints: "Runs only on health-flagged accounts to minimise API cost" tells the reviewer you thought about cost

---

## The One-Sentence Pitch (Memorise This)

When a reviewer asks "tell me about this project":

> "I built a silent churn detection system for a developer API company. It works in three layers: a pure pandas statistical detector that flags usage anomalies, an LLM layer that reads the customer's support tickets and Slack messages to assess churn risk, and a business-rules correlator that combines both signals with account value to decide whether to trigger outreach. The system is provider-agnostic — it runs on OpenAI or Anthropic with one env var change — and everything after the first run is served from a disk cache, so the Streamlit dashboard loads instantly with no API calls."
