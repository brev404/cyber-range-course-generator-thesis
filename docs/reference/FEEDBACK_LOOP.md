# Feedback Loop — RL Foundation

This document covers the reward signal definition, run history schema, and how to use the feedback loop for A/B analysis across prompt versions.

---

## 1. Reward Signal Definition

A **RewardRecord** is produced by `compute_reward(ranking_reports, judge_model)` in `src/utils/feedback_utils.py` after every ranking pass.

| Field | Type | Description |
|---|---|---|
| `run_id` | str (UUID4) | Unique identifier for this pipeline run |
| `timestamp` | str (ISO 8601) | UTC timestamp of reward computation |
| `judge_model` | str | Model used as the ranking judge (e.g. `claude-haiku-4-5-20251001`) |
| `per_challenge_scores` | dict[str, float] | `challenge_id → overall_score` for each evaluated challenge |
| `mean_tech` | float | Mean technical score across all challenges (rounded to 3dp) |
| `mean_ped` | float | Mean pedagogical score across all challenges (rounded to 3dp) |
| `pass_rate` | float | Fraction of challenges where both tech and ped ≥ `FEEDBACK_REWARD_THRESHOLD` |
| `reward` | bool | True only when the judge guard passes and both means ≥ threshold (see §2) |
| `prompt_version` | str (8-char hex) | SHA-256 hash prefix of the content-generation system prompt |

### reward=True conditions (all must hold)

1. `judge_model == settings.FEEDBACK_JUDGE_MODEL` — judge guard (see §2)
2. `mean_tech >= settings.FEEDBACK_REWARD_THRESHOLD`
3. `mean_ped >= settings.FEEDBACK_REWARD_THRESHOLD`

---

## 2. Why FEEDBACK_JUDGE_MODEL Must Be Fixed Across a Series

`FEEDBACK_JUDGE_MODEL` (default: `claude-haiku-4-5-20251001`) is the **single fixed judge** whose scores are admitted into the feedback store.

If multiple models are used as judges across runs, their scores are not comparable: a run judged by GPT-4 may score higher than one judged by Haiku even if the content quality is identical. Mixing judges would make A/B comparisons across `prompt_version` meaningless.

The guard `reward = (judge_model == FEEDBACK_JUDGE_MODEL) and ...` ensures that only runs evaluated by the designated judge contribute `reward=True` records to the history. Records with a different judge are still stored (for auditing) but always have `reward=False`.

**To change the fixed judge:** update `FEEDBACK_JUDGE_MODEL` in `.env` and discard (or archive) all previously collected records before resuming, because historical comparisons would be invalid.

---

## 3. How to Enable

The feedback loop is **opt-in**. Default: `FEEDBACK_ENABLED=False`.

To enable for an experiment run:

```bash
# In .env or as an env var override:
FEEDBACK_ENABLED=True
FEEDBACK_JUDGE_MODEL=claude-haiku-4-5-20251001   # must match RANKING_MODEL
FEEDBACK_REWARD_THRESHOLD=7.0
```

Then run the pipeline normally. After ranking, call `post_ranking_feedback_to_langsmith(reward_record)` from `langsmith_service` — this is already wired into the ranking feedback path and will write to `data/feedback/run_history.jsonl` automatically.

---

## 4. Run History Schema

File: `data/feedback/run_history.jsonl`

Format: **JSONL** — one JSON object per line, newline-delimited. The file is created on first write (directory created if needed); subsequent calls append.

Example line:
```json
{"run_id":"550e8400-e29b-41d4-a716-446655440000","timestamp":"2026-05-03T10:00:00+00:00","judge_model":"claude-haiku-4-5-20251001","per_challenge_scores":{"crypto_001":8.5,"web_002":9.0},"mean_tech":8.2,"mean_ped":8.7,"pass_rate":0.5,"reward":true,"prompt_version":"a1b2c3d4"}
```

---

## 5. Querying History for A/B Analysis

### Group by prompt_version and count reward=True

```bash
jq -r 'select(.reward == true) | .prompt_version' data/feedback/run_history.jsonl \
  | sort | uniq -c | sort -rn
```

### Mean tech score per prompt version

```bash
jq -r '[.prompt_version, .mean_tech] | @tsv' data/feedback/run_history.jsonl \
  | awk '{sum[$1]+=$2; count[$1]++} END {for (v in sum) print v, sum[v]/count[v]}' \
  | sort -k2 -rn
```

### All records for a specific judge model

```bash
jq 'select(.judge_model == "claude-haiku-4-5-20251001")' data/feedback/run_history.jsonl
```

### Pass rate trend (chronological)

```bash
jq -r '[.timestamp, .pass_rate, .prompt_version] | @tsv' data/feedback/run_history.jsonl
```

---

## 6. Prompt Version

`prompt_version` is the first 8 characters of the SHA-256 hash of the `_WRITEUP_SYSTEM` string in `src/agents/content_generation_agent.py`. It is deterministic: the same prompt always produces the same hash. This allows grouping records by prompt revision without storing the full prompt text.

When the system prompt is modified (e.g. adding a new anti-pattern rule or changing section order), the hash changes, automatically starting a new group in the history.

---

## 7. Future RL Integration

The current system is a **passive feedback store** — it records reward signals without updating the prompt. A future RL loop would:

1. **Policy**: parameterise the prompt as a set of toggleable instructions or few-shot examples.
2. **Reward signal**: use `reward` (bool) or `mean_tech`/`mean_ped` as the scalar reward.
3. **Update step**: after N runs, select the prompt variant with the highest `pass_rate` or mean reward among records with a matching `judge_model`.
4. **Exploration**: occasionally test new prompt variants (different `prompt_version`) to avoid reward hacking.

The `run_history.jsonl` provides the data needed for offline policy gradient or bandit-style updates without requiring online RL infrastructure. The judge-model guard ensures the reward signal remains unbiased across policy updates.
