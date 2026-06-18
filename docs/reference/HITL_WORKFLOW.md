# HITL Workflow

## When HITL fires

The HITL (Human-in-the-Loop) node is reached under two conditions:

1. **Quality failure**: A challenge's `technical_review.score` or `pedagogical_review.score`
   is below `RANKING_PASS_THRESHOLD` (default 9.0) after the auto-refinement cap is exhausted.
2. **Max refinements reached**: `refinement_count >= MAX_REFINEMENT_ROUNDS` (default 5)
   and `MAX_REFINEMENT_STRATEGY = "hitl"` (the default strategy).

HITL runs inside a LangGraph interrupt/resume cycle. The graph pauses, presents a review
summary to the operator via the CLI, and waits for a `Command(resume=...)` to continue.

## What the display shows

When HITL fires, the CLI prints:

- **Routing cause line**: `Routing cause: quality failure` or
  `Routing cause: max refinements reached (round N)`
- **Iteration counter**: `HITL iteration X / Y` where Y is `max_hitl_iterations` (default 3)
- **Challenge review table**: one row per challenge showing:
  - `challenge_id`
  - `technical` score (from `technical_review.score`)
  - `pedagogical` score (from `pedagogical_review.score`)
  - `overall` score
  - `failed_dimensions` — dimension names from `challenge_dimension_scores` where
    `score < RANKING_PASS_THRESHOLD` (column omitted when dimension scores unavailable)

Rich is used for the table if installed; a plain-text fallback is used otherwise.

## Available actions

| Input | Action | Effect |
|-------|--------|--------|
| `approve_all` or `y` | Approve all | Accept all challenges as-is; route to END. |
| `approve <id1,id2>` | Partial approve | Accept named challenges; remaining re-enter `content_generation`. |
| `edit_retry` or `n` | Edit and retry | Prompt for operator hint; inject into `human_feedback`; re-run refinement. |
| `abort` | Abort | Stop the run cleanly (`SystemExit(0)`). |

Input is case-insensitive. Empty input or `y`/`yes` defaults to `approve_all`.

## State changes per action

### approve_all
- `hitl_approved = True`
- `iteration_count += 1`
- `route_after_hitl` → END

### approve \<ids\>
- `hitl_approved = False`
- `content_generation_subset_ids = [non-approved IDs]`
- `ranking_subset_ids = [non-approved IDs]`
- `iteration_count += 1`
- `route_after_hitl` → `content_generation` (only re-processes remaining challenges)

### edit_retry
- Operator hint appended to `human_feedback[challenge_id]` for every challenge
  with `overall_score < RANKING_PASS_THRESHOLD`
- `hitl_approved = False`
- `iteration_count += 1`
- `route_after_hitl` → `content_generation` while `iteration_count < max_hitl_iterations`;
  then → END when cap is reached

### abort
- Raises `SystemExit(0)` immediately — no state update, no graph resume

## HITL loop cap

After `max_hitl_iterations` (default 3) HITL rounds, `route_after_hitl` in `graph.py`
routes to END regardless of `hitl_approved`. This prevents infinite loops when the LLM
cannot satisfy the rubric within the allowed budget.

## Resume payload format (programmatic / API use)

When invoking the graph programmatically (not via CLI), pass the resume value as:

```python
from langgraph.types import Command

# Approve all
app.invoke(Command(resume={"action": "approve_all"}), config=config)

# Approve subset
app.invoke(Command(resume={"action": "approve_ids", "ids": ["web/sqli", "crypto/rsa"]}), config=config)

# Edit and retry with hint
app.invoke(Command(resume={"action": "edit_retry", "hint": "Improve XSS section with DOM-based example"}), config=config)

# Legacy format (still accepted for backward compatibility)
app.invoke(Command(resume={"approved": True}), config=config)
app.invoke(Command(resume={"approved": False, "human_feedback": {"web/sqli": ["Add CVE reference"]}}), config=config)
```

## Key source files

- `src/agents/hitl_agent.py` — `run_hitl_agent`, `print_hitl_summary`, `_parse_resume_value_extended`
- `src/core/graph.py` — `route_after_hitl` routing function
- `src/main.py` — `_invoke_graph_once` CLI loop handling GraphInterrupt
