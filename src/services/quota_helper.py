"""Quota reset time helper.

Mirrors the heuristic from scripts/quota_heuristic.sh and suite_runner.sh:
- QUOTA_RESET_ANCHOR (HH:MM, local time) marks a recurring reset point.
- QUOTA_CYCLE_HOURS defines the period between resets.
- seconds_until_next_reset() computes how many seconds from *now* until the
  next reset window opens.

Assumption: server local time matches the anchor timezone (same as the shell
scripts). No DST conversion is performed — documented limitation.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional


def seconds_until_next_reset(
    now: Optional[datetime] = None,
    anchor: Optional[str] = None,
    cycle_hours: Optional[int] = None,
) -> int:
    """Return seconds from *now* until the next quota reset window.

    Args:
        now: Current datetime (naive, local time). Defaults to datetime.now()
            (no timezone info — matches server local-time assumption).
        anchor: HH:MM string for the daily anchor reset time.
            Defaults to settings.QUOTA_RESET_ANCHOR.
        cycle_hours: Cycle length in hours. Defaults to settings.QUOTA_CYCLE_HOURS.

    Returns:
        Non-negative integer number of seconds until next reset.
        Minimum return value is 1 (never 0).

    Notes:
        The algorithm mirrors quota_heuristic.sh section 2:
        1. Find today's anchor as a datetime.
        2. Walk backward/forward in cycle_hours steps to find PREV_RESET
           (the most recent reset <= now).
        3. NEXT_RESET = PREV_RESET + cycle_hours.
        4. Return (NEXT_RESET - now).total_seconds(), floored to int.
    """
    if now is None:
        now = datetime.now()

    if anchor is None:
        from src.config.settings import settings

        anchor = settings.QUOTA_RESET_ANCHOR
    if cycle_hours is None:
        from src.config.settings import settings

        cycle_hours = settings.QUOTA_CYCLE_HOURS

    # Parse anchor HH:MM
    try:
        anchor_h, anchor_m = (int(p) for p in anchor.split(":"))
    except (ValueError, AttributeError):
        # Fallback to 00:30 on bad config
        anchor_h, anchor_m = 0, 30

    cycle_delta = timedelta(hours=cycle_hours)

    # Today's anchor as a naive datetime
    today_anchor = now.replace(hour=anchor_h, minute=anchor_m, second=0, microsecond=0)

    # Find PREV_RESET: the most recent reset <= now.
    # Start from today's anchor and walk backward/forward in cycle steps.
    prev_reset = today_anchor

    # If today's anchor is in the future, step backward until prev_reset <= now.
    while prev_reset > now:
        prev_reset -= cycle_delta

    # Step forward while the next step is still <= now.
    while prev_reset + cycle_delta <= now:
        prev_reset += cycle_delta

    next_reset = prev_reset + cycle_delta
    secs = int((next_reset - now).total_seconds())
    return max(1, secs)
