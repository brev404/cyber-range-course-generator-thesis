"""Duplicate variant: byte-for-byte re-export of baseline constants."""

from src.agents.content_generation_agent import _WRITEUP_SYSTEM as WRITEUP_SYSTEM
from src.agents.ranking_agent import _PEDAGOGICAL_SYSTEM as PEDAGOGICAL_SYSTEM
from src.agents.ranking_agent import _TECHNICAL_SYSTEM as TECHNICAL_SYSTEM

__all__ = ["WRITEUP_SYSTEM", "TECHNICAL_SYSTEM", "PEDAGOGICAL_SYSTEM"]
