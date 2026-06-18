"""PromptCaptureHandler — minimal AsyncCallbackHandler that captures LLM input prompts.

All other LLM events (tokens, finish, errors) are handled via astream() stream_mode="messages".
This callback is the only place we need a callback handler — prompts are not in the stream.
"""

from __future__ import annotations

import asyncio

from langchain_core.callbacks import AsyncCallbackHandler


class PromptCaptureHandler(AsyncCallbackHandler):
    """Captures only input prompts sent to the LLM.

    All other events come from astream() stream_mode=["updates","tasks","messages"].
    This handler is injected via RunnableConfig(callbacks=[handler]) and the queue
    is drained opportunistically inside the astream() loop.

    Usage:
        prompt_q: asyncio.Queue = asyncio.Queue()
        handler = PromptCaptureHandler(prompt_q)
        lg_config = RunnableConfig(callbacks=[handler])
        async for part in graph.astream(state, config=lg_config, ...):
            while not prompt_q.empty():
                item = prompt_q.get_nowait()   # {"prompt": "..."}
    """

    def __init__(self, queue: asyncio.Queue) -> None:
        super().__init__()
        self._q = queue

    async def on_llm_start(
        self,
        serialized: dict,
        prompts: list[str],
        *,
        run_id,
        **kwargs,
    ) -> None:
        """Called just before each LLM call. Puts the first prompt into the queue."""
        await self._q.put({"prompt": prompts[0] if prompts else ""})
