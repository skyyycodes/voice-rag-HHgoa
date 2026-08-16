"""Tool registry.

Each pipeline stage is registered as a named callable with a description and a
recorded invocation history. The registry is what turns "the pipeline calls
some functions" into an inspectable surface: stages can be listed, swapped for
stubs in tests, invoked individually from the debug API, and every call is
timed and its outcome recorded without wrapping each call site by hand.

Deliberately synchronous and dependency-free — the stages it wraps are pure
CPU work measured in microseconds, and an async registry would add an event
loop hop to every one of them for no benefit. The two genuinely async stages
(STT, LLM) are awaited directly by the orchestrator under their own policy.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from .contracts import ToolCall


@dataclass(slots=True)
class Tool:
    name: str
    fn: Callable[..., Any]
    description: str


@dataclass
class ToolRegistry:
    tools: dict[str, Tool] = field(default_factory=dict)
    calls: list[ToolCall] = field(default_factory=list)
    # Bounded so a long-lived server process cannot grow this without limit.
    max_history: int = 256

    def register(self, name: str, fn: Callable[..., Any], description: str = "") -> None:
        if name in self.tools:
            raise ValueError(f"tool already registered: {name}")
        self.tools[name] = Tool(name=name, fn=fn, description=description)

    def call(self, name: str, **kwargs: Any) -> Any:
        tool = self.tools.get(name)
        if tool is None:
            raise KeyError(f"unknown tool: {name!r} (have: {sorted(self.tools)})")

        started = time.perf_counter()
        try:
            result = tool.fn(**kwargs)
        except Exception as exc:
            self._record(name, kwargs, started, ok=False, error=f"{type(exc).__name__}: {exc}")
            raise
        self._record(name, kwargs, started, ok=True)
        return result

    def _record(
        self, name: str, kwargs: dict, started: float, ok: bool, error: str = ""
    ) -> None:
        self.calls.append(
            ToolCall(
                name=name,
                # Argument *shapes*, not values: a full candidate list or raw
                # query text in the history would balloon memory and put user
                # input into a structure that gets serialised into debug output.
                args={k: _summarise(v) for k, v in kwargs.items()},
                ms=round((time.perf_counter() - started) * 1000, 3),
                ok=ok,
                error=error,
            )
        )
        if len(self.calls) > self.max_history:
            del self.calls[: len(self.calls) - self.max_history]

    def describe(self) -> list[dict[str, str]]:
        return [
            {"name": t.name, "description": t.description}
            for t in sorted(self.tools.values(), key=lambda t: t.name)
        ]


def _summarise(value: Any) -> Any:
    if isinstance(value, str):
        return f"<str len={len(value)}>"
    if isinstance(value, (list, tuple)):
        return f"<{type(value).__name__} n={len(value)}>"
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return f"<{type(value).__name__}>"
