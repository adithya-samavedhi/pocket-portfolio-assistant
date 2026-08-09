"""Structured tracing for multi-agent runs (Phase 2, Milestone 5).

Multi-agent debugging without traces is miserable: when an answer is wrong you
need to know which agent was wrong, what it was asked, and how big its summary
was. Every orchestrator run writes one JSON trace to data/traces/.

Deliberately dependency-free (no Langfuse/LangSmith) — a run is a small tree of
spans, and the questions we need answered are local: what was routed, what each
sub-agent returned, and whether the orchestrator's context stayed bounded.
"""
import json
import sys
import time
import uuid
from datetime import datetime, timezone

import config

TRACE_DIR = config.DATA / "traces"


class Trace:
    """One orchestrator run. Spans are appended in order; timings are wall-clock."""

    def __init__(self, question: str, verbose: bool = False, on_span=None):
        self.id = uuid.uuid4().hex[:12]
        self.question = question
        self.verbose = verbose
        # Optional callback fired as each span closes, so a caller can watch a
        # run unfold instead of waiting for the finished trace (the web UI uses
        # it to stream progress over a ~20s multi-agent run).
        self.on_span = on_span
        self.started = time.time()
        self.spans: list[dict] = []

    def span(self, name: str, **fields):
        """Record a completed step. `fields` are whatever matters for that step."""
        span = {"name": name, "at_ms": round((time.time() - self.started) * 1000), **fields}
        self.spans.append(span)
        if self.verbose:
            detail = " ".join(f"{k}={v}" for k, v in fields.items() if k != "error")
            print(f"[trace {self.id}] {span['at_ms']:>6}ms  {name:<22} {detail}", file=sys.stderr)
        if self.on_span:
            try:
                self.on_span(span)
            except Exception:            # a broken listener must not kill the run
                pass
        return span

    def timer(self, name: str, **fields):
        """Context manager recording duration and any exception raised inside."""
        return _Timer(self, name, fields)

    def finish(self, **fields) -> dict:
        data = {
            "trace_id": self.id,
            "question": self.question,
            "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "duration_ms": round((time.time() - self.started) * 1000),
            **fields,
            "spans": self.spans,
        }
        TRACE_DIR.mkdir(parents=True, exist_ok=True)
        (TRACE_DIR / f"{self.id}.json").write_text(json.dumps(data, indent=2))
        if self.verbose:
            print(f"[trace {self.id}] written to {TRACE_DIR / (self.id + '.json')}", file=sys.stderr)
        return data


class _Timer:
    def __init__(self, trace, name, fields):
        self.trace, self.name, self.fields = trace, name, fields

    def __enter__(self):
        self.t0 = time.time()
        return self

    def add(self, **fields):
        """Attach fields discovered inside the block (e.g. the result's size)."""
        self.fields.update(fields)

    def __exit__(self, exc_type, exc, tb):
        self.fields["ms"] = round((time.time() - self.t0) * 1000)
        if exc is not None:
            self.fields["error"] = f"{exc_type.__name__}: {exc}"
        self.trace.span(self.name, **self.fields)
        return False
