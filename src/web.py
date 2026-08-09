"""A small local web UI in front of the orchestrator.

    ./venv/bin/python src/web.py          # then open http://127.0.0.1:8000

Built on starlette + uvicorn, which are already installed as transitive
dependencies — this adds nothing to requirements.txt.

A question takes ~20s (router + two sub-agents + synthesis), so the answer is
streamed over Server-Sent Events: the trace spans that `trace.py` already
produces are pushed to the browser as they close, and the UI shows which
specialist is working. The finished answer arrives as the final event.

Local and single-user by design: it binds to 127.0.0.1 and has no auth. Don't
expose it to a network.
"""
import asyncio
import json
import re
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import uvicorn
from starlette.applications import Starlette
from starlette.responses import FileResponse, JSONResponse, StreamingResponse
from starlette.routing import Route

from llm import CACHE_ENABLED, DEFAULT_MODEL
from orchestrator import SPECIALISTS, Orchestrator
from technicals import TICKERS

HERE = Path(__file__).resolve().parent
PAGE = HERE / "web_ui.html"

# One orchestrator for the process: it holds the loaded embedding model and the
# Chroma handle, which are expensive to build per request.
_orch = None
_orch_lock = threading.Lock()


def orchestrator() -> Orchestrator:
    global _orch
    with _orch_lock:
        if _orch is None:
            _orch = Orchestrator()
    return _orch


async def index(request):
    return FileResponse(PAGE)


async def meta(request):
    """What the UI needs to describe the system it is talking to."""
    return JSONResponse({
        "model": DEFAULT_MODEL,
        "cache_enabled": CACHE_ENABLED,
        "tickers": sorted(TICKERS),
        "specialists": SPECIALISTS,
    })


def readable_error(e: Exception) -> str:
    """Turn a provider exception into one line a person can act on.

    LiteLLM surfaces provider errors as a wall of embedded JSON. The UI wants
    the human-readable part, and for quota — by far the most common failure
    here — the actual remedy.
    """
    raw = str(e)
    if "RESOURCE_EXHAUSTED" in raw or "RateLimitError" in type(e).__name__:
        scope = "daily" if re.search(r"PerDay|per.day", raw, re.I) else "per-minute"
        return (f"LLM {scope} quota exhausted for {DEFAULT_MODEL}. One question costs "
                f"3–4 calls. Quotas are per-model, so switching gives a fresh allowance: "
                f"restart with LLM_MODEL=gemini/gemini-flash-lite-latest")
    # Pull the provider's own "message" field out of the embedded JSON if present.
    m = re.search(r'"message"\s*:\s*"([^"]{10,300})', raw)
    detail = m.group(1) if m else raw.split("\n")[0]
    return f"{type(e).__name__}: {detail[:300]}"


async def ask(request):
    """Run one question, streaming trace spans then the final answer as SSE."""
    question = (request.query_params.get("q") or "").strip()
    if not question:
        return JSONResponse({"error": "empty question"}, status_code=400)
    try:
        k = max(1, min(int(request.query_params.get("k", 6)), 30))
    except ValueError:
        k = 6

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def emit(item):
        # Called from the worker thread — hop back onto the event loop.
        loop.call_soon_threadsafe(queue.put_nowait, item)

    def run():
        try:
            res = orchestrator().ask(question, k=k, on_span=lambda s: emit({"type": "span", **s}))
            emit({
                "type": "done",
                "answer": res.answer.answer,
                "confidence": res.answer.confidence,
                "insufficient_evidence": res.answer.insufficient_evidence,
                "routed": res.routed,
                "route_reason": res.route_reason,
                "ticker": res.ticker,
                "context_chars": res.context_chars,
                "trace_id": res.trace.get("trace_id"),
                "duration_ms": res.trace.get("duration_ms"),
                "citations": [
                    {"label": c.label(), "url": c.source_url,
                     "kind": "market" if c.filing_type.startswith("market-data") else "filing"}
                    for c in res.answer.citations
                ],
                "sub_answers": [
                    {"agent": n, "answer": a.answer, "chars": a.summary_chars(),
                     "confidence": a.confidence, "insufficient": a.insufficient_evidence}
                    for n, a in res.sub_answers.items()
                ],
            })
        except Exception as e:                       # noqa: BLE001 - surface it in the UI
            emit({"type": "error", "error": readable_error(e)})

    threading.Thread(target=run, daemon=True).start()

    async def events():
        while True:
            item = await queue.get()
            yield f"data: {json.dumps(item)}\n\n"
            if item["type"] in ("done", "error"):
                return

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


app = Starlette(routes=[
    Route("/", index),
    Route("/api/meta", meta),
    Route("/api/ask", ask),
])


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Local web UI for the finance agent.")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()

    print(f"\n  Finance agent UI -> http://{args.host}:{args.port}")
    print(f"  model: {DEFAULT_MODEL}   cache: {'on' if CACHE_ENABLED else 'off'}")
    print("  note: 4 LLM calls per two-specialist question, 3 for one, 1 if declined\n")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
