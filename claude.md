# pocket-portfolio-assistant — working guide

A personal investment assistant: MCP servers expose data sources as tools, an
orchestrator routes questions to specialist sub-agents, and each answer is
grounded and cited.

**The plan and current status live in `docs/FINANCE-AGENT-PROJECT-PLAN.md` — that
file is the single source of truth. Read its "Current status" section first.**
Design decisions and how to reverse each one: `docs/PHASE2-DESIGN-DECISIONS.md`.

Both are **local-only** — gitignored, so they aren't part of the public repo. If
you cloned this and they're missing, that's why. The published page is
`docs/index.html`.

*(This file used to duplicate the plan and drifted badly out of date. Don't
restate the plan here — link to it.)*

---

## Layout

```
src/
  config.py       paths, embedding model, chunking constants
  fetch.py        EDGAR download (raw HTML cached to data/raw/)
  parse.py        HTML -> text;  sections.py  item boundary detection
  chunk.py        section-aware chunking;  ingest.py  embed + store in Chroma
  retrieval.py    dense + temporal retrieval;  query.py  retrieval CLI
  tools.py        search_filings() — the filings tool boundary
  mcp_server.py   MCP server over the filings tools
  marketdata.py   market tool boundary: quote / valuation / history / technicals
  market_mcp_server.py   MCP server over the market tools
  contracts.py    AgentAnswer + Citation — the locked sub-agent hand-off schema
  llm.py          provider-agnostic LLM (LiteLLM) + retry, failover, disk cache
  fundamentals.py Fundamentals sub-agent (filings)
  technicals.py   Technicals sub-agent (market data)
  orchestrator.py router + fan-out/fan-in + synthesis
  trace.py        per-run structured traces -> data/traces/
  agent.py        single-agent CLI (fundamentals)
tests/                             pytest suite — periods, filters, contracts
eval/
  eval_set.json / run_eval.py        retrieval recall@k (Phase 1 M3)
  run_temporal.py                    period coverage (Phase 1 M5)
  routing_set.json / run_routing.py  routing accuracy + reconciliation (Phase 2 M5)
  context_isolation.py               orchestrator context stays bounded (Phase 2 M4)
```

## Running things

Always use the venv (`./venv/bin/python`), Python 3.14.

```bash
# local web UI (streams progress; http://127.0.0.1:8000)
./venv/bin/python src/web.py

# ask the full multi-agent system
./venv/bin/python src/orchestrator.py "Is NVDA expensive right now?" -v

# single specialist
./venv/bin/python src/agent.py "What are NVDA's risk factors?" --ticker NVDA

# tests (fast, no network, no LLM quota)
./venv/bin/python -m pytest tests/ -q

# evals
./venv/bin/python eval/run_routing.py            # routing accuracy
./venv/bin/python eval/run_routing.py --reconcile
./venv/bin/python eval/context_isolation.py
./venv/bin/python eval/run_eval.py               # retrieval recall
```

Set these when running anything that loads the embedding model, or it will try
to reach Hugging Face and stall:

```bash
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
```

## Gotchas worth knowing

- **LLM quota is the binding constraint, and the model choice matters enormously.**
  One question costs **1 call** if the router declines it, **3** for a single
  specialist, **4** for a two-specialist question (route + 2 sub-agents + synthesis).
  - `gemini-flash-latest` resolves to the newest Flash, which carries a **20
    requests/day** preview quota — measured, not documented. Avoid it.
  - The default is now `gemini-flash-lite-latest`, which runs the documented free
    tier (~1,500 RPD, 30 RPM) and produced all the Phase 2 eval numbers.
  - Quotas are **per-model**, so switching `LLM_MODEL` always buys a fresh allowance.
  - `llm.py` fails fast on daily quotas (waiting cannot help) and retries per-minute
    ones honoring the provider's own `retryDelay`.
  - For real headroom add Groq to `LLM_FALLBACK_MODELS` (30 RPM / 1,000 RPD free,
    no card). Failover is already wired — it is configuration, not code.
- **Completions are cached** to `data/cache/llm/`, keyed on the exact prompt, so
  re-running an eval is free. Editing a prompt invalidates its entries. Force
  live calls with `LLM_CACHE=0`.
- **Market data is yfinance by default** (keyless). Set `FINNHUB_API_KEY` to use
  Finnhub for quotes/valuation instead. Price history is always yfinance.
- **Sub-agents must return summaries, never raw dumps.** The orchestrator only
  ever sees `AgentAnswer` text plus citation labels; `eval/context_isolation.py`
  is what proves this still holds. If you add a sub-agent, it implements
  `contracts.AgentAnswer` and stays within `SUMMARY_CHAR_BUDGET`.
- **Deterministic math belongs in the tool, not the prompt.** Technical metrics
  are computed in `marketdata.py` over the full price series; the LLM is told not
  to do arithmetic.
- **Periods use the company's own fiscal labels**, read from each filing's inline
  XBRL (`dei:DocumentFiscalPeriodFocus`). NVDA's quarter ending 2025-07-27 is
  `Q2 FY2026`, *not* calendar `Q3 2025`. Every chunk also carries `period_end`
  (calendar anchor, and the join key to price data), `calendar_quarter`, and
  `period_type`. A bare `"Q2 2026"` in a query matches both readings rather than
  guessing. **Temporal fan-out excludes annual filings** — a fiscal year contains
  its own quarters, so mixing them invites a year-vs-quarter jump to read as a
  trend. Re-label without re-downloading: `python src/fetch.py --relabel`, then
  `python src/ingest.py --all --reset`.

## Conventions

- Every data source sits behind an MCP interface.
- Tools return structured summaries, not raw vendor JSON.
- Nothing ships without a number: each milestone has a checkpoint in the plan,
  and retrieval/routing changes get measured against the eval sets.
