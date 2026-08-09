# pocket-portfolio-assistant

A research assistant that answers questions about five companies from their SEC
filings and live market data, and cites every claim.

**[How it works →](https://adithya-samavedhi.github.io/pocket-portfolio-assistant/)**

Ask *"is NVDA expensive right now?"* and it consults both the filings and the
market, reconciles them, and returns one answer with its sources — or tells you
plainly when the evidence isn't there.

Covers **AAPL, AMZN, GOOGL, MSFT, NVDA**. Everything runs locally except the LLM
call; embeddings, retrieval and the vector store need no API key.

## How it fits together

A router decides which specialists a question needs, they run in parallel, and
their structured summaries are synthesized into one cited answer. The
orchestrator only ever sees those summaries — never raw passages — so its context
stays bounded no matter how much the specialists read.

| Specialist | Source | Answers |
|---|---|---|
| **Fundamentals** | SEC 10-K / 10-Q | revenue, segments, margins, risk factors, change across quarters |
| **Technicals** | market data | price, P/E and P/S, momentum, 52-week position, drawdown, volatility |

Each data source sits behind its own MCP server. Technical measures are computed
from the full price series in Python — the model interprets numbers, it never
does arithmetic.

Periods use the **company's own fiscal labels**, read from each filing's inline
XBRL: NVIDIA's quarter ending 2025-07-27 is `Q2 FY2026`, not calendar `Q3 2025`.
Each chunk also carries the calendar anchor, so a query for `"Q2 2026"` matches
either reading rather than silently picking one.

## Setup

Requires **Python ≥3.10** (the MCP SDK's floor; built and tested on 3.14).

```bash
python3.14 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env      # then set GEMINI_API_KEY=...
```

Only the answering agents call an LLM. Fetching, ingestion, retrieval and the
whole test suite need no key.

> **Quota matters more than you'd expect.** One question costs up to 4 LLM calls
> (route + two specialists + synthesis). Free-tier quotas are **per model**, and
> `gemini-flash-latest` resolves to a model with a 20-requests/day preview quota —
> hence the `gemini-flash-lite-latest` default. Completions are cached to disk, so
> re-running an eval is free. See `.env.example`.

## Run

```bash
# one-time: download filings and build the local index
python src/fetch.py                    # latest 10-K + 3 recent 10-Qs per company
python src/ingest.py --all --reset

# ask it
python src/web.py                      # browser UI at http://127.0.0.1:8000
python src/orchestrator.py "Is NVDA expensive right now?" -v

# a single specialist, or raw retrieval
python src/agent.py "What risks does Apple name?" --ticker AAPL
python src/query.py "data center revenue" --ticker NVDA --temporal

# tests (fast, no network, no quota)
python -m pytest tests/ -q
```

Set `TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` when
running anything that loads the embedding model, or it will reach for Hugging
Face and stall.

Re-label periods after changing fiscal parsing, without re-downloading:
`python src/fetch.py --relabel && python src/ingest.py --all --reset`.

## Measured

Every change is scored against a fixed eval rather than eyeballed.

| | |
|---|---|
| Retrieval (101-question set, 17 categories) | recall@1 **76%**, @5 **97%**, MRR **0.853** |
| Table coverage on numeric questions | **24/24** (was 1/11 before tables were given a reserved slot) |
| Period coverage on cross-quarter questions | fan-out **100%** (vs 89% global top-k) |
| Company coverage on comparative questions | fan-out **100%** (vs 75% global top-k) |
| Routing (29 labelled questions) | **29/29** — but the router prompt was tuned against this set, so treat it as fitted, not held out |
| Reconciliation on "both" questions | **6/7** cite a filings *and* a market source |
| Context isolation | specialist reading grew **12.7×**; orchestrator context grew **1.04×** |

```bash
python eval/run_eval.py              # retrieval recall@k, MRR, table coverage
python eval/run_eval.py --method rerank   # dense / bm25 / hybrid / rerank
python eval/run_temporal.py          # period coverage
python eval/run_company_coverage.py  # company coverage on comparative questions
python eval/run_routing.py           # routing accuracy (--reconcile for the rest)
python eval/context_isolation.py     # orchestrator context stays bounded
```

Dense is the default because it wins on the eval: rerank scores MRR 0.761,
hybrid 0.731 and bm25 0.545 against dense's 0.853. Rerank is nonetheless much
better on a couple of narrow categories, so it stays available behind a flag.

## Layout

```
src/fetch.py       EDGAR download + fiscal-period metadata
src/parse.py       HTML -> narrative stream + extracted tables
src/sections.py    Item-boundary detection + section-aware chunking
src/ingest.py      parse -> chunk -> embed -> store
src/retrieval.py   dense / BM25 / hybrid / rerank + per-period fan-out
src/tools.py       search_filings() — the filings tool boundary
src/marketdata.py  quote / valuation / history / technicals
src/contracts.py   AgentAnswer + Citation — the sub-agent hand-off schema
src/fundamentals.py  src/technicals.py   the two specialists
src/orchestrator.py  router + parallel fan-out + reconciling synthesis
src/trace.py       per-run structured traces
src/web.py         local web UI (streams progress over SSE)
src/*_mcp_server.py  MCP servers over each data source
tests/             pytest suite
eval/              eval sets and their runners
docs/index.html    the "how it works" page
```

`data/` (raw filings, vector store, caches, traces) is gitignored and rebuilt by
`fetch.py` + `ingest.py`.

## Not

Financial advice. It declines buy/sell questions and price predictions by design,
and answers only from the two sources above.
