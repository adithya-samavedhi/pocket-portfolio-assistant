# pocket-portfolio-assistant

A research assistant that answers questions about a tracked set of companies from
their SEC filings and live market data, and cites every claim.

**[How it works →](https://adithya-samavedhi.github.io/pocket-portfolio-assistant/)**

```
$ python src/orchestrator.py "Is NVDA expensive right now?"

routed to: fundamentals, technicals

Market data shows NVDA trading at $223.96, near its 52-week high of $235.47 [1][3].
Multiples are elevated — P/S 21.4, trailing P/E 33.58 [2] — though the forward P/E
compresses to 17.37 [2], a tension between today's valuation and expected earnings
growth. The filings show growth supporting those multiples: revenue and net income
each rose 65%, to $215.9bn and $120.1bn, at a 71.1% gross margin [4].

  [1] NVDA market-data (yfinance) as of 2026-08-07 — technicals
  [4] NVDA 10-K FY2026 — Item 7: Management's Discussion and Analysis
```

Tracks **AAPL, AMZN, GOOGL, MSFT, NVDA** by default; add more with one command.
Everything runs locally except the LLM call — embeddings, retrieval and the vector
store need no API key.

---

## How it works

A **router** decides which specialists a question needs, they run **in parallel**,
and their structured summaries are **reconciled** into one cited answer.

```
question
   │
   ├─ route ─────► which specialists? which company? across quarters?
   │               is this analytical (needs a whole section) or a lookup?
   │
   ├─ fan out (parallel)
   │     ├─ Fundamentals ──► SEC filings   (Chroma, local embeddings)
   │     └─ Technicals   ──► market data   (yfinance / Finnhub)
   │
   └─ fan in ────► reconcile the two views, keep every citation
```

The orchestrator **never sees a filing passage or a price series** — only each
specialist's `AgentAnswer` summary. Its context is therefore bounded by the number
of specialists, not by how much they read. That is measured, not assumed: sub-agent
reading grew **12.7×** while orchestrator context grew **1.04×**.

| Specialist | Source | Answers |
|---|---|---|
| **Fundamentals** | SEC 10-K / 10-Q | revenue, segments, margins, risk factors, MD&A, change across quarters |
| **Technicals** | market data | price, P/E and P/S, momentum, 52-week position, drawdown, volatility |

Each data source also sits behind its own MCP server (`src/*_mcp_server.py`).

### Two design rules that carry most of the quality

**Deterministic work belongs in code, not the prompt.** Every technical measure is
computed in Python over the full price series; the model interprets numbers and is
told never to do arithmetic.

**Retrieval guarantees representation.** No ranking function beat plain dense
retrieval on the eval — but top-k silently starves whatever dimension a question
spans, so capacity is *reserved* for it:

| guarantee | metric | before → after |
|---|---|---|
| per-period fan-out | period coverage | 89% → **100%** |
| reserved table slots | table coverage | 1/11 → **24/24** |
| per-company fan-out | company coverage | 75% → **100%** |
| whole-section expansion | relevant text shown | **8.3×** |

Each guarantee is *additive* — it never displaces the normal results. Displacing
was tried twice and measurably lost information both times.

### Periods use the company's own labels

Read from each filing's inline XBRL (`dei:DocumentFiscalPeriodFocus`): NVIDIA's
quarter ending 2025-07-27 is **`Q2 FY2026`**, not calendar `Q3 2025`. Calendar
labels were wrong for 14 of 20 filings and contradicted the passage text they were
attached to; fixing it raised recall@1 from 68% to 76%.

Each chunk also carries `period_end` (the calendar anchor, and the join key to
price data), `calendar_quarter`, and `period_type`. A query for `"Q2 2026"` matches
**both** readings rather than silently guessing, and annual filings are excluded
from quarter-over-quarter fan-out — a fiscal year contains its own quarters.

---

## Setup

Requires **Python ≥3.10** (the MCP SDK's floor; built and tested on 3.14).

```bash
python3.14 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env      # then set GEMINI_API_KEY=... and SEC_USER_AGENT=...
```

Set these whenever you run something that loads the embedding model, or it will
reach for Hugging Face and stall:

```bash
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
```

### Build the corpus

```bash
python src/fetch.py                 # latest 10-K + 3 recent 10-Qs per company
python src/ingest.py --all          # parse, chunk, embed, store
```

First ingest downloads the embedding model and takes a few minutes for 20 filings.

---

## Using it

```bash
python src/web.py                                   # browser UI, streams progress
python src/orchestrator.py "Is NVDA expensive right now?" -v

python src/agent.py "What risks does Apple name?" --ticker AAPL   # one specialist
python src/query.py "data center revenue" --ticker NVDA --temporal  # raw retrieval
```

`src/web.py` has no auto-reload — **restart it after changing anything in `src/`**,
or you will be testing the old code.

### Keeping the corpus current

Tracking companies over years is a corpus-management problem, so there's a loop
for it. Nothing is re-downloaded or re-embedded unnecessarily.

```bash
python src/refresh.py --status              # how stale is it?
python src/refresh.py                       # fetch + index only what's new
python src/refresh.py --add TSLA META       # track more companies
python src/refresh.py --quarters 20         # deepen to ~5 years of history
```

After re-labelling periods or changing chunking, force a rebuild — skipping is
keyed on presence, not content:

```bash
python src/fetch.py --relabel && python src/ingest.py --all --reset
```

### Scale

Measured, extrapolated from the real corpus:

| | 5 companies × 1yr | 50 companies × 5yr |
|---|---|---|
| Filings | 20 | 1,000 |
| Chunks | 5,946 | ~297,000 |
| Disk | ~115 MB | ~1–2 GB |
| **RAM** | **~600 MB** | **~600 MB** (flat — the corpus never enters the process) |
| First ingest | minutes | ~2.5 h, once |

---

## Models

Provider-agnostic through LiteLLM — switching is one line in `.env`:

```bash
LLM_MODEL=gemini/gemini-3.5-flash          # free tier
LLM_MODEL=anthropic/claude-sonnet-4-5      # strongest
LLM_MODEL=ollama/llama3.1                  # local, no key
```

> **Quota shapes this project more than you'd expect.** One question costs up to
> **4 LLM calls** (route + two specialists + synthesis), and analytical questions
> send ~10k tokens each. Free quotas are **per model**, so switching gives a fresh
> allowance — and `gemini-flash-latest` resolves to a model with a 20-requests/day
> preview quota, which is why it isn't the default. Completions are cached to disk,
> so re-running an eval is free. `llm.py` retries per-minute limits and fails fast
> on daily ones. See `.env.example`.

Changing the **embedding** model is not cheap the same way: it needs a full
re-ingest, and `ingest.py` refuses to mix vectors from two models rather than
silently degrading retrieval.

---

## Measured

Nothing ships on vibes. 101 questions across 17 categories — factual, section,
cross-quarter, segment, trend, moat, headwind, growth driver, multi-hop,
comparative, quantitative, risk-linkage, accounting, capital allocation,
concentration, legal, and negatives that are deliberately unanswerable.

| | |
|---|---|
| Retrieval | recall@1 **76%**, @5 **97%**, MRR **0.853** |
| Table coverage on numeric questions | **24/24** |
| Period coverage, cross-quarter | **100%** (vs 89% global top-k) |
| Company coverage, comparative | **100%** (vs 75%) |
| Evidence shown, analytical | **8.3×** more than chunks alone |
| Routing accuracy | 29/29 — *but tuned on that set, so treat as fitted* |
| Reconciliation on "both" questions | **6/7** cite filings *and* market |
| Orchestrator context isolation | reading **12.7×** ↑, context **1.04×** ↑ |
| Tests | **50**, no network or quota needed |

```bash
python -m pytest tests/ -q           # fast, offline, free

python eval/run_eval.py              # recall@k, MRR, table coverage
python eval/run_eval.py --method rerank        # dense / bm25 / hybrid / rerank
python eval/run_temporal.py          # period coverage
python eval/run_company_coverage.py  # company coverage
python eval/run_section_coverage.py  # evidence volume on analytical questions
python eval/run_routing.py           # routing accuracy (--reconcile for the rest)
python eval/context_isolation.py     # orchestrator context stays bounded
```

Dense is the default because it wins: rerank 0.761, hybrid 0.731, bm25 0.545
against dense's 0.853. Rerank is much better on a couple of narrow categories, so
it stays available behind a flag.

### Known limits

- **Answer correctness is not tested.** Everything above measures *retrieval*.
  No test confirms a figure in a final answer matches the filing. This is the
  largest gap in the project.
- The **routing eval is overfit** — the prompt was tuned against the same set.
- Weakest retrieval categories: `concentration` (R@1 0%), `risk_linkage`,
  `growth_driver`.
- Coverage is **filings only** — no earnings-call transcripts, and no historical
  valuation multiples, so "expensive vs. its own history" is answered on price
  and trend alone (and the agent says so rather than implying otherwise).

---

## Layout

```
src/
  config.py        paths, embedding model, chunking, tracked companies
  fetch.py         EDGAR download + fiscal-period metadata from inline XBRL
  parse.py         HTML -> narrative stream + extracted tables
  sections.py      Item-boundary detection + section-aware chunking
  ingest.py        parse -> chunk -> embed -> store (incremental)
  refresh.py       fetch new filings, index what's new, report staleness
  retrieval.py     dense / bm25 / hybrid / rerank, fan-outs, section expansion
  tools.py         search_filings() — the filings tool boundary
  marketdata.py    quote / valuation / history / technicals
  contracts.py     AgentAnswer + Citation — the sub-agent hand-off schema
  llm.py           provider-agnostic LLM + retry, failover, disk cache
  fundamentals.py  technicals.py      the two specialists
  orchestrator.py  router + parallel fan-out + reconciling synthesis
  trace.py         per-run structured traces -> data/traces/
  web.py           local web UI (streams progress over SSE)
  *_mcp_server.py  MCP servers over each data source
tests/             pytest suite
eval/              eval sets and their runners
docs/index.html    the published "how it works" page
```

`data/` (raw filings, vector store, caches, traces) is gitignored and rebuilt by
`fetch.py` + `ingest.py`.

Every run writes a JSON trace to `data/traces/` — which specialists were consulted
and why, what each was *asked*, and how large its summary was. That is how the
failures in this project were diagnosed.

---

## Not

Financial advice. It declines buy/sell questions and price predictions by design,
answers only from the two sources above, and says so when the evidence isn't there.
