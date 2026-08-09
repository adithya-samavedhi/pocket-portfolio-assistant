"""Provider-agnostic LLM layer (Milestone 6).

Switching models is a one-line change: set the LLM_MODEL env var to any model
string LiteLLM understands, and provide that provider's API key. Examples:

    LLM_MODEL=gemini/gemini-flash-latest     GEMINI_API_KEY=...     (default, free tier)
    LLM_MODEL=anthropic/claude-sonnet-4-5    ANTHROPIC_API_KEY=...
    LLM_MODEL=openai/gpt-4o-mini             OPENAI_API_KEY=...
    LLM_MODEL=ollama/llama3.1                (local, no key)

The rest of the codebase depends only on LLM.complete(), never on a provider.
"""
import json
import os
import re
import sys
import time
from pathlib import Path

import litellm
from dotenv import load_dotenv

# Load keys/settings from the repo-root .env so this module works standalone too.
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# Flash-LITE, not Flash: `gemini-flash-latest` resolves to the newest Flash,
# which carries a preview quota of only 20 requests/DAY (measured). Flash-lite
# runs the documented free tier instead, which is orders of magnitude larger.
# The `-latest` alias avoids pinned versions Google deprecates on the free tier.
DEFAULT_MODEL = os.environ.get("LLM_MODEL", "gemini/gemini-flash-lite-latest")

# Which env var holds the key, by provider prefix — used only for a friendly
# preflight error; LiteLLM reads the keys itself.
_KEY_BY_PREFIX = {
    "gemini/": "GEMINI_API_KEY",
    "anthropic/": "ANTHROPIC_API_KEY",
    "openai/": "OPENAI_API_KEY",
    "groq/": "GROQ_API_KEY",
    "openrouter/": "OPENROUTER_API_KEY",
}


def missing_key(model: str):
    """Return the name of the missing API-key env var, or None if set/needed."""
    for prefix, env in _KEY_BY_PREFIX.items():
        if model.startswith(prefix):
            return None if os.environ.get(env) else env
    return None  # local providers (e.g. ollama) need no key


# Free tiers rate-limit aggressively (Gemini's is 5 requests/minute), and one
# orchestrator run costs up to 4. Without this, any multi-question eval dies
# partway through. Retries honor the provider's own suggested delay when it
# sends one, and fall back to exponential backoff when it doesn't.
MAX_RETRIES = int(os.environ.get("LLM_MAX_RETRIES", "5"))
MAX_BACKOFF = 65          # seconds; per-minute quotas reset inside this

# Optional comma-separated failover chain, e.g.
#   LLM_FALLBACK_MODELS=groq/llama-3.3-70b-versatile,openai/gpt-4o-mini
FALLBACK_MODELS = [m.strip() for m in
                   os.environ.get("LLM_FALLBACK_MODELS", "").split(",") if m.strip()]


# Disk cache for completions. At temperature 0 a repeated (model, system, user)
# is a pure function, so caching is semantically free — and it is what makes the
# eval sets re-runnable without spending quota. Disable with LLM_CACHE=0.
CACHE_ENABLED = os.environ.get("LLM_CACHE", "1") != "0"
LLM_CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache" / "llm"


def _cache_key(model: str, system: str, user: str, temperature: float) -> str:
    import hashlib
    blob = json.dumps([model, system, user, temperature], sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:32]


def _retry_after(err: Exception) -> float | None:
    """Seconds the provider asked us to wait, if it said so."""
    m = re.search(r"retry(?:Delay|.after)\D{0,4}(\d+(?:\.\d+)?)s?", str(err), re.I)
    return float(m.group(1)) if m else None


def _is_daily_quota(err: Exception) -> bool:
    """True when the exhausted quota is a per-DAY one.

    Worth distinguishing: a per-minute limit clears if you wait, but a daily
    limit does not, and the provider still advertises a ~60s retryDelay for it.
    Retrying a daily quota just burns minutes and more quota, so fail fast and
    let the caller fail over to another model (quotas are per-model) or stop.
    """
    return bool(re.search(r"PerDay|per.day|daily", str(err), re.I))


class LLM:
    def __init__(self, model: str = None, temperature: float = 0.0):
        self.model = model or DEFAULT_MODEL
        self.temperature = temperature

    def _call(self, model: str, system: str, user: str) -> str:
        resp = litellm.completion(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=self.temperature,
        )
        return resp.choices[0].message.content

    def complete(self, system: str, user: str) -> str:
        env = missing_key(self.model)
        if env:
            raise RuntimeError(
                f"{self.model} needs {env}. Set it, e.g.  export {env}=...  "
                f"(or choose another model with LLM_MODEL=...)."
            )

        # A cache hit costs nothing and keeps eval re-runs deterministic.
        key = _cache_key(self.model, system, user, self.temperature)
        cached = LLM_CACHE_DIR / f"{key}.json"
        if CACHE_ENABLED and cached.exists():
            return json.loads(cached.read_text())["response"]

        # Try the primary model, then any configured fallbacks in order.
        last_err = None
        for model in [self.model, *FALLBACK_MODELS]:
            if missing_key(model):
                continue                       # no key for this fallback; skip it
            for attempt in range(MAX_RETRIES):
                try:
                    out = self._call(model, system, user)
                    if CACHE_ENABLED:
                        LLM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
                        cached.write_text(json.dumps({"model": model, "response": out}))
                    return out
                except litellm.RateLimitError as e:
                    last_err = e
                    if _is_daily_quota(e):
                        print(f"[llm] {model}: DAILY quota exhausted — not retrying "
                              f"(it resets tomorrow). Set LLM_MODEL to another model "
                              f"(quotas are per-model) or configure "
                              f"LLM_FALLBACK_MODELS.", file=sys.stderr)
                        break                  # no amount of waiting fixes this
                    if attempt == MAX_RETRIES - 1:
                        break                  # exhausted here; try next model
                    wait = _retry_after(e) or min(2 ** attempt, MAX_BACKOFF)
                    wait = min(wait + 1, MAX_BACKOFF)   # +1s guards a tight reset
                    print(f"[llm] {model} rate-limited; retrying in {wait:.0f}s "
                          f"({attempt + 1}/{MAX_RETRIES})", file=sys.stderr)
                    time.sleep(wait)
                except litellm.APIError as e:
                    last_err = e               # transient server-side; move on
                    print(f"[llm] {model} error: {type(e).__name__}; "
                          f"trying next option", file=sys.stderr)
                    break
        raise last_err

    def complete_json(self, system: str, user: str) -> dict:
        """Like complete(), but parse the reply as a JSON object.

        Kept provider-agnostic: we ask for JSON in the prompt and parse
        tolerantly (strip code fences, or grab the first {...} block) rather
        than relying on a provider-specific structured-output mode.
        """
        return _extract_json(self.complete(system, user))


def _extract_json(text: str) -> dict:
    t = text.strip()
    if t.startswith("```"):                       # ```json ... ``` fences
        t = re.sub(r"^```(?:json)?", "", t).strip()
        t = re.sub(r"```$", "", t).strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", t, re.S)          # first {...} block
        if m:
            return json.loads(m.group(0))
        raise
