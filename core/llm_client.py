import json
import os
import random
import time
from typing import Optional, Dict, Any, Tuple, Union

import requests

# --- Shared API Settings ---
# Load/update the API key in this file once and the change will propagate
# anywhere the helper is imported.
API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
if not API_KEY:
    raise RuntimeError("OPENROUTER_API_KEY environment variable is required")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
MAX_RETRIES = 5
INITIAL_RETRY_DELAY = 5

# Claude models routed through the native Anthropic API instead of OpenRouter
# to access the full output_config.effort system (incl. "max"). OpenRouter
# currently flattens effort on 4.7's adaptive thinking — the direct path
# gives real effort differentiation and access to the thinking content block.
ANTHROPIC_NATIVE_MODELS = {
    "anthropic/claude-opus-4.7":   "claude-opus-4-7",
    "anthropic/claude-opus-4.6":   "claude-opus-4-6",
    "anthropic/claude-sonnet-4.6": "claude-sonnet-4-6",
}


class APIError(Exception):
    """Custom exception for API-related errors."""
    pass


def _call_anthropic_direct(
    prompt: str,
    model: str,
    effort: str,
    system_prompt: Optional[str],
    include_usage: bool,
) -> Union[str, Tuple[str, Dict[str, Any]]]:
    """Call Claude via the native Anthropic /v1/messages endpoint (streaming).

    Uses output_config.effort to control adaptive thinking depth. Streaming
    is mandatory here — at effort="max" on hard prompts the model can think
    for >10min, which blows past any sane non-streaming read timeout.
    Anthropic's docs require streaming for any request that may exceed 10min.
    """
    anth_model = ANTHROPIC_NATIVE_MODELS[model]
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
        "accept": "text/event-stream",
    }
    payload: Dict[str, Any] = {
        "model": anth_model,
        # 16k is plenty of headroom for a benchmark response + its thinking
        # trace. 64k burned ~$5/call at Opus pricing on runaway thinking —
        # and retrying didn't help because the model deterministically
        # thinks to the cap on hard prompts.
        "max_tokens": 16000,
        "messages": [{"role": "user", "content": prompt}],
        "output_config": {"effort": effort},
        # Must be explicit — without this 4.7 won't emit thinking blocks even
        # at max effort for open-ended prompts (the model chooses to inline
        # deliberation into visible text instead).
        "thinking": {"type": "adaptive"},
        "stream": True,
    }
    if system_prompt:
        payload["system"] = system_prompt

    delay = INITIAL_RETRY_DELAY
    last_exc: Optional[Exception] = None
    for attempt in range(MAX_RETRIES):
        try:
            time.sleep(random.uniform(0.02, 0.12))
            # Read timeout is per-event. Streaming resets it on every SSE
            # event so thinking can run indefinitely as long as the server
            # keeps emitting deltas (it emits thinking_deltas / pings often).
            r = requests.post(
                url, json=payload, headers=headers,
                timeout=(10, 300), stream=True,
            )
            if not r.ok:
                try:
                    err = r.json().get("error", r.text[:500])
                except ValueError:
                    err = (r.text or "")[:500]
                last_exc = APIError(f"Anthropic HTTP {r.status_code}: {err}")
                r.close()
                if r.status_code == 429:
                    retry_after = r.headers.get("Retry-After")
                    if retry_after:
                        try:
                            delay = max(delay, float(retry_after))
                        except ValueError:
                            pass
                if 400 <= r.status_code < 500 and r.status_code not in (408, 409, 429):
                    raise last_exc
            else:
                # SSE parse: accumulate text/thinking deltas per content block,
                # capture usage from message_start and message_delta.
                blocks: Dict[int, Dict[str, Any]] = {}
                input_tok = 0
                output_tok = 0
                cache_read = 0
                cache_write = 0
                stream_error: Optional[str] = None
                stop_reason: Optional[str] = None

                try:
                    for raw_line in r.iter_lines(decode_unicode=True):
                        if not raw_line:
                            continue
                        if not raw_line.startswith("data:"):
                            continue
                        data_str = raw_line[5:].strip()
                        if not data_str:
                            continue
                        try:
                            evt = json.loads(data_str)
                        except (json.JSONDecodeError, ValueError):
                            continue
                        etype = evt.get("type")
                        if etype == "message_start":
                            u = (evt.get("message") or {}).get("usage") or {}
                            input_tok = u.get("input_tokens", 0) or 0
                            output_tok = u.get("output_tokens", 0) or 0
                            cache_read = u.get("cache_read_input_tokens", 0) or 0
                            cache_write = u.get("cache_creation_input_tokens", 0) or 0
                        elif etype == "content_block_start":
                            idx = evt.get("index", 0)
                            cb = evt.get("content_block") or {}
                            blocks[idx] = {
                                "type": cb.get("type"),
                                "text": "",
                                "thinking": cb.get("thinking", "") or "",
                            }
                        elif etype == "content_block_delta":
                            idx = evt.get("index", 0)
                            delta = evt.get("delta") or {}
                            dtype = delta.get("type")
                            if idx not in blocks:
                                blocks[idx] = {"type": None, "text": "", "thinking": ""}
                            if dtype == "text_delta":
                                blocks[idx]["text"] += delta.get("text", "") or ""
                                if blocks[idx].get("type") is None:
                                    blocks[idx]["type"] = "text"
                            elif dtype == "thinking_delta":
                                blocks[idx]["thinking"] += delta.get("thinking", "") or ""
                                if blocks[idx].get("type") is None:
                                    blocks[idx]["type"] = "thinking"
                            # signature_delta is 4.7's encrypted thinking — ignore content, just note presence
                            elif dtype == "signature_delta":
                                if blocks[idx].get("type") is None:
                                    blocks[idx]["type"] = "thinking"
                        elif etype == "message_delta":
                            u = evt.get("usage") or {}
                            if "output_tokens" in u:
                                output_tok = u.get("output_tokens", output_tok) or output_tok
                            d = evt.get("delta") or {}
                            if d.get("stop_reason"):
                                stop_reason = d.get("stop_reason")
                        elif etype == "error":
                            err_obj = evt.get("error") or {}
                            stream_error = f"{err_obj.get('type', 'error')}: {err_obj.get('message', data_str[:300])}"
                            break
                        elif etype == "message_stop":
                            break
                finally:
                    r.close()

                if stream_error:
                    last_exc = APIError(f"Anthropic stream error: {stream_error}")
                else:
                    text_parts = []
                    thinking_block_present = False
                    thinking_visible = []
                    for idx in sorted(blocks.keys()):
                        blk = blocks[idx]
                        if blk.get("type") == "text":
                            text_parts.append(blk.get("text", ""))
                        elif blk.get("type") == "thinking":
                            thinking_block_present = True
                            tvis = blk.get("thinking", "")
                            if tvis:
                                thinking_visible.append(tvis)
                    text = "".join(text_parts).strip()

                    # Max-tokens truncation with no visible text: the model
                    # thought to the cap and never wrote anything. Retrying
                    # at the same effort is deterministic waste ($5/call at
                    # Opus pricing). Raise immediately so caller-level
                    # fallbacks (e.g. collect.py's medium-effort retry) can
                    # kick in with a lower-cost config.
                    if stop_reason == "max_tokens" and not text:
                        raise APIError(
                            f"Anthropic max_tokens ({output_tok} output tokens) "
                            f"with no visible text — not retrying at same effort"
                        )

                    if not text:
                        last_exc = APIError("Anthropic returned empty text content")
                    else:
                        if not include_usage:
                            return text
                        # Estimate thinking-token share: total output minus visible
                        # text (~4 chars/token). If no thinking block present, zero.
                        if thinking_block_present:
                            visible_tok = max(1, len(text) // 4)
                            est_reasoning = max(0, output_tok - visible_tok)
                        else:
                            est_reasoning = 0
                        usage = {
                            "prompt_tokens": input_tok,
                            "completion_tokens": output_tok,
                            "reasoning_tokens": est_reasoning,
                            "total_tokens": input_tok + output_tok,
                            "cost": None,
                            "cost_details": None,
                            "prompt_tokens_details": {
                                "cached_tokens": cache_read,
                                "cache_write_tokens": cache_write,
                            },
                            "thinking_content": "\n".join(thinking_visible),
                            "thinking_present": thinking_block_present,
                            "raw": {
                                "input_tokens": input_tok,
                                "output_tokens": output_tok,
                                "cache_read_input_tokens": cache_read,
                                "cache_creation_input_tokens": cache_write,
                            },
                        }
                        return text, usage
        except requests.RequestException as e:
            last_exc = e

        if attempt < MAX_RETRIES - 1:
            print(f"Retrying after error: {last_exc}. Waiting {delay:.2f}s...")
            time.sleep(delay + random.uniform(0, 0.5))
            delay *= 2

    raise last_exc or APIError("Anthropic: exhausted retries")


def _pick_anthropic_effort(
    name: str,
    reasoning: bool,
    reasoning_effort: Optional[str],
    tier_lists: Dict[str, list],
) -> str:
    """Map a model name + flags to an Anthropic effort level.

    Matches the tier system used by the OpenRouter branch so model
    routing stays consistent. Bare thinking models (the fallback "else"
    branch on OpenRouter) default to "max" per user request.
    """
    if reasoning_effort is not None:
        return reasoning_effort
    if not reasoning or name in tier_lists["nonthinking_variants"] or name in tier_lists["none_models"]:
        return "low"
    if name in tier_lists["minimal_models"]:
        return "low"
    if name in tier_lists["low_models"]:
        return "low"
    if name in tier_lists["medium_models"]:
        return "medium"
    if name in tier_lists["high_models"]:
        return "high"
    if name in tier_lists["xhigh_models"]:
        return "xhigh"
    return "max"


def get_llm_response(
    prompt: str,
    model: str,
    name: str,
    reasoning: bool,
    system_prompt: Optional[str] = None,
    include_usage: bool = False,
    reasoning_effort: Optional[str] = None,
) -> Union[str, Tuple[str, Dict[str, Any]]]:
    # print(model, prompt)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    provider_candidates = {
        # Try these in order; adjust to what you know works for your key
        "qwen/qwen3-30b-a3b": ["novita/fp8", "nebius/fp8", "deepinfra/fp8", 'chutes' ],
    }
    provider_overrides = {
        # "qwen/qwen3-32b": "cerebras",
        "qwen/qwen3-235b-a22b": "chutes/bf16",
        "qwen/qwen3-235b-a22b-thinking-2507": "chutes",
        "qwen/qwen3-coder": "cerebras/fp8",
        "qwen/qwen3-235b-a22b-2507": "targon/bf16",
        "meta-llama/llama-4-maverick": "baseten/fp16",
        "deepseek/deepseek-r1-0528": "targon/fp8",
        "deepseek/deepseek-chat-v3-0324": "targon/fp8",
        "deepseek/deepseek-r1": "targon/fp8",
        "deepseek/deepseek-chat": "targon/fp8",
        "meta-llama/llama-3.3-70b-instruct": "novita/bf16",
        "qwen/qwen3-32b": "novita/fp8",
        "openai/gpt-oss-120b": "novita",
        "openai/gpt-oss-20b": "novita",
        "google/gemma-3-27b-it": "deepinfra/bf16",
        "google/gemma-3-12b-it": "deepinfra/bf16",
        "moonshotai/kimi-k2": "chutes/fp8",
        "z-ai/glm-4.5": "chutes/fp8",
        "deepseek/deepseek-chat-v3.1": "fireworks",
        "qwen/qwen3-next-80b-a3b-thinking":"deepinfra/bf16",
        "qwen/qwen3-next-80b-a3b-instruct":"deepinfra/bf16",
        "deepseek/deepseek-v3.1-terminus":"novita",
        'z-ai/glm-4.6':'novita',
        'deepseek/deepseek-v3.2': 'deepseek',
        'deepseek/deepseek-v3.2-speciale': 'deepseek',
        'z-ai/glm-4.7': 'z-ai',
        'z-ai/glm-5': 'z-ai',
        'z-ai/glm-5.1': 'z-ai'
    }

    nonthinking_variants = [
        "Gemini 2.5 Flash Lite Preview (2025-06-17) Nonthinking",
        "Gemini 2.5 Flash Preview Nonthinking",
        "Claude 3.7 Sonnet",
        "Claude 4 Opus",
        "Claude 4 Sonnet",
        "Claude Opus 4.1",
        "DeepSeek V3.1 (Non-Reasoning)",

    ]
    # Models that reject explicit reasoning-disable payloads — just omit the param
    no_reasoning_param_models = [
        "Intellect-3", "Grok 3 Mini Beta",
    ]
    high_models = ["o3 High", "o3-Mini High", "o4-Mini High",
                   "Grok 3 Mini Beta (High)", "GPT-5 (high)",
                   "GPT-5 Mini (High)", "GPT-5 Nano (high)", 'GPT-5.2 (high)',
                   "GPT-5 Mini",
                   "GPT-5 Codex", "GPT-5.1 Codex", "GPT-5.1 Codex Mini",
                   "GPT-5.1 (high)"]
    xhigh_models = ['GPT-5.1 Codex Max (xhigh)', 'GPT-5.2 (xhigh)']
    medium_models = ['GPT-5.1 (medium)', "GPT-5 (medium)", "GPT-5 Mini (medium)", "GPT-5 Nano", "GPT-5 Nano (medium)", 'GPT-5.2 (medium)',
                     "o3-Mini Medium", "o4-Mini Medium", "o3 Medium"]
    low_models = ["GPT-5 (low)", "GPT-5 Mini (low)", "GPT-5 Nano (low)", 'Claude Opus 4.5 Thinking (Low)', 'Gemini 3.0 Pro Preview (2025-11-18) (Low)', 'Gemini 3.0 Flash Preview (2025-12-17) (Low)']
    minimal_models = ["GPT-5 (minimal)", "GPT-5 Nano (minimal)", "GPT-5 Mini (minimal)", 'Gemini 3.0 Flash Preview (2025-12-17) (Minimal)']
    none_models = ["GPT-5.1 (Non-reasoning)", "GPT-5.2 (Non-reasoning)"]

    # Route supported Claude models through native Anthropic API for full
    # effort control (OpenRouter flattens effort on 4.7's adaptive thinking).
    if ANTHROPIC_API_KEY and model in ANTHROPIC_NATIVE_MODELS:
        effort = _pick_anthropic_effort(name, reasoning, reasoning_effort, {
            "nonthinking_variants": nonthinking_variants,
            "none_models": none_models,
            "minimal_models": minimal_models,
            "low_models": low_models,
            "medium_models": medium_models,
            "high_models": high_models,
            "xhigh_models": xhigh_models,
        })
        return _call_anthropic_direct(
            prompt=prompt, model=model, effort=effort,
            system_prompt=system_prompt, include_usage=include_usage,
        )

    url = "https://openrouter.ai/api/v1/chat/completions"
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        # "reasoning": {"enabled": reasoning},
        "usage": {"include": include_usage},
        "stream": False,
    }

    if reasoning_effort is not None:
        # Explicit override from caller — skip all name-based logic
        payload["reasoning"] = {"effort": reasoning_effort}
    elif name in no_reasoning_param_models:
        pass  # Don't set reasoning param — these endpoints reject it
    elif name == "Gemini 2.5 Pro Preview (2025-06-05) Limited":
        payload["reasoning"] = {"max_tokens": 8000}
    elif name == "Gemini 3.0 Pro Preview (2025-11-18) 1k Limited":
        payload["reasoning"] = {"max_tokens": 1000}
    elif name in none_models:
        payload["reasoning"] = {"effort": "none"}
    elif name in nonthinking_variants or reasoning == False:
        payload["reasoning"] = {"max_tokens": 0, 'enabled' : False}
    elif name in medium_models:
        payload["reasoning"] = {"effort": "medium"}
    elif name in low_models:
        payload["reasoning"] = {"effort": "low"}
    elif name in minimal_models:
        payload["reasoning"] = {"effort": "minimal"}
    elif name in high_models:
        payload["reasoning"] = {"effort": "high"}
    elif name in xhigh_models:
        payload["reasoning"] = {"effort": "xhigh", "enabled": True}
    else:
        payload["reasoning"] = {"effort": "xhigh", "enabled": True}

    forced_provider = provider_overrides.get(model)
    delay = INITIAL_RETRY_DELAY

    def _summarize_usage(u: Dict[str, Any]) -> Dict[str, Any]:
        comp_details = (u or {}).get("completion_tokens_details") or {}
        prompt_details = (u or {}).get("prompt_tokens_details") or {}
        return {
            "prompt_tokens": u.get("prompt_tokens"),
            "completion_tokens": u.get("completion_tokens"),
            "reasoning_tokens": comp_details.get("reasoning_tokens", 0),
            "total_tokens": u.get("total_tokens"),
            "cost": u.get("cost"),
            "cost_details": u.get("cost_details"),
            "prompt_tokens_details": prompt_details,
            "raw": u,
        }



    with requests.Session() as s:
        s.trust_env = False  # ignore env proxies
        base_headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "openbench/1.0",
            "HTTP-Referer": "http://localhost",
            "X-Title": "Parallel-Benchmarker",
        }
        s.headers.update(base_headers)

        for attempt in range(MAX_RETRIES):
            try:
                time.sleep(random.uniform(0.02, 0.12))

                # Try three transport variants each attempt
                transport_variants = [
                    {"connection_close": False, "identity": False},  # normal keep-alive
                    {"connection_close": True,  "identity": False},  # force fresh TCP/TLS
                    {"connection_close": True,  "identity": True},   # fresh + no compression
                ]
                prov_list = provider_candidates.get(model)
                if prov_list:
                    modes = [("prov", p, False) for p in prov_list]           # force each provider
                    modes += [("prov", prov_list[0], True)]                    # allow fallbacks from the first
                elif forced_provider:  # <- keep your old single override behavior
                    modes = [("prov", forced_provider, False), ("prov", forced_provider, True)]
                else:
                    modes = []

                # Always add a router-chosen lane at the end
                modes += [("no-override", None, None)]

                last_exc = None
                rate_limited = False

                for tv in transport_variants:
                    if rate_limited:
                        break
                    # fresh session when we need to change transport knobs
                    sess = s
                    if tv["connection_close"] or tv["identity"]:
                        sess = requests.Session()
                        sess.trust_env = False
                        hdrs = dict(base_headers)
                        if tv["connection_close"]:
                            hdrs["Connection"] = "close"
                        if tv["identity"]:
                            hdrs["Accept-Encoding"] = "identity"
                        sess.headers.update(hdrs)

                    try:
                        for mode in modes:
                            req_payload = dict(payload)
                            kind, prov, allow_fb = mode  # mode is a tuple

                            if kind == "prov":
                                req_payload["provider"] = {"order": [prov], "allow_fallbacks": bool(allow_fb)}
                            # elif kind == "no-override": leave provider unset

                            try:
                                r = sess.post(url, json=req_payload, timeout=(10, 60))
                            except requests.RequestException as e:
                                last_exc = e
                                # rotate to next provider mode or next transport variant
                                continue

                            if not r.ok:
                                try:
                                    err_body = r.json()
                                    err_msg = err_body.get("error") or err_body
                                except ValueError:
                                    err_msg = (r.text or "")[:1000]
                                last_exc = APIError(f"HTTP {r.status_code} ({mode}): {err_msg}")
                                if r.status_code == 429:
                                    # Don't waste retries cycling variants — break to backoff sleep
                                    retry_after = r.headers.get("Retry-After")
                                    if retry_after:
                                        try:
                                            delay = max(delay, float(retry_after))
                                        except ValueError:
                                            pass
                                    rate_limited = True
                                    break
                                continue

                            raw = r.content or b""
                            if not raw.strip():
                                last_exc = APIError(
                                    f"Whitespace body ({mode}, close={tv['connection_close']}, identity={tv['identity']}); "
                                    f"CT={r.headers.get('Content-Type')}, CE={r.headers.get('Content-Encoding')}, "
                                    f"TE={r.headers.get('Transfer-Encoding')}, CL={r.headers.get('Content-Length')}"
                                )
                                continue

                            try:
                                data = r.json()
                            except (json.JSONDecodeError, ValueError):
                                snippet = (r.text or "")[:400]
                                last_exc = APIError(
                                    f"JSON decode failed ({mode}, close={tv['connection_close']}, identity={tv['identity']}, "
                                    f"CT={r.headers.get('Content-Type')}, CE={r.headers.get('Content-Encoding')}). "
                                    f"Body starts: {snippet!r}"
                                )
                                continue

                            msg = (data.get("choices", [{}])[0].get("message") or {})
                            content = msg.get("content")
                            if not content:
                                last_exc = APIError(f"Malformed response ({mode}): missing choices[0].message.content")
                                continue

                            # Detect reasoning-token leakage: model was asked
                            # to reason but returned 0 reasoning tokens, which
                            # means the thinking ended up in the content field.
                            reasoning_cfg = payload.get("reasoning", {})
                            reasoning_expected = (
                                reasoning_cfg
                                and reasoning_cfg.get("enabled") is not False
                                and reasoning_cfg.get("max_tokens") != 0
                                and reasoning_cfg.get("effort") not in ("none", "minimal")
                            )
                            if reasoning_expected:
                                raw_usage = data.get("usage", {})
                                comp_details = (raw_usage.get("completion_tokens_details") or {})
                                r_tokens = comp_details.get("reasoning_tokens", 0) or 0
                                if r_tokens == 0:
                                    last_exc = APIError(f"Reasoning leak: 0 reasoning tokens ({mode})")
                                    continue

                            if not include_usage:
                                return content.strip()

                            usage = _summarize_usage(data.get("usage", {}))
                            return content.strip(), usage

                    finally:
                        if sess is not s:
                            sess.close()

                # all variants+modes failed in this attempt
                if attempt == MAX_RETRIES - 1:
                    raise last_exc or APIError("Exhausted attempts")

                sleep_for = delay + random.uniform(0, 1)
                print(f"Retrying after error: {last_exc}. Waiting {sleep_for:.2f}s...")
                time.sleep(sleep_for)
                delay = min(delay * 2, 60)

            except requests.RequestException as e:
                # catch anything unexpected at the attempt level
                if attempt == MAX_RETRIES - 1:
                    raise APIError(f"Network error: {e}") from e
                print(f"Retrying after network error: {e}. Waiting {delay:.2f}s...")
                time.sleep(delay + random.uniform(0, 1))
                delay = min(delay * 2, 60)
