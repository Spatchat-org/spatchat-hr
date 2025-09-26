import os, time, random, sys
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from together import Together
from together.error import RateLimitError, ServiceUnavailableError

load_dotenv()

def _choice_content(choice):
    msg = getattr(choice, "message", None)
    if msg is None and isinstance(choice, dict):
        msg = choice.get("message")
    content = None
    if msg is not None:
        if isinstance(msg, dict):
            content = msg.get("content")
        else:
            content = getattr(msg, "content", None)
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(part.get("text", ""))
            elif isinstance(part, str):
                parts.append(part)
        content = "".join(parts)
    return content or ""

def _delta_text(delta):
    if isinstance(delta, dict):
        return delta.get("content", "")
    return getattr(delta, "content", "")

HF_MODEL_DEFAULT = "meta-llama/Meta-Llama-3.1-8B-Instruct"
TOGETHER_MODEL_DEFAULT = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"

class _SpacedCallLimiter:
    def __init__(self, min_interval_seconds: float):
        self.min_interval = float(min_interval_seconds)
        self._last = 0.0
        import threading
        self._lock = threading.Lock()
    def wait(self):
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
            self._last = time.monotonic()

class UnifiedLLM:
    def __init__(self):
        hf_model_or_url = (os.getenv("HF_ENDPOINT_URL") or HF_MODEL_DEFAULT).strip()
        hf_token = (os.getenv("HF_TOKEN") or "").strip()
        self.hf_client = InferenceClient(model=hf_model_or_url, token=hf_token, timeout=300)

        self.together = None
        self.together_model = (os.getenv("TOGETHER_MODEL") or TOGETHER_MODEL_DEFAULT).strip()
        tg_key = (os.getenv("TOGETHER_API_KEY") or "").strip()
        if tg_key:
            self.together = Together(api_key=tg_key)
            self._tg_limiter = _SpacedCallLimiter(min_interval_seconds=100.0)

    def _hf_chat(self, messages, max_tokens=512, temperature=0.0, stream=False):
        tries, delay, last_err = 3, 2.5, None
        for _ in range(tries):
            try:
                if hasattr(self.hf_client, "chat_completion"):
                    resp = self.hf_client.chat_completion(messages=messages, max_tokens=max_tokens, temperature=temperature, stream=stream)
                    if stream:
                        return "".join(_delta_text(ch.choices[0].delta) for ch in resp)
                    return _choice_content(resp.choices[0])
                else:
                    prompt = self._messages_to_prompt(messages)
                    return self.hf_client.text_generation(prompt, max_new_tokens=max_tokens, temperature=temperature, stream=False, return_full_text=False)
            except Exception as e:
                last_err = e
                time.sleep(delay); delay *= 1.8
        raise last_err

    @staticmethod
    def _messages_to_prompt(messages):
        parts = []
        for m in messages:
            role = m.get("role", "user"); content = m.get("content", "")
            tag = {"system":"system","user":"user"}.get(role, "assistant")
            parts.append(f"<|{tag}|>\n{content}\n")
        parts.append("<|assistant|>\n"); return "".join(parts)

    def chat(self, messages, temperature=0.0, max_tokens=512, stream=False):
        try:
            return self._hf_chat(messages, max_tokens=max_tokens, temperature=temperature, stream=stream)
        except Exception as hf_err:
            print(f"[LLM] HF primary failed: {hf_err}", file=sys.stderr)
            if self.together is None: raise
            self._tg_limiter.wait()
            backoff = 12.0
            for attempt in range(4):
                try:
                    resp = self.together.chat.completions.create(
                        model=self.together_model, messages=messages,
                        temperature=temperature, max_tokens=max_tokens, stream=stream
                    )
                    return _choice_content(resp.choices[0])
                except (RateLimitError, ServiceUnavailableError):
                    if attempt == 3: raise
                    time.sleep(backoff + random.uniform(0, 3)); backoff *= 1.8

SYSTEM_PROMPT = """
You are SpatChat, an expert wildlife home range analysis assistant.
If the user asks for a home range calculation (MCP, KDE, dBBMM, AKDE, etc.), reply ONLY in JSON using this format:
{"tool": "home_range", "method": "mcp", "levels": [95, 50]}
- method: one of "mcp", "kde", "akde", "bbmm", "dbbmm"
- levels: list of percentages for the home range (default [95] if user doesn't specify)
- Optionally, include animal_id if the user specifies a particular animal.
For any other questions, answer as an expert movement ecologist in plain text (keep to 2-3 sentences).
""".strip()

FALLBACK_PROMPT = """
You are SpatChat, a wildlife movement expert.
If you can't map a request to a home range tool, just answer naturally.
Keep replies under three sentences.
""".strip()

llm = UnifiedLLM()

def ask_llm(chat_history, user_input):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + chat_history + [{"role": "user", "content": user_input}]
    resp = llm.chat(messages, temperature=0.0, max_tokens=256, stream=False)
    import json
    try:
        call = json.loads(resp)
        return call, resp
    except Exception:
        conv = llm.chat([{"role": "system", "content": FALLBACK_PROMPT}] + messages, temperature=0.7, max_tokens=256, stream=False)
        return None, conv
