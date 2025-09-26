# llm_utils.py
import os, sys, time, random, json
from huggingface_hub import InferenceClient
from together import Together
from together.error import RateLimitError, ServiceUnavailableError

HF_MODEL_DEFAULT = "meta-llama/Meta-Llama-3.1-8B-Instruct"
TOGETHER_MODEL_DEFAULT = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"

def _choice_content(choice):
    msg = getattr(choice, "message", None) or (choice.get("message") if isinstance(choice, dict) else None)
    content = None if msg is None else (msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", None))
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(part.get("text",""))
            elif isinstance(part, str):
                parts.append(part)
        content = "".join(parts)
    return content or ""

def _delta_text(delta):
    return delta.get("content","") if isinstance(delta, dict) else getattr(delta, "content", "")

class _SpacedCallLimiter:
    def __init__(self, min_interval_seconds: float): self.min, self._last = float(min_interval_seconds), 0.0
    def wait(self):
        now = time.monotonic(); gap = now - self._last
        if gap < self.min: time.sleep(self.min - gap)
        self._last = time.monotonic()

class UnifiedLLM:
    def __init__(self):
        model_or_url = (os.getenv("HF_ENDPOINT_URL") or HF_MODEL_DEFAULT).strip()
        token = (os.getenv("HF_TOKEN") or "").strip()
        self.hf = InferenceClient(model=model_or_url, token=token, timeout=300)

        self.together = None
        self.together_model = (os.getenv("TOGETHER_MODEL") or TOGETHER_MODEL_DEFAULT).strip()
        tk = (os.getenv("TOGETHER_API_KEY") or "").strip()
        if tk:
            self.together = Together(api_key=tk)
            self._lim = _SpacedCallLimiter(100.0)  # ~0.6 QPM

    def _hf_chat(self, messages, max_tokens=256, temperature=0.0, stream=False):
        tries, delay, last = 3, 2.5, None
        for _ in range(tries):
            try:
                if hasattr(self.hf, "chat_completion"):
                    resp = self.hf.chat_completion(messages=messages, max_tokens=max_tokens, temperature=temperature, stream=stream)
                    return "".join(_delta_text(ch.choices[0].delta) for ch in resp) if stream else _choice_content(resp.choices[0])
                prompt = self._messages_to_prompt(messages)
                return self.hf.text_generation(prompt, max_new_tokens=max_tokens, temperature=temperature, stream=False, return_full_text=False)
            except Exception as e:
                last = e; time.sleep(delay); delay *= 1.8
        raise last

    @staticmethod
    def _messages_to_prompt(messages):
        parts=[]
        for m in messages:
            role=m.get("role","user"); content=m.get("content","")
            parts.append(f"<|{role}|>\n{content}\n")
        parts.append("<|assistant|>\n")
        return "".join(parts)

    def chat(self, messages, temperature=0.0, max_tokens=256, stream=False):
        try:
            return self._hf_chat(messages, max_tokens=max_tokens, temperature=temperature, stream=stream)
        except Exception as e:
            if not self.together: raise
            self._lim.wait()
            backoff=12.0
            for i in range(4):
                try:
                    resp = self.together.chat.completions.create(model=self.together_model, messages=messages, temperature=temperature, max_tokens=max_tokens, stream=False)
                    return _choice_content(resp.choices[0])
                except (RateLimitError, ServiceUnavailableError):
                    if i==3: raise
                    time.sleep(backoff + random.uniform(0,3)); backoff*=1.8

SYSTEM_PROMPT = """
You are SpatChat, an expert wildlife home range analysis assistant.
If the user asks for a home range calculation (MCP, KDE, dBBMM, AKDE, etc.), reply ONLY in JSON using:
{"tool":"home_range","method":"mcp","levels":[95,50]}
For other questions, answer concisely in plain text.
""".strip()

FALLBACK_PROMPT = """
You are SpatChat, a wildlife movement expert. If you can't map to a tool, answer naturally in <=3 sentences.
""".strip()

_llm = UnifiedLLM()

def ask_llm(chat_history, user_input):
    msgs = [{"role":"system","content":SYSTEM_PROMPT}] + chat_history + [{"role":"user","content":user_input}]
    resp = _llm.chat(msgs, temperature=0.0, max_tokens=256, stream=False)
    try:
        call = json.loads(resp)
        return call, resp
    except Exception:
        conv = _llm.chat([{"role":"system","content":FALLBACK_PROMPT}] + msgs, temperature=0.7, max_tokens=256, stream=False)
        return None, conv
