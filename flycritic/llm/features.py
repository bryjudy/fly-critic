"""Frozen-LLM feature extractor with caching.

Given observation texts, returns
  hidden: (B, d_model) last-layer hidden state at the final prompt token (chat template applied), and
  tool_logits: (B, 6) = log-prob of the FIRST token of each tool name as the assistant's first output token.
Method (documented, deterministic): we build the chat prompt with the tokenizer's chat template
(add_generation_prompt=True), run one forward pass, take logits at the last position, and read off the
log-softmax at the first token id of " WebSearch", "Calculator", ... (each tool name tokenised standalone; if a
name's first token collides with another's we fall back to the full-name sum of log-probs). Cached by sha1 of
the text in memory and, optionally, in an on-disk shelve.
"""
import hashlib, shelve, os, time, torch

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
MODELS = {"0.5b": "Qwen/Qwen2.5-0.5B-Instruct", "1.5b": "Qwen/Qwen2.5-1.5B-Instruct", "7b": "Qwen/Qwen2.5-7B-Instruct"}

class FrozenLLM:
    def __init__(self, model=DEFAULT_MODEL, device=None, dtype=None, cache_path=None, tools=None, chunk=16):
        self.chunk = chunk   # max prompts per forward pass (use 1-2 for long-context prompts on a 24 GB GPU)
        from transformers import AutoTokenizer, AutoModelForCausalLM
        model = MODELS.get(model, model)
        self.device = device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        dtype = dtype or (torch.bfloat16 if self.device == "cuda" else torch.float32)
        self.tok = AutoTokenizer.from_pretrained(model); self.tok.padding_side = "left"
        if self.tok.pad_token is None: self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model, torch_dtype=dtype).to(self.device).eval()
        for p in self.model.parameters(): p.requires_grad_(False)
        self.d_model = self.model.config.hidden_size
        from .toolworld import TOOLS
        self.tools = tools or TOOLS
        self._tool_ids = [self.tok.encode(t, add_special_tokens=False) for t in self.tools]
        firsts = [ids[0] for ids in self._tool_ids]
        self.first_unique = len(set(firsts)) == len(firsts)
        self.mem = {}; self.disk = shelve.open(cache_path) if cache_path else None
        self.calls = 0; self.call_time = 0.0

    def _prompt(self, text):
        msgs = [{"role": "user", "content": text}]
        return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    @staticmethod
    def _key(s): return hashlib.sha1(s.encode()).hexdigest()

    @torch.no_grad()
    def _forward(self, texts):
        prompts = [self._prompt(t) for t in texts]
        enc = self.tok(prompts, return_tensors="pt", padding=True).to(self.device)
        t0 = time.time()
        out = self.model(**enc, output_hidden_states=True)
        self.calls += 1; self.call_time += time.time() - t0
        hidden = out.hidden_states[-1][:, -1].float()                   # left-padded -> last position is final token
        logp = torch.log_softmax(out.logits[:, -1].float(), -1)
        if self.first_unique:
            tl = torch.stack([logp[:, ids[0]] for ids in self._tool_ids], 1)
        else:  # rare: sum log-probs over full name via teacher forcing
            tl = torch.stack([self._name_logprob(enc, ids) for ids in self._tool_ids], 1)
        return hidden.cpu(), tl.cpu()

    @torch.no_grad()
    def _name_logprob(self, enc, ids):
        B = enc["input_ids"].shape[0]; total = torch.zeros(B, device=self.device)
        inp = enc["input_ids"]; att = enc["attention_mask"]
        for tid in ids:
            out = self.model(input_ids=inp, attention_mask=att)
            total += torch.log_softmax(out.logits[:, -1].float(), -1)[:, tid]
            inp = torch.cat([inp, torch.full((B, 1), tid, device=self.device)], 1); att = torch.cat([att, torch.ones(B, 1, device=self.device, dtype=att.dtype)], 1)
        return total

    def __call__(self, texts):
        """texts: list[str] -> hidden (B, d_model) float32 cpu, tool_logits (B, n_tools) cpu."""
        keys = [self._key(t) for t in texts]
        miss = [i for i, k in enumerate(keys) if k not in self.mem and not (self.disk is not None and k in self.disk)]
        if miss:
            for c0 in range(0, len(miss), self.chunk):
                sub = miss[c0:c0 + self.chunk]
                h, tl = self._forward([texts[i] for i in sub])
                for j, i in enumerate(sub):
                    self.mem[keys[i]] = (h[j], tl[j])
                    if self.disk is not None: self.disk[keys[i]] = (h[j], tl[j])
        got = [self.mem[k] if k in self.mem else self.disk[k] for k in keys]
        return torch.stack([g[0] for g in got]), torch.stack([g[1] for g in got])
