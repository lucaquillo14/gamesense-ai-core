# backend/ai/local_coach.py
# ------------------------------------------------------------
# GameSense Local AI Coach — offline-friendly, with rolling memory
# ------------------------------------------------------------
from __future__ import annotations
import os, json, time, glob, math
from typing import List, Dict, Any, Optional

# Optional tiny LM (local)
_HAVE_LM = False
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    _HAVE_LM = True
except Exception:
    _HAVE_LM = False

MODELS_DIR = os.environ.get("GS_MODELS_DIR", "models")
MEMO_FILE  = os.environ.get("GS_COACH_MEMO", "data/coach_memory.jsonl")

def _ensure_dir(p: str):
    os.makedirs(os.path.dirname(p), exist_ok=True)

def _cosine(a: Dict[str,int], b: Dict[str,int]) -> float:
    # tiny bag-of-words cosine
    if not a or not b: return 0.0
    keys = set(a.keys()) | set(b.keys())
    da = sum((a.get(k,0))**2 for k in keys) ** 0.5
    db = sum((b.get(k,0))**2 for k in keys) ** 0.5
    if da == 0 or db == 0: return 0.0
    dot = sum(a.get(k,0)*b.get(k,0) for k in keys)
    return float(dot / (da*db))

def _bow(text: str) -> Dict[str,int]:
    out: Dict[str,int] = {}
    for tok in text.lower().split():
        if tok.isalpha() and len(tok) > 2:
            out[tok] = out.get(tok, 0) + 1
    return out

class LocalGameSenseCoach:
    def __init__(self, model_dir: Optional[str] = None):
        self.model_dir = model_dir or os.path.join(MODELS_DIR, "gpt2")
        self.lm_ready = False
        self.tokenizer = None
        self.model = None

        if _HAVE_LM and os.path.isdir(self.model_dir):
            try:
                print("🧠 Loading local LM from", self.model_dir)
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, local_files_only=True)
                self.model = AutoModelForCausalLM.from_pretrained(self.model_dir, local_files_only=True)
                self.model.eval()
                self.lm_ready = True
            except Exception as e:
                print("⚠️ LM load failed, using rule-based coach:", e)

        _ensure_dir(MEMO_FILE)

    # ---------- Memory ----------
    def add_to_memory(self, summary: str, meta: Dict[str,Any]):
        _ensure_dir(MEMO_FILE)
        rec = {
            "ts": int(time.time()),
            "summary": summary,
            "meta": meta,
            "bow": _bow(summary),
        }
        with open(MEMO_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    def _read_memory(self) -> List[Dict[str,Any]]:
        if not os.path.exists(MEMO_FILE):
            return []
        out=[]
        with open(MEMO_FILE, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    out.append(json.loads(line))
                except Exception:
                    continue
        return out[-200:]  # last 200 only

    def _nearest_memories(self, query: str, k: int = 3) -> List[Dict[str,Any]]:
        bowq = _bow(query)
        mem = self._read_memory()
        scored = []
        for r in mem:
            sim = _cosine(bowq, r.get("bow", {}))
            scored.append((sim, r))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [r for _, r in scored[:k]]

    # ---------- Context Loader ----------
    def load_latest_context(self, sessions_dir: str) -> Dict[str,Any]:
        js = glob.glob(os.path.join(sessions_dir, "*.json"))
        if not js:
            return {}
        latest = max(js, key=os.path.getmtime)
        try:
            with open(latest, "r") as f:
                data = json.load(f)
            return {
                "feedback": data.get("feedback", []),
                "tactical_summary": data.get("tactical_summary", {}),
                "sprint": data.get("sprint"),
                "shot": data.get("shot"),
            }
        except Exception:
            return {}

    # ---------- Answering ----------
    def generate(self, prompt: str, max_new_tokens: int = 220) -> str:
        if not self.lm_ready:
            # fallback template
            return prompt[:1200]
        try:
            import torch
            inputs = self.tokenizer(prompt, return_tensors="pt")
            with torch.no_grad():
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            return self.tokenizer.decode(out[0], skip_special_tokens=True)
        except Exception as e:
            return f"[Coach fallback] {prompt[:1000]}\n\n(Note: LM error: {e})"

    def answer(self, user_msg: str, sessions_dir: str) -> str:
        ctx = self.load_latest_context(sessions_dir)
        memories = self._nearest_memories(user_msg, k=3)

        bullets = []
        if ctx.get("feedback"):
            bullets.extend(ctx["feedback"])
        sp = ctx.get("sprint") or {}
        if sp:
            bullets.append(f"Sprint: max {sp.get('max_speed_kmh','-')} km/h, avg {sp.get('mean_speed_kmh','-')} km/h, distance {sp.get('total_distance_m','-')} m.")
        sh = ctx.get("shot") or {}
        if sh:
            bullets.append(f"Shot: peak {sh.get('max_speed_kmh','-')} km/h. {sh.get('feedback','')}")

        mem_lines = []
        for m in memories:
            mem_lines.append(f"- {time.strftime('%d %b', time.localtime(m['ts']))}: {m['summary']}")

        prompt = (
            "You are GameSense Coach. Give concise, actionable football coaching feedback.\n\n"
            f"User message: {user_msg}\n\n"
            f"Latest clip context:\n- " + "\n- ".join(bullets[:6]) + "\n\n"
            "Past memory (most relevant):\n" + ("\n".join(mem_lines) if mem_lines else "None") + "\n\n"
            "Advice (short bullets, concrete, elite-level phrasing):"
        )
        raw = self.generate(prompt)
        # if LM not present, we returned the prompt; trim to the answer part
        if raw.startswith("You are GameSense Coach"):
            return "\n".join([
                "• Keep scanning before receiving; pre-orient to play forward in 1–2 touches.",
                "• Create a passing lane by stepping wide or dropping 2m; demand the ball on back foot.",
                "• Increase sprint intent in transition—first 5m explosive, then reach 90% quickly.",
            ])
        return raw

# Singleton accessor
_COACH_SINGLETON: Optional[LocalGameSenseCoach] = None

def get_coach_ai() -> LocalGameSenseCoach:
    global _COACH_SINGLETON
    if _COACH_SINGLETON is None:
        _COACH_SINGLETON = LocalGameSenseCoach()
    return _COACH_SINGLETON