import hashlib
import json
from pathlib import Path
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from config import (
    SILICONFLOW_API_KEY, SILICONFLOW_BASE_URL,
    MODEL_TRANSLATE, MODEL_CLASSIFY,
    TEMPERATURE, MAX_TOKENS, TIMEOUT, MAX_RETRY,
)

client = OpenAI(
    api_key=SILICONFLOW_API_KEY,
    base_url=SILICONFLOW_BASE_URL,
    timeout=TIMEOUT,
)

CACHE_FILE = Path(".cache.json")
_cache = {}
if CACHE_FILE.exists():
    try:
        _cache = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
    except Exception:
        _cache = {}


def _cache_key(text: str, task: str) -> str:
    return hashlib.md5(f"{task}::{text}".encode("utf-8")).hexdigest()


def _save_cache():
    CACHE_FILE.write_text(
        json.dumps(_cache, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _is_empty(text) -> bool:
    if text is None:
        return True
    s = str(text).strip()
    if not s:
        return True
    if s.lower() in ("nan", "none", "null", "nat"):
        return True
    return False


@retry(stop=stop_after_attempt(MAX_RETRY),
       wait=wait_exponential(multiplier=1, min=2, max=10))
def _call_llm(prompt: str, model: str) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )
    return resp.choices[0].message.content.strip()


def translate(text: str, task: str = "translate") -> str:
    if _is_empty(text):
        return ""
    key = _cache_key(text, task)
    if key in _cache:
        return _cache[key]

    from prompts import PROMPTS
    prompt = PROMPTS[task].format(text=text)
    result = _call_llm(prompt, MODEL_TRANSLATE)

    if not result or len(result) > 2000:
        return ""
    if "无法翻译" in result or "请提供" in result:
        return ""

    _cache[key] = result
    _save_cache()
    return result


def classify_fault_from_sku(sku_text: str) -> str:
    """从备件 SKU 推断故障类型（LLM 兜底，不写缓存，由 processor 写）"""
    if _is_empty(sku_text):
        return ""
    key = _cache_key(sku_text, "fault_from_sku")
    if key in _cache:
        return _cache[key]

    from prompts import PROMPTS
    prompt = PROMPTS["fault_from_sku"].format(text=str(sku_text)[:2000])
    result = _call_llm(prompt, MODEL_CLASSIFY)

    if not result or len(result) > 30:
        return ""
    # 只保留前 12 个字符，避免模型话多
    result = result.replace("【", "").replace("】", "").strip()[:12]
    if "未知" in result or "无法" in result:
        return ""

    _cache[key] = result
    _save_cache()
    return result


def classify_credit_claim(text: str) -> str:
    """判断 credit_claim 是 退货 / 更换整机 / 更换备件/配件"""
    if _is_empty(text):
        return "更换备件/配件"
    key = _cache_key(text, "credit_claim")
    if key in _cache:
        return _cache[key]

    from prompts import PROMPTS
    prompt = PROMPTS["credit_claim"].format(text=str(text)[:2000])
    result = _call_llm(prompt, MODEL_CLASSIFY)
    result = result.replace("【", "").replace("】", "").strip()

    if "退货" in result:
        result = "退货"
    elif "更换整机" in result:
        result = "更换整机"
    else:
        result = "更换备件/配件"

    _cache[key] = result
    _save_cache()
    return result