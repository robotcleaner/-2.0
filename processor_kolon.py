import io
import json
import pandas as pd
from datetime import datetime
from pathlib import Path
from config_kolon import (
    COLUMN_MAP_KOLON, OUTPUT_COLUMNS_KOLON,
    FIXED_REGION_KOLON, FIXED_AGENT_KOLON,
    PROCESS_RESULT_FIXED_KOLON,
    KR_PROMPT_TRANSLATE_KOLON,
    KR_PROMPT_FAULT_FROM_TEXT_KOLON,
    KR_PROMPT_PROCESS_RESULT_KOLON,
)
from config import PRODUCT_BU_MAP
from config_kr import KR_TO_CN_FAULT
from llm_client import _call_llm, MODEL_TRANSLATE, MODEL_CLASSIFY


# ==================== 工具 ====================
def _is_blank(s) -> bool:
    if s is None:
        return True
    t = str(s).strip()
    if not t:
        return True
    if t.lower() in ("nan", "none", "null", "nat"):
        return True
    return False


# ==================== 自动探测表头行 ====================
def find_header_row(file_bytes: bytes, max_check: int = 10) -> int:
    """
    自动检测 KOLON 表头在第几行（找含 '접수번호' 的那一行）。
    返回 skiprows 的值（即表头之前的行数）。
    """
    try:
        df = pd.read_excel(io.BytesIO(file_bytes), engine="openpyxl",
                           header=None, nrows=max_check)
    except Exception:
        return 0

    for i in range(len(df)):
        row_vals = [str(v) for v in df.iloc[i].tolist()]
        if any("접수번호" in v for v in row_vals):
            return i
    return 0


# ==================== 列名归一化 ====================
def _normalize_col_name(col: str) -> str:
    s = str(col)
    s = s.replace("\ufeff", "")
    s = s.strip().strip("*").strip()
    s = s.replace("\u3000", "").strip()
    s = s.replace("\n", "").replace("\r", "").replace("\t", "").strip()
    return s


def _normalize_df_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    new_cols = {c: _normalize_col_name(c) for c in df.columns}
    df.rename(columns=new_cols, inplace=True)
    return df


# ==================== 本地缓存 ====================
KOLON_CACHE_FILE = "fault_mapping_kolon.json"
_kolon_cache = {}
if Path(KOLON_CACHE_FILE).exists():
    try:
        _kolon_cache = json.loads(Path(KOLON_CACHE_FILE).read_text(encoding="utf-8"))
    except Exception:
        _kolon_cache = {}


def _save_kolon_cache():
    Path(KOLON_CACHE_FILE).write_text(
        json.dumps(_kolon_cache, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# ==================== 韩译中 ====================
def kr_translate(text: str) -> str:
    if _is_blank(text):
        return ""
    key = f"__TR__{str(text).strip()}"
    if key in _kolon_cache:
        return _kolon_cache[key]
    prompt = KR_PROMPT_TRANSLATE_KOLON.format(text=str(text).strip())
    result = _call_llm(prompt, MODEL_TRANSLATE)
    if not result or len(result) > 3000:
        return ""
    if "无法翻译" in result or "请提供" in result:
        return ""
    _kolon_cache[key] = result
    _save_kolon_cache()
    return result


# ==================== 故障类型 ====================
def translate_fault_class(text: str) -> str:
    if _is_blank(text):
        return ""
    t = str(text).strip()
    if t in KR_TO_CN_FAULT:
        return KR_TO_CN_FAULT[t]
    key = f"__FC3__{t}"
    if key in _kolon_cache:
        return _kolon_cache[key]

    prompt = (
        "你是家电维修故障分类专家。将下面这段韩文故障描述翻译成一个简短的"
        "中文故障类别（4-8个字），只输出这个中文短语，不要标点，不要解释。\n\n"
        f"韩文：{t}"
    )
    result = _call_llm(prompt, MODEL_CLASSIFY)
    if not result:
        return ""
    result = result.replace("【", "").replace("】", "").strip()[:12]
    if "未知" in result:
        return ""
    _kolon_cache[key] = result
    _save_kolon_cache()
    return result


def fault_from_text(text: str) -> str:
    if _is_blank(text):
        return ""
    key = f"__FT__{str(text).strip()}"
    if key in _kolon_cache:
        return _kolon_cache[key]

    prompt = KR_PROMPT_FAULT_FROM_TEXT_KOLON.format(text=str(text)[:2000])
    result = _call_llm(prompt, MODEL_CLASSIFY)
    if not result:
        return ""
    result = result.replace("【", "").replace("】", "").strip()[:12]
    if "未知" in result:
        return ""
    _kolon_cache[key] = result
    _save_kolon_cache()
    return result


# ==================== 处理方式 ====================
def translate_process_result(complaint_kr: str, repair_kr: str) -> str:
    if _is_blank(complaint_kr) and _is_blank(repair_kr):
        return ""
    key = f"__PR__{complaint_kr}||{repair_kr}"
    if key in _kolon_cache:
        return _kolon_cache[key]

    prompt = KR_PROMPT_PROCESS_RESULT_KOLON.format(
        complaint=str(complaint_kr or "")[:1000],
        repair=str(repair_kr or "")[:1500],
    )
    result = _call_llm(prompt, MODEL_CLASSIFY)
    if not result:
        return ""
    for fixed in PROCESS_RESULT_FIXED_KOLON:
        if fixed in result:
            _kolon_cache[key] = fixed
            _save_kolon_cache()
            return fixed
    return ""


# ==================== 产品BU ====================
def get_product_bu(device_name: str) -> str:
    if _is_blank(device_name):
        return ""
    name = str(device_name).upper()
    for key in sorted(PRODUCT_BU_MAP.keys(), key=len, reverse=True):
        if key.upper() in name:
            return PRODUCT_BU_MAP[key]
    return ""


# ==================== 日期 ====================
def parse_date_kr(val):
    if pd.isna(val) or val is None or str(val).strip() == "":
        return None
    s = str(val).strip()
    # 去掉时间部分（如 "2026-09-22 10:38:34"）
    if " " in s:
        s = s.split(" ")[0]
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y%m%d"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def normalize_date_kr(val) -> str:
    d = parse_date_kr(val)
    return d.strftime("%Y/%m/%d") if d else ""


# ==================== 类别 ====================
def determine_category_kolon(warranty: str,
                             complaint_kr: str,
                             repair_kr: str,
                             repair_count: int = 1) -> str:
    w = str(warranty or "").strip()

    # ① 保外（유상）
    if "유상" in w:
        return "维修（保外）"

    # ② DOA
    text_all = f"{complaint_kr} {repair_kr}".upper()
    if "DOA" in text_all:
        return "DOA"

    # ③ NFF
    text_orig = f"{complaint_kr} {repair_kr}"
    for kw in ("NFF", "NO FAULT FOUND", "未发现", "未复现", "无故障",
               "미발견", "미재현", "이상 없음", "이상없음", "재현안됨"):
        if kw in text_all or kw in text_orig:
            return "无异常（NFF）"

    # ④ 二返 / 多返
    if repair_count >= 3:
        return "多返"
    if repair_count == 2:
        return "二返"

    # ⑤ 默认保内
    return "维修（保内）"


# ==================== 单行处理 ====================
def process_row_kolon(row: pd.Series) -> dict:
    out = {}

    for src, dst in COLUMN_MAP_KOLON.items():
        out[dst] = row.get(src, "")

    out["工单号"] = out.get("工单号", "")
    out["区域"] = FIXED_REGION_KOLON
    out["代理"] = FIXED_AGENT_KOLON
    out["型号"] = out.get("型号", "")
    out["产品BU"] = get_product_bu(out.get("型号", ""))
    out["日期"] = normalize_date_kr(out.get("处理日", ""))
    out["SN"] = out.get("SN", "")

    fault_text = str(out.get("用户描述韩文", "") or "").strip()
    repair_text = str(out.get("维修详情韩文", "") or "").strip()
    warranty = str(out.get("保内保外", "") or "").strip()

    # H 故障类型（一级）：① 对照表 → ② 维修详情 → ③ 用户描述
    out["故障类型（一级）"] = ""

    if not _is_blank(fault_text):
        out["故障类型（一级）"] = translate_fault_class(fault_text)

    if _is_blank(out["故障类型（一级）"]) and not _is_blank(repair_text):
        out["故障类型（一级）"] = fault_from_text(repair_text)

    if _is_blank(out["故障类型（一级）"]) and not _is_blank(fault_text):
        out["故障类型（一级）"] = fault_from_text(fault_text)

    # I 处理方式
    out["处理方式"] = translate_process_result(fault_text, repair_text)

    # J 类别
    out["类别"] = determine_category_kolon(
        warranty=warranty,
        complaint_kr=fault_text,
        repair_kr=repair_text,
        repair_count=1,
    )

    # K 故障描述
    out["故障描述"] = kr_translate(fault_text) if not _is_blank(fault_text) else ""

    # L 维修详情
    out["维修详情"] = kr_translate(repair_text) if not _is_blank(repair_text) else ""

    return out


# ==================== 批量处理 ====================
def process_dataframe_kolon(df, progress_cb=None,
                            on_row_done=None, stop_flag_getter=None):
    # 列名归一化
    df = _normalize_df_columns(df)

    rows = []
    total = len(df)
    for i, (_, row) in enumerate(df.iterrows()):
        if stop_flag_getter and stop_flag_getter():
            break
        try:
            out = process_row_kolon(row)
            rows.append(out)
            if on_row_done:
                on_row_done(out)
        except Exception:
            empty = {col: "" for col in OUTPUT_COLUMNS_KOLON}
            empty["工单号"] = row.get("접수번호", "ERROR")
            rows.append(empty)
            if on_row_done:
                on_row_done(empty)
        if progress_cb:
            progress_cb(i + 1, total)

    result = pd.DataFrame(rows)
    for col in OUTPUT_COLUMNS_KOLON:
        if col not in result.columns:
            result[col] = ""
    return result[OUTPUT_COLUMNS_KOLON]