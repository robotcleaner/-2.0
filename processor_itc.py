# processor_itc.py
import io
import json
import pandas as pd
from datetime import datetime
from pathlib import Path

from config_itc import (
    COLUMN_MAP_ITC, OUTPUT_COLUMNS_ITC,
    FIXED_REGION_ITC, FIXED_AGENT_ITC,
    PROCESS_RESULT_FIXED_ITC,
    JP_PROMPT_TRANSLATE_ITC,
    JP_PROMPT_FAULT_FROM_TEXT_ITC,
    JP_PROMPT_PROCESS_RESULT_ITC,
)
from config import PRODUCT_BU_MAP
from config_common import is_mova_ticket, is_old_model
from llm_client import _call_llm, MODEL_TRANSLATE, MODEL_CLASSIFY

try:
    from config_kr import KR_TO_CN_FAULT
except Exception:
    KR_TO_CN_FAULT = {}


# ==================== 工具 ====================
def _is_blank(s) -> bool:
    if s is None:
        return True
    t = str(s).strip()
    if not t:
        return True
    if t.lower() in ("nan", "none", "null", "nat", "/", "／", "-"):
        return True
    return False


# ==================== 自动探测表头行 ====================
def find_header_row(file_bytes: bytes, max_check: int = 10) -> int:
    """找含 'ITC受付番号' 的那一行，返回 skiprows 值。"""
    try:
        df = pd.read_excel(io.BytesIO(file_bytes), engine="openpyxl",
                           header=None, nrows=max_check)
    except Exception:
        return 0

    for i in range(len(df)):
        row_vals = [str(v) for v in df.iloc[i].tolist()]
        if any("ITC受付番号" in v for v in row_vals):
            return i
    for i in range(len(df)):
        row_vals = [str(v) for v in df.iloc[i].tolist()]
        if any(v.strip() == "RMA" for v in row_vals):
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


def _get_col(row, name):
    """按归一化名模糊匹配列（兼容 'xxx' 与 'xxx备注'）"""
    if name in row.index:
        return row.get(name, "")
    for c in row.index:
        if name and name in str(c):
            return row.get(c, "")
    return ""


# ==================== 本地缓存 ====================
ITC_CACHE_FILE = "fault_mapping_itc.json"
_itc_cache = {}
if Path(ITC_CACHE_FILE).exists():
    try:
        _itc_cache = json.loads(Path(ITC_CACHE_FILE).read_text(encoding="utf-8"))
    except Exception:
        _itc_cache = {}


def _save_itc_cache():
    Path(ITC_CACHE_FILE).write_text(
        json.dumps(_itc_cache, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# ==================== 日译中 ====================
def jp_translate(text: str) -> str:
    if _is_blank(text):
        return ""
    key = f"__TR__{str(text).strip()}"
    if key in _itc_cache:
        return _itc_cache[key]
    prompt = JP_PROMPT_TRANSLATE_ITC.format(text=str(text).strip()[:3000])
    result = _call_llm(prompt, MODEL_TRANSLATE)
    if not result or len(result) > 3000:
        return ""
    if "无法翻译" in result or "请提供" in result:
        return ""
    _itc_cache[key] = result
    _save_itc_cache()
    return result


# ==================== 故障类型 ====================
def translate_fault_class(text: str) -> str:
    if _is_blank(text):
        return ""
    t = str(text).strip()
    if t in KR_TO_CN_FAULT:
        return KR_TO_CN_FAULT[t]
    key = f"__FC3__{t}"
    if key in _itc_cache:
        return _itc_cache[key]

    prompt = (
        "你是家电维修故障分类专家。将下面这段日文故障描述翻译成一个简短的"
        "中文故障类别（4-8个字），只输出这个中文短语，不要标点，不要解释。\n\n"
        f"日文：{t}"
    )
    result = _call_llm(prompt, MODEL_CLASSIFY)
    if not result:
        return ""
    result = result.replace("【", "").replace("】", "").strip()[:12]
    if "未知" in result:
        return ""
    _itc_cache[key] = result
    _save_itc_cache()
    return result


def fault_from_text(text: str) -> str:
    if _is_blank(text):
        return ""
    key = f"__FT__{str(text).strip()}"
    if key in _itc_cache:
        return _itc_cache[key]

    prompt = JP_PROMPT_FAULT_FROM_TEXT_ITC.format(text=str(text)[:2000])
    result = _call_llm(prompt, MODEL_CLASSIFY)
    if not result:
        return ""
    result = result.replace("【", "").replace("】", "").strip()[:12]
    if "未知" in result:
        return ""
    _itc_cache[key] = result
    _save_itc_cache()
    return result


# ==================== 处理方式 ====================
def translate_process_result(complaint_jp: str, repair_jp: str) -> str:
    if _is_blank(complaint_jp) and _is_blank(repair_jp):
        return ""
    key = f"__PR__{complaint_jp}||{repair_jp}"
    if key in _itc_cache:
        return _itc_cache[key]

    prompt = JP_PROMPT_PROCESS_RESULT_ITC.format(
        complaint=str(complaint_jp or "")[:1000],
        repair=str(repair_jp or "")[:1500],
    )
    result = _call_llm(prompt, MODEL_CLASSIFY)
    if not result:
        return ""
    for fixed in PROCESS_RESULT_FIXED_ITC:
        if fixed in result:
            _itc_cache[key] = fixed
            _save_itc_cache()
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
def parse_date_jp(val):
    if pd.isna(val) or val is None or str(val).strip() == "":
        return None
    s = str(val).strip()
    if " " in s:
        s = s.split(" ")[0]
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y%m%d"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def normalize_date_jp(val) -> str:
    d = parse_date_jp(val)
    return d.strftime("%Y/%m/%d") if d else ""


# ==================== 类别 ====================
def determine_category_itc(warranty: str,
                           complaint_jp: str,
                           repair_jp: str,
                           repair_count: int = 1,
                           model: str = "") -> str:
    w = str(warranty or "").strip()

    # ① 保外（保証外）
    if "保証外" in w or "保外" in w:
        return "维修（保外）"

    # ② DOA（优先于老机型，DOA 不改）
    text_all = f"{complaint_jp} {repair_jp}".upper()
    if "DOA" in text_all:
        return "DOA"

    # ③ NFF
    text_orig = f"{complaint_jp} {repair_jp}"
    for kw in ("NFF", "NO FAULT FOUND", "未发现", "未复现", "无故障",
               "異常なし", "異常無し", "再現せず", "再現されず",
               "症状確認されず", "異常はありません"):
        if kw in text_all or kw in text_orig:
            return "无异常（NFF）"

    # ④ 老机型 → 强制保外
    if is_old_model(model):
        return "维修（保外）"

    # ⑤ 二返 / 多返
    if repair_count >= 3:
        return "多返"
    if repair_count == 2:
        return "二返"

    # ⑥ 默认保内
    return "维修（保内）"


# ==================== 工单状态 ====================
def determine_status_itc(arrival: str, progress: str, done_flag: str) -> str:
    done = str(done_flag or "").strip()
    if done in ("✔", "√", "Y", "y", "TRUE", "True", "true", "True "):
        return "已完成"
    prog = str(progress or "").strip()
    if "完了" in prog or "完成" in prog:
        return "已完成"
    arr = str(arrival or "").strip().upper()
    if arr == "Y":
        return "进行中"
    return "待处理"


# ==================== 单行处理 ====================
def process_row_itc(row: pd.Series) -> dict:
    out = {}
    for src, dst in COLUMN_MAP_ITC.items():
        out[dst] = _get_col(row, src)

    out["工单号"] = out.get("工单号", "")
    out["区域"] = FIXED_REGION_ITC
    out["代理"] = FIXED_AGENT_ITC
    out["型号"] = out.get("型号", "")
    out["产品BU"] = get_product_bu(out.get("型号", ""))
    out["日期"] = normalize_date_jp(out.get("处理日", ""))
    out["SN"] = out.get("SN", "")

    fault_text = str(out.get("用户描述日文", "") or "").strip()
    repair_text = str(out.get("维修详情日文", "") or "").strip()
    warranty = str(out.get("案例类别日文", "") or "").strip()
    note_text = str(out.get("备注", "") or "").strip()

    # 故障类型（一级）
    out["故障类型（一级）"] = ""
    if not _is_blank(repair_text):
        out["故障类型（一级）"] = fault_from_text(repair_text)
    if _is_blank(out["故障类型（一级）"]) and not _is_blank(fault_text):
        out["故障类型（一级）"] = translate_fault_class(fault_text)
    if _is_blank(out["故障类型（一级）"]) and not _is_blank(fault_text):
        out["故障类型（一级）"] = fault_from_text(fault_text)

    # 处理方式
    out["处理方式"] = translate_process_result(fault_text, repair_text)

    # 类别
    out["类别"] = determine_category_itc(
        warranty=warranty,
        complaint_jp=fault_text,
        repair_jp=repair_text,
        repair_count=1,
        model=out.get("型号", ""),
    )

    # 故障描述
    out["故障描述"] = jp_translate(fault_text) if not _is_blank(fault_text) else ""

    # 维修详情（维修详情 + 备注）
    detail_parts = []
    if not _is_blank(repair_text):
        detail_parts.append(jp_translate(repair_text))
    if not _is_blank(note_text):
        detail_parts.append(jp_translate(note_text))
    out["维修详情"] = " | ".join([p for p in detail_parts if p])

    # 物流单号（寄回单号 / 发货单号）
    send_back = str(out.get("寄回单号", "") or "").strip()
    send_out = str(out.get("发货单号", "") or "").strip()
    out["物流单号"] = " / ".join(
        [x for x in [send_back, send_out]
         if x and x.lower() not in ("nan", "/", "-", "／")]
    )

    # 到货状态
    arr = str(out.get("到货状态", "") or "").strip().upper()
    out["到货状态"] = "已到货" if arr == "Y" else "未到货"

    # 故障部件/料号
    part = str(out.get("故障部件", "") or "").strip()
    pno = str(out.get("料号", "") or "").strip()
    out["故障部件/料号"] = " / ".join(
        [x for x in [part, pno]
         if x and x.lower() not in ("nan", "/", "-", "／")]
    )

    # 工单状态
    out["工单状态"] = determine_status_itc(
        arr, out.get("处理结果日文", ""), out.get("完成标记", "")
    )

    # 唯一键
    rma_str = str(out.get("工单号", "") or "").strip()
    sn_str = str(out.get("SN", "") or "").strip()
    out["唯一键"] = f"{rma_str}__{sn_str}"

    return out


# ==================== 批量处理 ====================
def process_dataframe_itc(df, progress_cb=None,
                          on_row_done=None, stop_flag_getter=None):
    df = _normalize_df_columns(df)

    # ===== 剔除 MOVA 工单（按工单号前缀）=====
    if "RMA" in df.columns:
        before = len(df)
        df = df[~df["RMA"].astype(str).apply(is_mova_ticket)].copy()
        removed = before - len(df)
        if removed > 0:
            print(f"[ITC] 已剔除 MOVA 工单 {removed} 条")

    rows = []
    total = len(df)
    for i, (_, row) in enumerate(df.iterrows()):
        if stop_flag_getter and stop_flag_getter():
            break
        try:
            out = process_row_itc(row)
            rows.append(out)
            if on_row_done:
                on_row_done(out)
        except Exception:
            empty = {col: "" for col in OUTPUT_COLUMNS_ITC}
            empty["工单号"] = row.get("RMA", "ERROR")
            empty["唯一键"] = f"ERROR__{i}"
            rows.append(empty)
            if on_row_done:
                on_row_done(empty)
        if progress_cb:
            progress_cb(i + 1, total)

    result = pd.DataFrame(rows)
    for col in OUTPUT_COLUMNS_ITC:
        if col not in result.columns:
            result[col] = ""
    return result[OUTPUT_COLUMNS_ITC]