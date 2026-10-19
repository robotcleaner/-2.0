import json
import pandas as pd
from datetime import datetime
from pathlib import Path
from config_kr import (
    COLUMN_MAP_KR, OUTPUT_COLUMNS_KR,
    FIXED_REGION_KR, FIXED_AGENT_KR,
    PRODUCT_CATEGORY_KR,
    WARRANTY_TRUE_KR, WARRANTY_FALSE_KR,
    SKU_COLUMNS_KR,
    KR_TO_CN_FAULT,
    PROCESS_RESULT_KR_MAP, PROCESS_RESULT_FIXED,
    KR_PROMPT_TRANSLATE,
    KR_PROMPT_FAULT_FROM_TEXT,
    KR_PROMPT_PROCESS_RESULT,
)
from llm_client import _call_llm, MODEL_TRANSLATE, MODEL_CLASSIFY


# ==================== 工具：空值 ====================
def _is_blank(s) -> bool:
    if s is None:
        return True
    t = str(s).strip()
    if not t:
        return True
    if t.lower() in ("nan", "none", "null", "nat"):
        return True
    return False


# ==================== 列名归一化 ====================
def _normalize_col_name(col: str) -> str:
    s = str(col).strip()
    s = s.lstrip("*").strip()
    s = s.rstrip("*").strip()
    return s


def _normalize_df_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    new_cols = {c: _normalize_col_name(c) for c in df.columns}
    df.rename(columns=new_cols, inplace=True)
    return df


# ==================== 本地缓存 ====================
KR_FAULT_FILE = "fault_mapping_kr.json"
_kr_cache = {}
if Path(KR_FAULT_FILE).exists():
    try:
        _kr_cache = json.loads(Path(KR_FAULT_FILE).read_text(encoding="utf-8"))
    except Exception:
        _kr_cache = {}


def _save_kr_cache():
    Path(KR_FAULT_FILE).write_text(
        json.dumps(_kr_cache, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# ==================== 通用韩译中 ====================
def kr_translate(text: str) -> str:
    if _is_blank(text):
        return ""
    key = f"__TR__{str(text).strip()}"
    if key in _kr_cache:
        return _kr_cache[key]
    prompt = KR_PROMPT_TRANSLATE.format(text=str(text).strip())
    result = _call_llm(prompt, MODEL_TRANSLATE)
    if not result or len(result) > 3000:
        return ""
    if "无法翻译" in result or "请提供" in result:
        return ""
    _kr_cache[key] = result
    _save_kr_cache()
    return result


# ==================== 产品品类韩译中 ====================
def translate_product_category(kr_text: str) -> str:
    if _is_blank(kr_text):
        return ""
    t = str(kr_text).strip()
    if t in PRODUCT_CATEGORY_KR:
        return PRODUCT_CATEGORY_KR[t]
    return kr_translate(t)


# ==================== 处理方式（8 个固定值） ====================
def translate_process_result(kr_text: str) -> str:
    if _is_blank(kr_text):
        return ""
    t = str(kr_text).strip()
    if t in PROCESS_RESULT_KR_MAP:
        return PROCESS_RESULT_KR_MAP[t]

    key = f"__PR__{t}"
    if key in _kr_cache:
        return _kr_cache[key]

    prompt = KR_PROMPT_PROCESS_RESULT.format(text=t)
    result = _call_llm(prompt, MODEL_CLASSIFY)
    if not result:
        return ""
    for fixed in PROCESS_RESULT_FIXED:
        if fixed in result:
            _kr_cache[key] = fixed
            _save_kr_cache()
            return fixed
    return ""


# ==================== 故障类型（一级） ====================
def translate_fault_class3(kr_text: str) -> str:
    if _is_blank(kr_text):
        return ""
    t = str(kr_text).strip()
    if t in KR_TO_CN_FAULT:
        return KR_TO_CN_FAULT[t]
    key = f"__FC3__{t}"
    if key in _kr_cache:
        return _kr_cache[key]

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
    _kr_cache[key] = result
    _save_kr_cache()
    return result


def fault_from_text(text: str) -> str:
    if _is_blank(text):
        return ""
    key = f"__FT__{str(text).strip()}"
    if key in _kr_cache:
        return _kr_cache[key]

    prompt = KR_PROMPT_FAULT_FROM_TEXT.format(text=str(text)[:2000])
    result = _call_llm(prompt, MODEL_CLASSIFY)
    if not result:
        return ""
    result = result.replace("【", "").replace("】", "").strip()[:12]
    if "未知" in result:
        return ""
    _kr_cache[key] = result
    _save_kr_cache()
    return result


def fault_from_kr_sku(sku_texts) -> str:
    if not sku_texts:
        return ""
    combined = " ".join([str(x) for x in sku_texts if not _is_blank(x)])
    if not combined:
        return ""
    key = f"__SKU__{combined}"
    if key in _kr_cache:
        return _kr_cache[key]

    prompt = (
        "你是家电维修配件专家。下面是韩国维修工单消耗的备件名称。"
        "请根据备件名称，推断对应的中文故障类型（4-8个字），"
        "只输出中文短语，不要标点、不要解释。\n\n"
        f"备件：\n{combined[:2000]}"
    )
    result = _call_llm(prompt, MODEL_CLASSIFY)
    if not result:
        return ""
    result = result.replace("【", "").replace("】", "").strip()[:12]
    if "未知" in result:
        return ""
    _kr_cache[key] = result
    _save_kr_cache()
    return result


# ==================== 日期 ====================
def parse_date_kr(val):
    if pd.isna(val) or val is None or str(val).strip() == "":
        return None
    s = str(val).strip()
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y%m%d"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def normalize_date_kr(val) -> str:
    d = parse_date_kr(val)
    return d.strftime("%Y/%m/%d") if d else ""


# ==================== 类别（韩国） ====================
# NFF 触发词
KR_NFF_TEXT_KEYWORDS = [
    # 直译
    "未发现", "未复现", "无故障",
    "미발견", "미재현", "이상 없음", "이상없음",
    # 业务规则新增
    "直排水安装", "直排解除", "配件购买", "安装移机申请",
    "직배수 설치", "직배수 해제", "부품 구매", "설치 이전 신청",
]


def _has_kr_nff_signal(repair_details_kr: str,
                       fault3_kr: str = "",
                       complaint_kr: str = "",
                       process_result_kr: str = "") -> bool:
    """
    判断是否 NFF。
      ① NFF / NO FAULT FOUND → NFF
      ② 未发现 / 未复现 / 无故障 → NFF
      ③ 直排水安装 / 直排解除 / 配件购买 / 安装移机申请 → NFF
      ④ 「无异常」：
           - 同时出现「更换」→ 不算 NFF
           - 否则 → NFF
    """
    text_all = " ".join([
        str(repair_details_kr or ""),
        str(fault3_kr or ""),
        str(complaint_kr or ""),
        str(process_result_kr or ""),
    ])
    text_upper = text_all.upper()

    # ① 明确 NFF
    if "NFF" in text_upper or "NO FAULT FOUND" in text_upper:
        return True

    # ② 关键词命中
    for kw in KR_NFF_TEXT_KEYWORDS:
        if kw in text_all:
            return True

    # ③ 「无异常」特判
    if "无异常" in text_all or "이상 없음" in text_all or "이상없음" in text_all:
        if "更换" in text_all or "교체" in text_all:
            return False
        return True

    return False


def determine_category_kr(process_result_kr: str,
                          warranty: str,
                          repair_count,
                          repair_details_kr: str,
                          fault3_kr: str = "",
                          complaint_kr: str = "") -> str:
    # ① DOA
    if "DOA" in str(process_result_kr).upper():
        return "DOA"

    # ② 保外
    w = str(warranty or "").strip()
    if w in WARRANTY_FALSE_KR:
        return "维修（保外）"

    # ③ NFF
    if _has_kr_nff_signal(
        repair_details_kr=repair_details_kr,
        fault3_kr=fault3_kr,
        complaint_kr=complaint_kr,
        process_result_kr=process_result_kr,
    ):
        return "无异常（NFF）"

    # ④ 二返 / 多返
    try:
        cnt = int(repair_count)
    except Exception:
        cnt = 1
    if cnt >= 3:
        return "多返"
    if cnt == 2:
        return "二返"

    # ⑤ 默认
    return "维修（保内）"


# ==================== 单行处理 ====================
def process_row_kr(row: pd.Series) -> dict:
    out = {}

    for src, dst in COLUMN_MAP_KR.items():
        out[dst] = row.get(src, "")

    # A 工单号
    out["工单号"] = out.get("工单号", "")
    # B 区域
    out["区域"] = FIXED_REGION_KR
    # C 代理
    out["代理"] = FIXED_AGENT_KR
    # D 型号
    out["型号"] = out.get("型号", "")
    # E 产品BU
    out["产品BU"] = translate_product_category(out.get("产品品类韩文", ""))
    # F 日期
    out["日期"] = normalize_date_kr(out.get("最终处理日", ""))
    # G SN
    out["SN"] = out.get("SN", "")

    # H 故障类型（一级）
    fault3_kr = str(out.get("报修分类3韩文", "") or "").strip()
    tech_kr = str(out.get("维修详情韩文", "") or "").strip()
    complaint_kr = str(out.get("用户描述韩文", "") or "").strip()

    if not _is_blank(fault3_kr):
        out["故障类型（一级）"] = translate_fault_class3(fault3_kr)
    else:
        out["故障类型（一级）"] = ""

    # 兜底：维修详情 → 备件 → 用户描述
    if _is_blank(out["故障类型（一级）"]):
        fault = fault_from_text(tech_kr)
        if not fault:
            skus = [row.get(c, "") for c in SKU_COLUMNS_KR]
            fault = fault_from_kr_sku(skus)
        if not fault:
            fault = fault_from_text(complaint_kr)
        out["故障类型（一级）"] = fault or ""

    # I 处理方式
    out["处理方式"] = translate_process_result(out.get("处理结果3韩文", ""))

    # J 类别
    out["类别"] = determine_category_kr(
        process_result_kr=out.get("处理结果3韩文", ""),
        warranty=out.get("Warranty", ""),
        repair_count=out.get("报修次数", 1),
        repair_details_kr=tech_kr,
        fault3_kr=fault3_kr,
        complaint_kr=complaint_kr,
    )

    # K 故障描述
    out["故障描述"] = kr_translate(complaint_kr) if not _is_blank(complaint_kr) else ""

    # L 维修详情
    out["维修详情"] = kr_translate(tech_kr) if not _is_blank(tech_kr) else ""

    return out


# ==================== 批量处理 ====================
def process_dataframe_kr(df, progress_cb=None, on_row_done=None, stop_flag_getter=None):
    # 列名归一化
    df = _normalize_df_columns(df)

    rows = []
    total = len(df)
    for i, (_, row) in enumerate(df.iterrows()):
        if stop_flag_getter and stop_flag_getter():
            break
        try:
            out = process_row_kr(row)
            rows.append(out)
            if on_row_done:
                on_row_done(out)
        except Exception:
            empty = {col: "" for col in OUTPUT_COLUMNS_KR}
            empty["工单号"] = row.get("접수번호", "ERROR")
            rows.append(empty)
            if on_row_done:
                on_row_done(empty)
        if progress_cb:
            progress_cb(i + 1, total)

    result = pd.DataFrame(rows)
    for col in OUTPUT_COLUMNS_KR:
        if col not in result.columns:
            result[col] = ""
    return result[OUTPUT_COLUMNS_KR]