import csv
import json
import re
import pandas as pd
from datetime import datetime
from pathlib import Path
from config import (
    COLUMN_MAP, OUTPUT_COLUMNS,
    PROCESS_METHOD_MAP,
    PRODUCT_BU_MAP, FIXED_REGION, FIXED_AGENT,
    FAULT_MAPPING_FILE, DEFAULT_FAULT_MAPPING,
    SKU_TO_FAULT,
    WARRANTY_TABLE_FILE, DEFAULT_WARRANTY_YEARS,
    REFURB_WINDOW_DAYS,
)
from llm_client import translate, classify_credit_claim, classify_fault_from_sku


# ==================== 工具：空值判断 ====================
def _is_blank(s) -> bool:
    if s is None:
        return True
    t = str(s).strip()
    if not t:
        return True
    if t.lower() in ("nan", "none", "null", "nat"):
        return True
    return False


# ==================== 质保表 ====================
def load_warranty_table() -> dict:
    path = Path(WARRANTY_TABLE_FILE)
    table = {}
    if not path.exists():
        return table
    try:
        with path.open("r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                model = (row.get("Model") or "").strip()
                yr = (row.get("Warranty_Yr") or "").strip()
                if not model:
                    continue
                try:
                    table[model] = float(yr) if yr else DEFAULT_WARRANTY_YEARS
                except ValueError:
                    table[model] = DEFAULT_WARRANTY_YEARS
    except Exception:
        return {}
    return table


_warranty_table = load_warranty_table()


def lookup_warranty_years(model: str) -> float:
    if _is_blank(model):
        return DEFAULT_WARRANTY_YEARS
    m = str(model).strip()
    if m in _warranty_table:
        return _warranty_table[m]
    for key in sorted(_warranty_table.keys(), key=len, reverse=True):
        if key and key in m:
            return _warranty_table[key]
    for key in sorted(_warranty_table.keys(), key=len, reverse=True):
        if key and m in key:
            return _warranty_table[key]
    return DEFAULT_WARRANTY_YEARS


# ==================== 本地故障对照表 ====================
def load_fault_mapping() -> dict:
    path = Path(FAULT_MAPPING_FILE)
    if not path.exists():
        path.write_text(
            json.dumps(DEFAULT_FAULT_MAPPING, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return dict(DEFAULT_FAULT_MAPPING)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return dict(DEFAULT_FAULT_MAPPING)


def save_fault_mapping(mapping: dict):
    Path(FAULT_MAPPING_FILE).write_text(
        json.dumps(mapping, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


_fault_map = load_fault_mapping()


def get_fault_cn(fault_en: str) -> str:
    if _is_blank(fault_en):
        return ""
    key = str(fault_en).strip()
    if key in _fault_map:
        return _fault_map[key]
    
    # 字典里没有，才调大模型（且结果会存入字典，下次直接白嫖）
    cn = translate(key, task="fault_class")
    if cn:
        _fault_map[key] = cn
        save_fault_mapping(_fault_map)
    return cn


def fault_from_sku(sku_text: str) -> str:
    if _is_blank(sku_text):
        return ""
    text = str(sku_text).strip()

    # ① 关键词匹配
    for key in sorted(SKU_TO_FAULT.keys(), key=len, reverse=True):
        if key in text:
            return SKU_TO_FAULT[key]

    # ② 查历史缓存
    cache_key = f"__SKU__{text}"
    if cache_key in _fault_map:
        return _fault_map[cache_key]

    # ③ LLM 兜底
    cn = classify_fault_from_sku(text)
    if cn:
        _fault_map[cache_key] = cn
        save_fault_mapping(_fault_map)
        return cn

    return ""


# ==================== 日期 ====================
def parse_date(val):
    if pd.isna(val) or val is None or str(val).strip() == "":
        return None
    s = str(val).strip()[:10] # 截取前10位，过滤时间戳
    for fmt in ("%Y/%m/%d", "%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y", "%d/%m/%Y", "%Y%m%d"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def normalize_date(val) -> str:
    d = parse_date(val)
    return d.strftime("%Y/%m/%d") if d else ""


# ==================== 产品BU ====================
def get_product_bu(device_name: str) -> str:
    if _is_blank(device_name):
        return ""
    name = str(device_name).upper()
    for key in sorted(PRODUCT_BU_MAP.keys(), key=len, reverse=True):
        if key.upper() in name:
            return PRODUCT_BU_MAP[key]
    return ""


# ==================== 型号 ====================
def extract_model(device_name: str) -> str:
    if _is_blank(device_name):
        return ""
    name = str(device_name).strip()
    name = re.sub(r"^DREAME\s*-\s*", "", name, flags=re.IGNORECASE)
    name = re.sub(r"^Dreame\s+", "", name, flags=re.IGNORECASE)
    return name.strip()


# ==================== NFF 判定 ====================
NFF_KEYWORDS_NON_QUALITY = [
    "买配件", "购买配件", "购买机器", "购买耗材", "买整机", "补购", "加购",
    "buy", "purchase", "咨询产品", "询问产品", "问一下产品",
    "如何使用产品", "怎么用产品", "用户手册", "保修政策", "保修咨询",
    "inquiry", "consult", "无质量问题", "非质量问题", "仅备件", "仅咨询",
    "仅购买", "非维修", "非故障", "spare part only", "inquiry only", 
    "purchase only", "no quality issue",
]


def _has_nff_signal(text) -> bool:
    if _is_blank(text): return False
    t = str(text).upper()
    return "NFF" in t or "NO FAULT FOUND" in t


def _has_non_quality_signal(text) -> bool:
    if _is_blank(text): return False
    t = str(text)
    for kw in NFF_KEYWORDS_NON_QUALITY:
        if kw in t: return True
    return False


def _decide_nff(fault3, repair_details, tech_notes, complaint) -> bool:
    if _has_nff_signal(tech_notes): return True
    if not _is_blank(tech_notes): return False

    if _has_nff_signal(repair_details): return True
    if not _is_blank(repair_details): return False

    if _has_nff_signal(fault3): return True
    if not _is_blank(fault3): return False

    if _has_non_quality_signal(complaint): return True

    return False


# ==================== 类别 ====================
def determine_category(warranty_years: float, date_created, date_purchased,
                       doa_validation: str, fault3: str, repair_details: str,
                       tech_notes: str, complaint: str, recent_count: int) -> str:
    if date_created and date_purchased:
        days = (date_created - date_purchased).days
        limit = (warranty_years or DEFAULT_WARRANTY_YEARS) * 365
        if days > limit:
            return "维修（保外）"

    if str(doa_validation).strip().upper() == "DOA":
        return "DOA"

    if recent_count >= 3: return "多返"
    if recent_count == 2: return "二返"

    if _decide_nff(fault3, repair_details, tech_notes, complaint):
        return "无异常（NFF）"

    return "维修（保内）"


# ==================== 单行处理 ====================
def process_row(row: pd.Series, sn_recent_count: dict) -> dict:
    out = {}

    # 10 月表头映射
    for src, dst in COLUMN_MAP.items():
        out[dst] = row.get(src, "")

    out["工单号"] = out.get("工单号", "")
    out["区域"] = FIXED_REGION
    out["代理"] = FIXED_AGENT
    out["型号"] = extract_model(out.get("设备名称", ""))
    out["产品BU"] = get_product_bu(out.get("设备名称", ""))
    out["日期"] = normalize_date(out.get("关单日期", ""))
    out["SN"] = out.get("SN", "")

    fault3 = str(out.get("Fault Classification 3", "") or "").strip()
    sku_text = str(out.get("SKU consumption", "") or "").strip()
    complaint_raw = str(out.get("用户描述原文", "") or "").strip()

    # 故障类型（一级）判断
    if not _is_blank(fault3):
        out["故障类型（一级）"] = get_fault_cn(fault3)
    else:
        sku_fault = fault_from_sku(sku_text)
        if sku_fault:
            out["故障类型（一级）"] = sku_fault
        elif not _is_blank(complaint_raw):
            out["故障类型（一级）"] = translate(complaint_raw[:50], task="fault_from_complaint")
        else:
            out["故障类型（一级）"] = ""

    # 处理方式判断
    raw_service = str(out.get("Type of Service原值", "") or "").strip().lower()
    if raw_service == "credit_claim":
        text = "\n".join([
            str(out.get("用户描述原文", "") or ""),
            str(out.get("Repair Details", "") or ""),
            str(out.get("Repair Details (Technician Notes)", "") or ""),
        ])
        out["处理方式"] = classify_credit_claim(text)
    else:
        out["处理方式"] = PROCESS_METHOD_MAP.get(raw_service, "")

    # 类别判断
    date_created = parse_date(out.get("创建日期", ""))
    date_purchased = parse_date(out.get("购买日期", ""))
    warranty_years = lookup_warranty_years(out["型号"])

    sn = str(out.get("SN", "")).strip().upper()
    recent_count = sn_recent_count.get(sn, 1)

    tech_notes_str = str(out.get("Repair Details (Technician Notes)", "") or "")
    repair_details_str = str(out.get("Repair Details", "") or "")

    out["类别"] = determine_category(
        warranty_years=warranty_years,
        date_created=date_created,
        date_purchased=date_purchased,
        doa_validation=out.get("DOA Validation", ""),
        fault3=fault3,
        repair_details=repair_details_str,
        tech_notes=tech_notes_str,
        complaint=complaint_raw,
        recent_count=recent_count,
    )

    # ---- K 故障描述（限制长度，省钱大法）----
    # 只截取前 150 个字符去翻译，防止用户小作文烧钱
    complaint_short = complaint_raw[:150] + "..." if len(complaint_raw) > 150 else complaint_raw
    out["故障描述"] = (
        translate(complaint_short, task="complaint")
        if not _is_blank(complaint_raw) else ""
    )

    # ---- L 维修详情（限制长度，省钱大法）----
    tech = str(out.get("Repair Details (Technician Notes)", "") or "").strip()
    tech_short = tech[:150] + "..." if len(tech) > 150 else tech
    out["维修详情"] = (
        translate(tech_short, task="tech_notes")
        if not _is_blank(tech) else ""
    )

    return out


# ==================== 批量处理 ====================
def process_dataframe(df: pd.DataFrame,
                      progress_cb=None,
                      on_row_done=None,
                      stop_flag_getter=None) -> pd.DataFrame:
    
    # 1. 过滤掉完全没数据的空行
    df = df.dropna(how='all').copy()
    
    # 2. 过滤掉连工单号都没有的行（防止脏数据报错）
    if "Ticket Number" in df.columns:
        df = df[df["Ticket Number"].notna() & (df["Ticket Number"].astype(str).str.strip() != "")].copy()

    sn_recent_count = {}
    from collections import defaultdict
    by_sn = defaultdict(list)

    for idx, row in df.iterrows():
        sn = str(row.get("Serial Number", "")).strip().upper()
        d = parse_date(row.get("Date Created", ""))
        if sn and d:
            by_sn[sn].append(d)

    for sn in by_sn:
        by_sn[sn].sort()

    for idx, row in df.iterrows():
        sn = str(row.get("Serial Number", "")).strip().upper()
        d = parse_date(row.get("Date Created", ""))
        if not sn or not d or sn not in by_sn:
            sn_recent_count[idx] = 1
            continue
        window_start = d - pd.Timedelta(days=REFURB_WINDOW_DAYS)
        count = sum(1 for x in by_sn[sn] if window_start <= x <= d)
        sn_recent_count[idx] = count

    rows = []
    total = len(df)
    for i, (idx, row) in enumerate(df.iterrows()):
        if stop_flag_getter and stop_flag_getter():
            break

        try:
            recent = sn_recent_count.get(idx, 1)
            sn_key = str(row.get("Serial Number", "")).strip().upper()
            out = process_row(row, {sn_key: recent})
            rows.append(out)
            if on_row_done:
                on_row_done(out)
        except Exception as e:
            print(f"单行报错跳过 [工单号: {row.get('Ticket Number', '')}]：", e)
            empty = {col: "" for col in OUTPUT_COLUMNS}
            empty["工单号"] = row.get("Ticket Number", "ERROR")
            rows.append(empty)
            if on_row_done:
                on_row_done(empty)

        if progress_cb:
            progress_cb(i + 1, total)

    result = pd.DataFrame(rows)
    for col in OUTPUT_COLUMNS:
        if col not in result.columns:
            result[col] = ""
    return result[OUTPUT_COLUMNS]