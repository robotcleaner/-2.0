# config_common.py
"""
通用业务规则：MOVA 剔除、老机型识别
"""

# ===== MOVA 工单剔除：工单号前缀（不区分大小写）=====
MOVA_TICKET_PREFIXES = ("mova-jp", "mova_jp", "mova-", "mova_")


def is_mova_ticket(ticket_no: str) -> bool:
    """工单号是否属于 MOVA，需要剔除"""
    if not ticket_no:
        return False
    t = str(ticket_no).strip().lower()
    return any(t.startswith(p) for p in MOVA_TICKET_PREFIXES)


# ===== 老机型列表（型号归一化后做子串匹配，按长度倒序）=====
OLD_MODELS_WASHER = [
    "H12S", "H11", "H12", "H12CORE", "H12PRO", "H12SAE",
    "H14ULTRA", "M12", "M13",
]
OLD_MODELS_SWEEPER = [
    "W10PRO", "L10ULTRA", "D10SPRO", "W10", "L10SPRIME",
    "D9", "F9", "L10SPRO", "D9PRO", "L20ULTRA",
    "L20ULTRACOMPLETE",
]

# 合并 + 去重 + 按长度倒序（长型号优先匹配，避免 H12 吃掉 H12S）
OLD_MODELS = sorted(
    set(OLD_MODELS_WASHER + OLD_MODELS_SWEEPER),
    key=len, reverse=True
)


def _normalize_model(model: str) -> str:
    """型号归一化：去空格、去横杠、转大写"""
    if not model:
        return ""
    s = str(model).upper()
    for ch in (" ", "-", "_", "　", "\t", "\n"):
        s = s.replace(ch, "")
    return s


def is_old_model(model: str) -> bool:
    """型号是否命中老机型列表"""
    if not model:
        return False
    m = _normalize_model(model)
    if not m:
        return False
    for old in OLD_MODELS:
        if old in m:
            return True
    return False