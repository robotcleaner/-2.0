import pandas as pd
from config import COLUMN_MAP

df = pd.read_excel("原始工单.xlsx", engine="openpyxl")

print("=== 原始表行数 ===", len(df))
print("=== 原始表列数 ===", len(df.columns))
print()

print("=== 检查每个 COLUMN_MAP 的 key 是否在原始表列名里 ===")
for src, dst in COLUMN_MAP.items():
    hit = src in df.columns
    print(f"[{'OK ' if hit else 'MISS'}] {src!r}  →  {dst!r}")