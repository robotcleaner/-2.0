import pandas as pd
from processor import process_row

df = pd.read_excel("原始工单.xlsx", engine="openpyxl")

print("=== 原始表行数 ===", len(df))
print("=== 原始表列数 ===", len(df.columns))

row = df.iloc[0]
out = process_row(row)

print()
print("=== 处理结果（第 1 行）===")
for k, v in out.items():
    print(f"{k!r}: {v!r}")