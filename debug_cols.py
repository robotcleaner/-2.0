import pandas as pd

# ⚠️ 把这行换成你原始工单的真实路径
df = pd.read_excel(r"D:\1\软件区\ticket_processor\原始工单.xlsx", engine="openpyxl")

print("=== 前 20 个列名（原样打印，带引号）===")
for i, c in enumerate(df.columns[:20]):
    print(i, repr(c))

print()
print("=== 总列数 ===", len(df.columns))
print("=== 总行数 ===", len(df))