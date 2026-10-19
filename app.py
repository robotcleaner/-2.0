import io
import os
from pathlib import Path
import pandas as pd
import streamlit as st

from processor import process_dataframe as process_dataframe_au
from config import OUTPUT_COLUMNS as OUTPUT_COLUMNS_AU

from processor_kr import process_dataframe_kr
from config_kr import OUTPUT_COLUMNS_KR

from processor_kolon import process_dataframe_kolon, find_header_row
from config_kolon import OUTPUT_COLUMNS_KOLON

# 日本模块（NGK）
from processor_jp import process_dataframe_jp
from config_jp import OUTPUT_COLUMNS_JP

# 日本模块（ITC）
from processor_itc import process_dataframe_itc, find_header_row as find_header_row_itc
from config_itc import OUTPUT_COLUMNS_ITC


st.set_page_config(
    page_title="维修工单自动处理",
    page_icon="🔧",
    layout="wide",
)

st.title("🔧 维修工单自动处理工具")
st.caption("上传海外维修商原始工单 → 自动翻译 + 分类 + 格式化 → 输出标准化工单")

# ========== 区域选择 ==========
region = st.selectbox(
    "选择区域 / 服务商",
    options=["澳洲（PAU）", "韩国（SK）", "韩国（KOLON）", "日本（NGK）", "日本（ITC）"],
    index=0,
)

if region.startswith("澳洲"):
    PARTIAL_FILE = "partial_result_au.csv"
    OUTPUT_COLUMNS = OUTPUT_COLUMNS_AU
    process_dataframe = process_dataframe_au
    TICKET_NUMBER_COL = "Ticket Number"
    OUTPUT_TICKET_COL = "工单号"
    IS_KOLON = False
elif "KOLON" in region:
    PARTIAL_FILE = "partial_result_kolon.csv"
    OUTPUT_COLUMNS = OUTPUT_COLUMNS_KOLON
    process_dataframe = process_dataframe_kolon
    TICKET_NUMBER_COL = "접수번호"
    OUTPUT_TICKET_COL = "工单号"
    IS_KOLON = True
elif "ITC" in region:
    PARTIAL_FILE = "partial_result_itc.csv"
    OUTPUT_COLUMNS = OUTPUT_COLUMNS_ITC
    process_dataframe = process_dataframe_itc
    TICKET_NUMBER_COL = "RMA"
    OUTPUT_TICKET_COL = "工单号"
    IS_KOLON = True   # 复用 KOLON 的"自动探测表头"分支
elif "NGK" in region or "日本" in region:
    PARTIAL_FILE = "partial_result_jp.csv"
    OUTPUT_COLUMNS = OUTPUT_COLUMNS_JP
    process_dataframe = process_dataframe_jp
    TICKET_NUMBER_COL = "唯一键"
    OUTPUT_TICKET_COL = "唯一键"
    IS_KOLON = False
else:
    PARTIAL_FILE = "partial_result_kr.csv"
    OUTPUT_COLUMNS = OUTPUT_COLUMNS_KR
    process_dataframe = process_dataframe_kr
    TICKET_NUMBER_COL = "접수번호"
    OUTPUT_TICKET_COL = "工单号"
    IS_KOLON = False

st.info(f"当前区域：**{region}**，输出列数：{len(OUTPUT_COLUMNS)}")

if "stop_flag" not in st.session_state:
    st.session_state.stop_flag = False
if "processing" not in st.session_state:
    st.session_state.processing = False

with st.sidebar:
    st.header("⚙️ 使用说明")
    st.markdown("""
    1. 上传原始工单 `.xlsx`
    2. 点击「开始处理」
    3. 可随时点「⏹ 暂停」
    4. 可随时点「📥 导出已处理部分」
    5. 重新上传同一文件 → 点开始 → 自动跳过已处理
    """)
    st.divider()
    st.caption("模型：硅基流动 DeepSeek-V3")

# ========== 上传 ==========
uploaded = st.file_uploader(
    "上传原始工单（.xlsx）",
    type=["xlsx"],
    accept_multiple_files=False,
)

if uploaded is not None:
    # 读文件
    try:
        if IS_KOLON:
            # KOLON / ITC：先自动探测表头行，再按探测结果跳过
            uploaded.seek(0)
            xlsx_bytes = uploaded.read()
            uploaded.seek(0)
            if "ITC" in region:
                skip_rows = find_header_row_itc(xlsx_bytes)
                st.caption(f"ITC 自动探测表头：跳过前 {skip_rows} 行")
            else:
                skip_rows = find_header_row(xlsx_bytes)
                st.caption(f"KOLON 自动探测表头：跳过前 {skip_rows} 行")
            df_raw = pd.read_excel(io.BytesIO(xlsx_bytes), engine="openpyxl",
                                   skiprows=skip_rows)
        else:
            df_raw = pd.read_excel(uploaded, engine="openpyxl")
    except Exception as e:
        st.error(f"读取失败：{e}")
        st.stop()

    st.success(f"已读取 {len(df_raw)} 条工单，共 {len(df_raw.columns)} 列")
    with st.expander("预览原始数据（前 5 行）"):
        st.dataframe(df_raw.head(5), use_container_width=True)

    # ========== 断点续跑 ==========
    if Path(PARTIAL_FILE).exists():
        try:
            partial_df = pd.read_csv(PARTIAL_FILE, encoding="utf-8-sig")
            if OUTPUT_TICKET_COL in partial_df.columns:
                done_ids = set(partial_df[OUTPUT_TICKET_COL].astype(str))
            else:
                done_ids = set()
            st.info(f"📂 检测到 `{PARTIAL_FILE}`，已处理 **{len(partial_df)}** 条")
        except Exception:
            partial_df = None
            done_ids = set()
    else:
        partial_df = None
        done_ids = set()

    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        if st.button("🚀 开始处理", type="primary",
                     disabled=st.session_state.processing):
            st.session_state.stop_flag = False
            st.session_state.processing = True
            st.rerun()

    with col2:
        if st.button("⏹ 暂停", disabled=not st.session_state.processing):
            st.session_state.stop_flag = True
            st.session_state.processing = False
            st.warning("已暂停。可随时导出。")

    with col3:
        if partial_df is not None and len(partial_df) > 0:
            export_df = partial_df[[c for c in OUTPUT_COLUMNS if c in partial_df.columns]]
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                export_df.to_excel(writer, index=False, sheet_name="已处理")
            buffer.seek(0)
            st.download_button(
                label=f"📥 导出已处理部分（{len(export_df)} 条）",
                data=buffer,
                file_name="已处理部分.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

    if st.session_state.processing:
        df_to_process = df_raw.copy()
        if done_ids and TICKET_NUMBER_COL in df_to_process.columns:
            df_to_process = df_to_process[
                ~df_to_process[TICKET_NUMBER_COL].astype(str).isin(done_ids)
            ]

        st.markdown(f"**本次需要处理：{len(df_to_process)} 条**")
        progress_bar = st.progress(0.0)
        status_text = st.empty()

        def cb(done, total):
            pct = done / total if total else 0
            progress_bar.progress(pct)
            status_text.text(f"处理进度：{done} / {total}")

        def on_row_done(out_dict):
            filtered = {col: out_dict.get(col, "") for col in OUTPUT_COLUMNS}
            df_one = pd.DataFrame([filtered])
            df_one.to_csv(
                PARTIAL_FILE,
                mode="a",
                header=not Path(PARTIAL_FILE).exists(),
                index=False,
                encoding="utf-8-sig",
            )

        def stop_getter():
            return st.session_state.stop_flag

        try:
            result_df = process_dataframe(
                df_to_process,
                progress_cb=cb,
                on_row_done=on_row_done,
                stop_flag_getter=stop_getter,
            )
            st.session_state.processing = False
            progress_bar.progress(1.0)
            status_text.text("✅ 本次处理完成")

            if Path(PARTIAL_FILE).exists():
                all_df = pd.read_csv(PARTIAL_FILE, encoding="utf-8-sig")
                all_df = all_df[[c for c in OUTPUT_COLUMNS if c in all_df.columns]]
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                    all_df.to_excel(writer, index=False, sheet_name="优化后工单")
                buffer.seek(0)
                st.download_button(
                    label=f"📥 下载全部已处理（{len(all_df)} 条）",
                    data=buffer,
                    file_name="优化后工单.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    type="primary",
                )
        except Exception as e:
            st.session_state.processing = False
            st.error(f"处理中断：{e}")

with st.sidebar:
    st.divider()
    if st.button("🗑 清空已处理记录"):
        if Path(PARTIAL_FILE).exists():
            os.remove(PARTIAL_FILE)
            st.success("已清空")
        else:
            st.info("没有可清空的记录")