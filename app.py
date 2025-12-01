import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# -------------------------------
# Streamlit Page Config
# -------------------------------
st.set_page_config(page_title="유의설계사 분석 대시보드", layout="wide")

# -------------------------------
# Scrollbar CSS
# -------------------------------
scrollbar_css = """
<style>
[data-testid="stDataFrame"] div {
    scrollbar-color: auto !important;
    scrollbar-width: auto !important;
}
.stDataFrame { overflow: auto !important; }
</style>
"""
st.markdown(scrollbar_css, unsafe_allow_html=True)

# -------------------------------
# Load Dataset
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_pickle("agent_example_small.pkl")

    # percentage = payouts/claims
    # df["claims"] = df["claims"].replace(0, np.nan)
    
    df["percentage"] = np.where(
    df["claims"] == 0,
    np.nan,
    (df["payouts"] / df["claims"]) * 100)
    df["percentage"] = df["percentage"].round(2)

    return df

df = load_data()

# -------------------------------
# Session saved groups
# -------------------------------
if "saved_groups" not in st.session_state:
    st.session_state.saved_groups = {}

# -------------------------------
# Sidebar Filters
# -------------------------------
st.sidebar.title("🔧 필터 조건 설정")

nc_min, nc_max = int(df["new_contracts"].min()), int(df["new_contracts"].max())
cl_min, cl_max = int(df["claims"].min(skipna=True)), int(df["claims"].max(skipna=True))
po_min, po_max = int(df["payouts"].min()), int(df["payouts"].max())
per_min, per_max = int(df["percentage"].min(skipna=True)), int(df["percentage"].max(skipna=True))


# ----------------------------------------------------
# Load pending saved group
# ----------------------------------------------------
if "pending_load" in st.session_state:
    cond = st.session_state.saved_groups[st.session_state.pending_load]
    st.session_state.new_contracts_range = cond["new_contracts_range"]
    st.session_state.claims_range = cond["claims_range"]
    st.session_state.payouts_range = cond["payouts_range"]
    st.session_state.percentage_range = cond["percentage_range"]
    del st.session_state.pending_load

# ----------------------------------------------------
# Initialize session vars
# ----------------------------------------------------
if "new_contracts_range" not in st.session_state:
    st.session_state.new_contracts_range = (nc_min, nc_max)
if "claims_range" not in st.session_state:
    st.session_state.claims_range = (cl_min, cl_max)
if "payouts_range" not in st.session_state:
    st.session_state.payouts_range = (po_min, po_max)
if "percentage_range" not in st.session_state:
    st.session_state.percentage_range = (per_min, per_max)


# ----------------------------------------------------
# Sliders
# ----------------------------------------------------
new_range = st.sidebar.slider("신계약 건수 범위", nc_min, nc_max,
                              st.session_state.new_contracts_range, key="new_contracts_range")
cl_range = st.sidebar.slider("청구 건수 범위", cl_min, cl_max,
                             st.session_state.claims_range, key="claims_range")
po_range = st.sidebar.slider("지급 건수 범위", po_min, po_max,
                             st.session_state.payouts_range, key="payouts_range")
per_range = st.sidebar.slider("Percentage (%) 범위", per_min, per_max,
                              st.session_state.percentage_range, key="percentage_range", step=1)


# ----------------------------------------------------
# Save group
# ----------------------------------------------------
st.sidebar.subheader("💾 조건 저장")
gname = st.sidebar.text_input("조건 그룹 이름 입력")

if st.sidebar.button("저장"):
    if gname.strip():
        st.session_state.saved_groups[gname] = {
            "new_contracts_range": new_range,
            "claims_range": cl_range,
            "payouts_range": po_range,
            "percentage_range": per_range,
        }
        st.sidebar.success(f"{gname} 저장 완료")
    else:
        st.sidebar.warning("이름을 입력하세요.")


# ----------------------------------------------------
# Load group
# ----------------------------------------------------
st.sidebar.subheader("📂 조건 불러오기")

if st.session_state.saved_groups:
    sel = st.sidebar.selectbox("저장된 그룹", list(st.session_state.saved_groups.keys()))
    if st.sidebar.button("불러오기"):
        st.session_state.pending_load = sel
        st.rerun()


# ============================================================
# Filtering Logic — ★ 핵심 수정: claims=0 (percentage NaN) 때문에 제거되지 않도록 보정
# ============================================================

is_full = (
    new_range == (nc_min, nc_max) and
    cl_range == (cl_min, cl_max) and
    po_range == (po_min, po_max) and
    per_range == (per_min, per_max)
)

if is_full:
    filtered = df.copy()
else:
    filtered = df[
        df["new_contracts"].between(*new_range)
        & df["claims"].between(*cl_range)
        & df["payouts"].between(*po_range)
        & (df["percentage"].between(*per_range) | df["percentage"].isna())
    ]

unique_count = filtered["agent_id"].nunique()

# ============================================================
# Layout
# ============================================================
left, right = st.columns([1, 1.6])

# LEFT
with left:
    st.markdown("## 🧾 조건 만족 설계사 목록")
    st.metric("조건 만족 설계사 수", unique_count)
    st.dataframe(filtered, height=500)


# ============================================================
# RIGHT — Bar chart (전체=필터 100% 동일 보장)
# ============================================================
with right:
    st.markdown("## 📊 분포 비교 (Bar Chart)")

    def plot_bar(df_total, df_filtered, field, title):
        t = df_total[field].value_counts(dropna=False).sort_index()
        f = df_filtered[field].value_counts(dropna=False).sort_index()

        # 전체 + 필터 index 합집합
        idx = sorted(list(set(t.index) | set(f.index)))

        # 재정렬
        t = t.reindex(idx, fill_value=0)
        f = f.reindex(idx, fill_value=0)

        fig = go.Figure()

        fig.add_trace(go.Bar(x=idx, y=t, name="전체", marker_color="#6FA8DC", opacity=0.7))
        fig.add_trace(go.Bar(x=idx, y=f, name="필터", marker_color="#E06666", opacity=0.7))

        # ⭐ x축 tick 전부 표시하도록 강제
        fig.update_xaxes(
            tickmode="array",
            tickvals=idx,
            ticktext=[str(x) for x in idx]
        )

        fig.update_layout(
            title=title,
            barmode="overlay",
            height=300
        )

        st.plotly_chart(fig, use_container_width=True)
    # new/claims/payouts
    plot_bar(df, filtered, "new_contracts", "신계약 건수 분포")
    plot_bar(df, filtered, "claims", "청구 건수 분포")
    plot_bar(df, filtered, "payouts", "지급 건수 분포")

    # percentage binning 10단위
    def per_bin(v):
        return int(v // 10) * 10

    df["percentage_bin"] = df["percentage"].fillna(-1).apply(per_bin)
    filtered["percentage_bin"] = filtered["percentage"].fillna(-1).apply(per_bin)

    plot_bar(df, filtered, "percentage_bin", "Percentage 분포")
