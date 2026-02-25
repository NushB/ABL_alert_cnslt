import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ─────────────────────────────────────────────
# Page Config & CSS
# ─────────────────────────────────────────────
st.set_page_config(page_title="유의설계사 분석 대시보드", layout="wide")

st.markdown("""
<style>
[data-testid="stDataFrame"] div {
    scrollbar-color: auto !important;
    scrollbar-width: auto !important;
}
.stDataFrame { overflow: auto !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_pickle("agent_example_small.pkl")
    df["percentage"] = np.where(
        df["claims"] == 0, np.nan,
        (df["payouts"] / df["claims"]) * 100
    )
    df["percentage"] = df["percentage"].round(2)
    return df

df = load_data()

# ─────────────────────────────────────────────
# Session State
# ─────────────────────────────────────────────
if "saved_groups"   not in st.session_state: st.session_state.saved_groups = {}
if "table_expanded" not in st.session_state: st.session_state.table_expanded = False

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
st.sidebar.title("🔧 필터 조건 설정")

nc_min, nc_max   = int(df["new_contracts"].min()),       int(df["new_contracts"].max())
cl_min, cl_max   = int(df["claims"].min(skipna=True)),   int(df["claims"].max(skipna=True))
po_min, po_max   = int(df["payouts"].min()),             int(df["payouts"].max())
per_min, per_max = int(df["percentage"].min(skipna=True)), int(df["percentage"].max(skipna=True))

if "pending_load" in st.session_state:
    cond = st.session_state.saved_groups[st.session_state.pending_load]
    st.session_state.new_contracts_range = cond["new_contracts_range"]
    st.session_state.claims_range        = cond["claims_range"]
    st.session_state.payouts_range       = cond["payouts_range"]
    st.session_state.percentage_range    = cond["percentage_range"]
    del st.session_state.pending_load

for key, default in [
    ("new_contracts_range", (nc_min, nc_max)),
    ("claims_range",        (cl_min, cl_max)),
    ("payouts_range",       (po_min, po_max)),
    ("percentage_range",    (per_min, per_max)),
]:
    if key not in st.session_state:
        st.session_state[key] = default

new_range = st.sidebar.slider("신계약 건수 범위", nc_min, nc_max,
                               st.session_state.new_contracts_range, key="new_contracts_range")
cl_range  = st.sidebar.slider("청구 건수 범위",   cl_min, cl_max,
                               st.session_state.claims_range,        key="claims_range")
po_range  = st.sidebar.slider("지급 건수 범위",   po_min, po_max,
                               st.session_state.payouts_range,       key="payouts_range")
per_range = st.sidebar.slider("Percentage (%) 범위", per_min, per_max,
                               st.session_state.percentage_range, key="percentage_range", step=1)

st.sidebar.subheader("💾 조건 저장")
gname = st.sidebar.text_input("조건 그룹 이름 입력")
if st.sidebar.button("저장"):
    if gname.strip():
        st.session_state.saved_groups[gname] = {
            "new_contracts_range": new_range, "claims_range": cl_range,
            "payouts_range": po_range,        "percentage_range": per_range,
        }
        st.sidebar.success(f"{gname} 저장 완료")
    else:
        st.sidebar.warning("이름을 입력하세요.")

st.sidebar.subheader("📂 조건 불러오기")
if st.session_state.saved_groups:
    sel = st.sidebar.selectbox("저장된 그룹", list(st.session_state.saved_groups.keys()))
    if st.sidebar.button("불러오기"):
        st.session_state.pending_load = sel
        st.rerun()

# ─────────────────────────────────────────────
# Filter
# ─────────────────────────────────────────────
is_full = (
    new_range == (nc_min, nc_max) and cl_range == (cl_min, cl_max) and
    po_range  == (po_min, po_max) and per_range == (per_min, per_max)
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

# ─────────────────────────────────────────────
# AI Chart Helpers
# ─────────────────────────────────────────────
_BG   = "rgba(8,12,24,0.97)"
_GRID = "rgba(100,150,255,0.07)"
_BLUE = "#00B4FF"
_PINK = "#FF4FD8"


def _kde(series, pts=200, bw=0.6):
    v = series.dropna().values.astype(float)
    if len(v) < 3:
        return np.array([]), np.array([])
    lo, hi = v.min(), v.max()
    if lo == hi:
        return np.array([lo]), np.array([1.0])
    sp = (hi - lo) * 0.15
    xs = np.linspace(lo - sp, hi + sp, pts)
    sd = max(v.std() * bw, 1e-6)
    ys = np.exp(-0.5 * ((xs[:, None] - v[None, :]) / sd) ** 2).sum(axis=1)
    return xs, ys / ys.max()


def _layout(title, h=265, extra=None):
    d = dict(
        title=dict(text=title, font=dict(color="#8AC8FF", size=13), x=0.01),
        paper_bgcolor=_BG, plot_bgcolor=_BG, font=dict(color="#6688AA"),
        height=h, margin=dict(l=44, r=16, t=44, b=32),
        legend=dict(
            bgcolor="rgba(0,0,0,0.45)", bordercolor="rgba(100,150,255,0.25)",
            borderwidth=1, font=dict(size=10, color="#99BBDD"),
            orientation="h", x=0.55, y=1.12,
        ),
        xaxis=dict(gridcolor=_GRID, zeroline=False, tickfont=dict(size=9, color="#445566")),
        yaxis=dict(gridcolor=_GRID, zeroline=False, tickfont=dict(size=9, color="#445566")),
    )
    if extra:
        d.update(extra)
    return d


def fig_distribution(df_all, df_flt, field, title):
    t  = df_all[field].value_counts(dropna=False).sort_index()
    f  = df_flt[field].value_counts(dropna=False).sort_index()
    ix = sorted(set(t.index) | set(f.index))
    t, f = t.reindex(ix, fill_value=0), f.reindex(ix, fill_value=0)
    mx = max(t.max(), 1)

    def _c(s, r, g, b):
        return [f"rgba({r},{g},{b},{min(0.95, 0.2 + 0.8 * v / mx):.2f})" for v in s.values]

    fig = go.Figure([
        go.Bar(x=ix, y=t.values, name="전체", opacity=0.8,
               marker=dict(color=_c(t, 0, 180, 255), line=dict(color=_BLUE, width=0.6))),
        go.Bar(x=ix, y=f.values, name="필터", opacity=0.88,
               marker=dict(color=_c(f, 255, 80, 216), line=dict(color=_PINK, width=0.6))),
    ])

    for s, col, fill in [
        (df_all[field], _BLUE, "rgba(0,180,255,0.09)"),
        (df_flt[field], _PINK, "rgba(255,79,216,0.13)"),
    ]:
        xs, ys = _kde(s)
        if len(xs):
            fig.add_trace(go.Scatter(
                x=xs, y=ys * mx, mode="lines", showlegend=False,
                line=dict(color=col, width=2.4),
                fill="tozeroy", fillcolor=fill,
            ))

    lo = _layout(title, extra=dict(barmode="overlay"))
    lo["xaxis"].update(tickmode="array", tickvals=ix, ticktext=[str(v) for v in ix])
    fig.update_layout(**lo)
    return fig


def fig_percentage(df_all, df_flt):
    def _b(v): return int(v // 10) * 10
    ab = df_all["percentage"].fillna(-1).apply(_b)
    fb = df_flt["percentage"].fillna(-1).apply(_b)
    t, f = ab.value_counts().sort_index(), fb.value_counts().sort_index()
    ix  = sorted(set(t.index) | set(f.index))
    t, f = t.reindex(ix, fill_value=0), f.reindex(ix, fill_value=0)
    lbl = [f"{v}%" if v >= 0 else "N/A" for v in ix]
    mx = max(t.max(), 1)

    def _c(s, r, g, b):
        return [f"rgba({r},{g},{b},{min(0.95, 0.2 + 0.8 * v / mx):.2f})" for v in s.values]

    fig = go.Figure([
        go.Bar(x=lbl, y=t.values, name="전체", opacity=0.8,
               marker=dict(color=_c(t, 0, 180, 255), line=dict(color=_BLUE, width=0.6))),
        go.Bar(x=lbl, y=f.values, name="필터", opacity=0.88,
               marker=dict(color=_c(f, 255, 80, 216), line=dict(color=_PINK, width=0.6))),
    ])
    fig.update_layout(**_layout("📉 Percentage 분포 (10% 구간)", extra=dict(barmode="overlay")))
    return fig


def fig_radar(df_all, df_flt):
    fields = ["new_contracts", "claims", "payouts", "percentage"]
    labels = ["신계약", "청구", "지급", "Pct(%)"]
    tm = [df_all[f].mean(skipna=True) or 0 for f in fields]
    fm = [df_flt[f].mean(skipna=True) or 0 for f in fields]
    mx = [max(a, b) or 1 for a, b in zip(tm, fm)]
    tn = [v / m * 100 for v, m in zip(tm, mx)]
    fn = [v / m * 100 for v, m in zip(fm, mx)]

    fig = go.Figure()
    for vals, col, fill, nm in [
        (tn, _BLUE, "rgba(0,180,255,0.12)", "전체 평균"),
        (fn, _PINK, "rgba(255,79,216,0.16)", "필터 평균"),
    ]:
        fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]], theta=labels + [labels[0]],
            fill="toself", name=nm,
            line=dict(color=col, width=2.2), fillcolor=fill,
            marker=dict(size=6, color=col),
        ))

    fig.update_layout(
        title=dict(text="🎯 평균 비교 (Radar)", font=dict(color="#8AC8FF", size=13), x=0.01),
        polar=dict(
            bgcolor="rgba(8,12,24,0.0)",
            radialaxis=dict(visible=True, range=[0, 115], gridcolor=_GRID,
                            tickfont=dict(color="#445566", size=8),
                            linecolor="rgba(100,150,255,0.15)"),
            angularaxis=dict(gridcolor=_GRID, tickfont=dict(color="#99BBDD", size=12),
                             linecolor="rgba(100,150,255,0.2)"),
        ),
        paper_bgcolor=_BG, font=dict(color="#6688AA"), height=265,
        margin=dict(l=44, r=44, t=44, b=32),
        legend=dict(bgcolor="rgba(0,0,0,0.45)", bordercolor="rgba(100,150,255,0.25)",
                    borderwidth=1, font=dict(size=10, color="#99BBDD"), x=0.72, y=1.12),
    )
    return fig


def fig_scatter(df_all, df_flt):
    fig = go.Figure([
        go.Scatter(
            x=df_all["claims"], y=df_all["payouts"],
            mode="markers", name="전체",
            marker=dict(
                color=df_all["new_contracts"],
                colorscale=[
                    [0,   "rgba(0,40,140,0.20)"],
                    [0.4, "rgba(0,140,255,0.40)"],
                    [1,   "rgba(80,255,230,0.65)"],
                ],
                size=5, line=dict(width=0),
            ),
        ),
        go.Scatter(
            x=df_flt["claims"], y=df_flt["payouts"],
            mode="markers", name="필터",
            marker=dict(
                color=_PINK, size=8, opacity=0.88,
                line=dict(color="rgba(255,180,240,0.6)", width=1),
            ),
        ),
    ])
    lo = _layout("⚡ 청구 vs 지급 산점도")
    lo["xaxis"].update(title=dict(text="청구 건수", font=dict(size=10)))
    lo["yaxis"].update(title=dict(text="지급 건수", font=dict(size=10)))
    fig.update_layout(**lo)
    return fig


# ─────────────────────────────────────────────
# MAIN LAYOUT
# ─────────────────────────────────────────────
hcol, mcol, bcol = st.columns([3.5, 1.5, 1])
with hcol:
    st.markdown("## 🧾 조건 만족 설계사 목록")
with mcol:
    st.metric("조건 만족 설계사 수", unique_count)
with bcol:
    st.write("")
    st.write("")
    label = "🔼 접기" if st.session_state.table_expanded else "🔽 펼치기"
    if st.button(label, use_container_width=True):
        st.session_state.table_expanded = not st.session_state.table_expanded
        st.rerun()

# ── 펼침 상태: 전체 너비 테이블 + 2열 차트 그리드
if st.session_state.table_expanded:
    st.dataframe(filtered, use_container_width=True, height=580)
    st.markdown("---")
    st.markdown("## 📊 분포 비교 — AI Dashboard")

    r1c1, r1c2 = st.columns(2)
    with r1c1:
        st.plotly_chart(fig_distribution(df, filtered, "new_contracts", "📈 신계약 건수 분포"),
                        use_container_width=True)
    with r1c2:
        st.plotly_chart(fig_distribution(df, filtered, "claims", "📋 청구 건수 분포"),
                        use_container_width=True)

    r2c1, r2c2 = st.columns(2)
    with r2c1:
        st.plotly_chart(fig_distribution(df, filtered, "payouts", "💰 지급 건수 분포"),
                        use_container_width=True)
    with r2c2:
        st.plotly_chart(fig_percentage(df, filtered), use_container_width=True)

    r3c1, r3c2 = st.columns(2)
    with r3c1:
        st.plotly_chart(fig_radar(df, filtered),   use_container_width=True)
    with r3c2:
        st.plotly_chart(fig_scatter(df, filtered), use_container_width=True)

# ── 접힘 상태: 왼쪽 테이블 + 오른쪽 차트
else:
    left, right = st.columns([1, 1.6])
    with left:
        st.dataframe(filtered, height=500)
    with right:
        st.markdown("## 📊 분포 비교 — AI Dashboard")
        st.plotly_chart(fig_distribution(df, filtered, "new_contracts", "📈 신계약 건수 분포"),
                        use_container_width=True)
        st.plotly_chart(fig_distribution(df, filtered, "claims", "📋 청구 건수 분포"),
                        use_container_width=True)
        st.plotly_chart(fig_distribution(df, filtered, "payouts", "💰 지급 건수 분포"),
                        use_container_width=True)
        st.plotly_chart(fig_percentage(df, filtered), use_container_width=True)
        st.plotly_chart(fig_radar(df, filtered),   use_container_width=True)
        st.plotly_chart(fig_scatter(df, filtered), use_container_width=True)
