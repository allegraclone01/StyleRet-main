"""
spe_ret 辅助函数：个股波动率时序 / 历史排位 / 截面均值 / 绘图
数据路径: E:/SJTU/intern/gtht/barra/data_base/spe_ret/v1
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
from rqdatac import get_specific_return

# 中文字体
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

BASE_DIR = Path(__file__).resolve().parent
SRCDIR = BASE_DIR / "data_base" / "spe_ret"
IDXDIR = BASE_DIR / "data_base" / "index_component_日频"
MODE = "v1"
DATA_DIR = SRCDIR / MODE


# ---------- 增量更新 ----------
def _latest_existing_date():
    """返回本地 spe_ret 数据中的最新日期，无数据则返回 None。"""
    files = sorted(DATA_DIR.glob("*.parquet"))
    if not files:
        return None
    last = pd.read_parquet(files[-1])
    return last["date"].max()


def update_spe_ret(end_date):
    """增量更新 spe_ret 至 end_date（含）。已是最新则直接返回新补数据。"""
    end = pd.Timestamp(end_date)
    latest = _latest_existing_date()
    if latest is not None and latest >= end:
        return pd.DataFrame(columns=["date", "stock_id", "spe_ret"])

    # 全 A 成分股日历
    dict_a = pd.read_pickle(IDXDIR / "866011.RI_19_26D_dict.pkl")
    all_dates = sorted(dict_a.keys())

    # 找出需要更新的日期
    if latest is None:
        missing_dates = [d for d in all_dates if d <= end]
    else:
        missing_dates = [d for d in all_dates if latest < d <= end]

    if not missing_dates:
        return pd.DataFrame(columns=["date", "stock_id", "spe_ret"])

    print(f"更新 spe_ret ({MODE})：{missing_dates[0].date()} ~ {missing_dates[-1].date()} 共 {len(missing_dates)} 天")

    new_rows = []
    for dt in missing_dates:
        stk = dict_a[dt].index.tolist()
        stk_fb = [s for s in stk if not s.endswith("BJSE")]
        spe = get_specific_return(stk_fb, dt, dt, model=MODE,
                                  industry_mapping='sws_2021').T

        df_spe = pd.DataFrame({
            'date': [dt] * len(spe),
            'stock_id': spe.index.tolist(),
            'spe_ret': spe.values.flatten()
        })
        new_rows.append(df_spe)

        quarter_path = DATA_DIR / f"{dt.year}Q{(dt.month - 1) // 3 + 1}.parquet"
        try:
            quarter_path.parent.mkdir(parents=True, exist_ok=True)
            if quarter_path.exists():
                existing = pd.read_parquet(quarter_path)
                combined = pd.concat([existing, df_spe], ignore_index=True)
                combined = combined.drop_duplicates(subset=['date', 'stock_id'], keep='last')
                combined.to_parquet(quarter_path, compression='zstd', index=False)
            else:
                df_spe.to_parquet(quarter_path, compression='zstd', index=False)
        except Exception:
            pass

        print(f"  ✓ {dt.date()}")

    return pd.concat(new_rows, ignore_index=True) if new_rows else pd.DataFrame(columns=["date", "stock_id", "spe_ret"])


# ---------- 数据加载 ----------
def _quarters_between(start_date, end_date):
    """返回 [start_date, end_date] 覆盖的季度标签列表，如 ['2019Q1','2019Q2']。"""
    quarters = []
    d = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    while d <= end:
        quarters.append(f"{d.year}Q{(d.month - 1) // 3 + 1}")
        # 跳到下一季度第一天
        next_month = ((d.month - 1) // 3 + 1) * 3 + 1
        if next_month > 12:
            d = pd.Timestamp(year=d.year + 1, month=1, day=1)
        else:
            d = pd.Timestamp(year=d.year, month=next_month, day=1)
    return quarters


def load_spe_ret_by_date(start_date, end_date, pad_days=0):
    """按日期范围加载 spe_ret，pad_days 表示往前多加载的天数（用于滚动窗口）。"""
    start = pd.Timestamp(start_date) - pd.Timedelta(days=pad_days)
    end = pd.Timestamp(end_date)
    quarters = _quarters_between(start, end)
    files = [DATA_DIR / f"{q}.parquet" for q in quarters]
    files = [f for f in files if f.exists()]
    if not files:
        return pd.DataFrame(columns=["date", "stock_id", "spe_ret"])
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return df[(df["date"] >= start) & (df["date"] <= end)].reset_index(drop=True)


# ---------- 核心计算 ----------
def calc_daily_vol(df, window=10):
    """① 每日个股时序波动率（滚动 std）。"""
    df = df.sort_values(["stock_id", "date"])
    df["vol"] = df.groupby("stock_id")["spe_ret"].transform(
        lambda x: x.rolling(window, min_periods=window // 2).std()
    )
    return df[["date", "stock_id", "vol"]].dropna()


def calc_vol_rank(df_vol, lookback_days=242):
    """② 每日个股波动率的近一年历史排位（百分位 0~1）。"""
    df = df_vol.sort_values(["stock_id", "date"])
    df["vol_rank"] = df.groupby("stock_id")["vol"].transform(
        lambda x: x.rolling(lookback_days, min_periods=lookback_days//2).rank(pct=True)
    )
    return df[["date", "stock_id", "vol_rank"]].dropna()


def calc_vol_summary(df_vol, df_vol_rank=None, lookback_days=242):
    """③ 每日截面平均波动率 & 平均历史排位 & 波动率日增长率。"""
    avg_vol = df_vol.groupby("date")["vol"].mean().rename("avg_vol")
    if df_vol_rank is None:
        df_vol_rank = calc_vol_rank(df_vol, lookback_days)
    avg_rank = df_vol_rank.groupby("date")["vol_rank"].mean().rename("avg_vol_rank")
    summary = pd.concat([avg_vol, avg_rank], axis=1)
    summary["vol_growth"] = summary["avg_vol"].pct_change()   # 日增长率
    return summary.reset_index()


# ---------- 一站式入口 ----------
def vol_pipeline(start_date, end_date, vol_window=10, lookback_days=242, auto_update=True):
    """给定日期区间，返回 [start_date, end_date] 的平均波动率 & 平均历史分位时序。

    自动往前加载一年数据用于滚动排位计算，结果按 start_date 截断返回。
    auto_update=True 时先检查并增量更新本地数据至 end_date。
    """
    df = load_spe_ret_by_date(start_date, end_date, pad_days=lookback_days)
    if auto_update:
        new_df = update_spe_ret(end_date)
        if not new_df.empty:
            start = pd.Timestamp(start_date) - pd.Timedelta(days=lookback_days)
            end = pd.Timestamp(end_date)
            new_df = new_df[(new_df["date"] >= start) & (new_df["date"] <= end)]
            df = pd.concat([df, new_df], ignore_index=True)
            df = df.drop_duplicates(subset=["date", "stock_id"], keep="last").reset_index(drop=True)

    vol = calc_daily_vol(df, window=vol_window)
    vol_rank = calc_vol_rank(vol, lookback_days=lookback_days)
    summary = calc_vol_summary(vol, vol_rank, lookback_days=lookback_days)

    # 按请求区间截断
    summary = summary[
        (summary["date"] >= pd.Timestamp(start_date))
        & (summary["date"] <= pd.Timestamp(end_date))
    ].reset_index(drop=True)
    return summary


# ---------- 事件分析 ----------
def event_vol_analysis(event_df, summary, event_col="start_date", window=5, current_date=None):
    """对事件表每行，取事件日前后 window 个交易日的三指标均值，拼回原表。

    新增列: pre_avg_vol / pre_avg_vol_rank / pre_avg_vol_growth
            post_avg_vol / post_avg_vol_rank / post_avg_vol_growth

    Parameters
    ----------
    event_df    : pd.DataFrame  含事件日期列（如 start_date）
    summary     : pd.DataFrame  vol_pipeline 输出 [date, avg_vol, avg_vol_rank, vol_growth]
    event_col   : str           事件日期列名
    window      : int           前后各取几天（交易日，不含事件日）
    current_date: str/Timestamp 可选，追加一行"当前"快照（start=end=current_date，
                  其他列 NaN），用于对比基准

    Returns
    -------
    pd.DataFrame  原 event_df + 6 列新指标
    """
    s = summary.set_index("date").sort_index()
    cols = [c for c in ["avg_vol", "avg_vol_rank", "vol_growth"] if c in s.columns]
    NAN_ROW = pd.Series({c: np.nan for c in cols})

    pre_rows, post_rows = [], []
    for dt in pd.to_datetime(event_df[event_col]):
        pre = s[s.index < dt].tail(window)
        post = s[s.index > dt].head(window)
        pre_rows.append(pre[cols].mean())
        post_rows.append(post[cols].mean() if len(post) >= window else NAN_ROW)

    pre_df = pd.DataFrame(pre_rows, index=event_df.index).add_prefix("前5_")
    post_df = pd.DataFrame(post_rows, index=event_df.index).add_prefix("后5_")

    base = event_df.drop(columns=["peak_overlap_day", "coverage_ratio"], errors="ignore")

    # 按指标分组排列：每个指标的前 / 后并排；转百分数并保留两位小数
    ordered = []
    for col in ["avg_vol", "avg_vol_rank", "vol_growth"]:
        ordered.append((pre_df[f"前5_{col}"] * 100).round(2))
        ordered.append((post_df[f"后5_{col}"] * 100).round(2))
    ordered_df = pd.concat(ordered, axis=1)

    result = pd.concat([base, ordered_df], axis=1).sort_values(by=event_col, ascending=False)

    # 追加当前快照行
    if current_date is not None:
        cur = pd.Timestamp(current_date)
        cur_row = {col: np.nan for col in result.columns}
        cur_row["start_date"] = cur
        cur_row["end_date"] = cur
        # 前 window 天均值（后 window 天为 NaN）
        pre_cur = s[s.index < cur].tail(window)[cols].mean()
        for c in cols:
            cur_row[f"前5_{c}"] = round(pre_cur[c] * 100, 2)
            cur_row[f"后5_{c}"] = np.nan
        result = pd.concat([pd.DataFrame([cur_row]), result], ignore_index=True)

    # 日期列转短日期字符串
    for col in ["start_date", "end_date"]:
        if col in result.columns:
            result[col] = pd.to_datetime(result[col], errors="coerce").dt.strftime("%Y-%m-%d")

    return result


def _fmt_dates(ax):
    """自适应日期刻度：约18个刻度，短区间 %m-%d / 中区间 %Y-%m / 长区间 季度"""
    lo, hi = ax.get_xlim()
    days = (mdates.num2date(hi) - mdates.num2date(lo)).days
    months = max(1, int(days / 30.44))
    interval = max(1, months // 18)

    if months > 72:
        loc = mdates.MonthLocator(bymonth=(1, 4, 7, 10), bymonthday=1)
        fmt = FuncFormatter(
            lambda x, _: f"{mdates.num2date(x).year}Q{(mdates.num2date(x).month - 1) // 3 + 1}"
        )
    elif months <= 4:
        loc = mdates.MonthLocator()
        fmt = mdates.DateFormatter("%m-%d")
    else:
        loc = mdates.MonthLocator() if interval <= 1 else mdates.MonthLocator(interval=interval)
        fmt = mdates.DateFormatter("%Y-%m")

    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(fmt)
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_ha("right")
        label.set_fontsize(7.5)


# ---------- 绘图 ----------
def plot_vol_summary(summary, title="个股特质收益波动率（截面均值）", figsize=(12, 4.8),
                     events=None, start_col="start_date", end_col="end_date"):
    """双 Y 轴折线图：左轴平均波动率，右轴平均历史分位。直接 plt.show() 展示。

    events: pd.DataFrame or None  事件表，画出 start_date（虚线）和 end_date（点线）区间。
            end_date 为 "未修复" / NaT 时不画结束线。
    """
    fig, ax1 = plt.subplots(figsize=figsize, constrained_layout=True)

    ax1.plot(summary["date"], summary["avg_vol"], color="#1f77b4", label="平均波动率")
    ax1.set_ylabel("平均波动率", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.set_ylim(0, None)
    ax1.grid(alpha=0.25)

    ax2 = ax1.twinx()
    ax2.fill_between(summary["date"], summary["avg_vol_rank"], alpha=0.12, color="#ff7f0e")
    ax2.plot(summary["date"], summary["avg_vol_rank"], color="#ff7f0e",
             label="平均历史分位", lw=1.2)
    ax2.set_ylabel("平均历史分位", color="#ff7f0e")
    ax2.tick_params(axis="y", labelcolor="#ff7f0e")
    ax2.set_ylim(0, 1)
    for level, ls in [(0.25, ":"), (0.50, "--"), (0.75, ":")]:
        ax2.axhline(level, color="gray", ls=ls, lw=0.8)

    # 绘制事件区间（红色虚线=开始，红色点线=结束，浅红背景=区间）
    if events is not None:
        starts, ends = [], []
        for _, row in events.iterrows():
            s = pd.to_datetime(row[start_col], errors="coerce")
            e = pd.to_datetime(row[end_col], errors="coerce")
            if not pd.isna(s):
                starts.append(s)
            if not pd.isna(e):
                ends.append(e)
            if not pd.isna(s) and not pd.isna(e):
                ax1.axvspan(s, e, color="red", alpha=0.08)
        for s in starts:
            ax1.axvline(s, color="red", ls="--", lw=1.2)
        for e in ends:
            ax1.axvline(e, color="red", ls=":", lw=1.2)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    # 如果有事件，手工加一条图例
    if events is not None and len(events) > 0:
        from matplotlib.lines import Line2D
        lines1.append(Line2D([0], [0], color="red", ls="--", lw=1.2))
        labels1.append("事件开始")
        lines1.append(Line2D([0], [0], color="red", ls=":", lw=1.2))
        labels1.append("事件结束")
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)

    ax1.set_title(title)
    _fmt_dates(ax1)
    return fig


def plot_vol_growth(summary, title="波动增长率 MA10", figsize=(12, 2.5),
                    ma_window=10, threshold=0):
    """波动率增长率 MA 折线图，突出 0 轴，便于观察向上突破。"""
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    ma = summary.set_index("date")["vol_growth"].rolling(ma_window).mean()

    # 0 轴以上红色填充，以下绿色填充
    ax.fill_between(ma.index, ma.values, 0, where=ma.values >= threshold,
                    color="#ff4b4b", alpha=0.3)
    ax.fill_between(ma.index, ma.values, 0, where=ma.values < threshold,
                    color="#2ca02c", alpha=0.3)
    ax.plot(ma.index, ma.values, color="#333", lw=1.0)
    ax.axhline(threshold, color="black", lw=1.0)

    ax.set_ylabel(f"MA{ma_window} 增长率")
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.2, axis="y")
    _fmt_dates(ax)
    return fig


# ------------ 示例 ------------
if __name__ == "__main__":
    # 1. 波动率时序
    summary = vol_pipeline("2020-01-01", "2026-08-24")
    print(summary.tail())

    # 2. 事件分析：共性回撤期前后波动率对比
    events = pd.read_excel(BASE_DIR / "comb" / "outputs" / "Alpha私募超额指数_回撤分析.xlsx",
                           sheet_name="共性回撤期")
    result = event_vol_analysis(events, summary, event_col="start_date", window=5,current_date="2026-08-24")
    print("\n事件分析结果:")
    print(result)

    # 3. 画图
    plot_vol_summary(summary,events=events)
    plot_vol_growth(summary)
    plt.show()


# ---------- 复盘 MD 文档渲染 ----------
_MD_SCROLL_CSS = """
<style>
.md-scroll-container {
    height: 600px;
    overflow-y: auto;
    border: 1px solid #d0d7de;
    border-radius: 8px;
    padding: 20px 24px;
    background-color: #ffffff !important;
    color: #000000 !important;
    line-height: 1.7;
    font-size: 15px;
}
.md-scroll-container h1,
.md-scroll-container h2,
.md-scroll-container h3,
.md-scroll-container h4,
.md-scroll-container h5,
.md-scroll-container h6 {
    color: #000000 !important;
    font-weight: 700 !important;
    margin-top: 1.2em;
    margin-bottom: 0.5em;
    line-height: 1.3;
    display: block !important;
}
.md-scroll-container h1 { font-size: 1.8em !important; border-bottom: 2px solid #5B8FF9; padding-bottom: 0.3em; }
.md-scroll-container h2 { font-size: 1.5em !important; border-bottom: 1px solid #d0d7de; padding-bottom: 0.2em; }
.md-scroll-container h3 { font-size: 1.25em !important; }
.md-scroll-container h4 { font-size: 1.1em !important; }
.md-scroll-container h5 { font-size: 1em !important; }
.md-scroll-container h6 { font-size: 0.9em !important; }
.md-scroll-container p { margin: 0.6em 0; color: #000000 !important; }
.md-scroll-container ul, .md-scroll-container ol { margin: 0.5em 0; padding-left: 1.8em; }
.md-scroll-container li { margin: 0.3em 0; }
.md-scroll-container code {
    background-color: #f6f8fa !important;
    color: #000000 !important;
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 0.9em;
}
.md-scroll-container pre {
    background-color: #f6f8fa !important;
    color: #000000 !important;
    padding: 12px 16px;
    border-radius: 6px;
    overflow-x: auto;
    border: 1px solid #d0d7de;
}
.md-scroll-container pre code {
    background: none !important;
    padding: 0;
}
.md-scroll-container blockquote {
    border-left: 4px solid #5B8FF9;
    padding-left: 1em;
    color: #000000 !important;
    margin: 0.8em 0;
    background-color: #f6f8fa;
    padding-top: 0.4em;
    padding-bottom: 0.4em;
    padding-right: 1em;
    border-radius: 0 6px 6px 0;
}
.md-scroll-container table { border-collapse: collapse; margin: 0.8em 0; width: auto; }
.md-scroll-container th, .md-scroll-container td {
    border: 1px solid #d0d7de;
    padding: 6px 12px;
    text-align: left;
    color: #000000 !important;
    background-color: #ffffff !important;
}
.md-scroll-container th { background-color: #f6f8fa !important; font-weight: 600; }
.md-scroll-container strong { color: #000000 !important; font-weight: 700; }
.md-scroll-container em { color: #000000 !important; }
.md-scroll-container a { color: #0969da !important; text-decoration: underline; }
.md-scroll-container hr { border: none; border-top: 1px solid #d0d7de; margin: 1.2em 0; }
</style>
"""


def render_summary_md(summary_dir):
    """
    在 streamlit 中渲染 comb/end_input/summary/ 目录下的 md 复盘文档。
    - 下拉框选择文件名，一次展示一个
    - 固定高度 + 滚动条
    - 美化排版（标题、列表、表格、代码块等）
    """
    import streamlit as st
    import os

    if not os.path.isdir(summary_dir):
        st.warning(f"⚠️ 目录不存在: {summary_dir}")
        return

    md_files = sorted([f for f in os.listdir(summary_dir) if f.endswith(".md")])
    if not md_files:
        st.info("📂 summary 目录下暂无 md 文档")
        return

    selected_md = st.selectbox("选择复盘文档", md_files)
    md_path = os.path.join(summary_dir, selected_md)
    with open(md_path, "r", encoding="utf-8") as f:
        md_content = f.read()

    st.markdown(_MD_SCROLL_CSS, unsafe_allow_html=True)

    try:
        import markdown as md_lib
        html_content = md_lib.markdown(
            md_content,
            extensions=["tables", "fenced_code", "toc", "nl2br", "sane_lists"],
        )
    except ImportError:
        html_content = f"<pre style='white-space:pre-wrap;'>{md_content}</pre>"

    st.markdown(
        f'<div class="md-scroll-container">{html_content}</div>',
        unsafe_allow_html=True,
    )