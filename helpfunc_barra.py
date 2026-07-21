import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import streamlit as st
from rqdatac import *

STYLE_COLS = ['beta', 'book_to_price', 'earnings_yield', 'growth',
       'leverage', 'liquidity', 'momentum', 'non_linear_size',
       'residual_volatility', 'size']

INDEX_NAME_MAP = {
    "000300.XSHG": "沪深300",
    "000905.XSHG": "中证500", 
    "000852.XSHG": "中证1000",
    "000016.XSHG": "上证50",
    "000688.XSHG": "科创50",
    "000680.XSHG": "科创综指",
    "399673.XSHE": "创业板50",
    "399102.XSHE": "创业板综指"
}

def cal_style_corr(df: pd.DataFrame, style_cols=None, window=20,
                   cmap='RdBu_r', annot=True, fmt='.2f'):
    """
    计算因子收益率的 Pearson 和 Spearman 时序相关系数，绘制组合热力图。

    图片上三角 = Pearson 相关系数，下三角 = Spearman 秩相关系数。

    :param df:         因子收益率 DataFrame，需包含 style_cols 中的列
    :param style_cols: 因子名称列表，默认使用全局 STYLE_COLS
    :param window:     计算相关系数的时间窗口（最近 N 个交易日）
    :param cmap:       色阶（默认 RdBu_r：红正蓝负）
    :param annot:      是否在格子中标注数值
    :param fmt:        数值格式字符串
    :return:           (pearson_corr, spearman_corr, fig)
    """
    if style_cols is None:
        style_cols = STYLE_COLS

    # ---------- 1. 截取窗口数据 ----------
    data = df[style_cols].iloc[-window:]

    # ---------- 2. 计算两种相关系数 ----------
    pearson_corr  = data.corr(method='pearson')
    spearman_corr = data.corr(method='spearman')

    # ---------- 3. 构建组合矩阵 ----------
    n = len(style_cols)
    combined = pd.DataFrame(np.eye(n), index=style_cols, columns=style_cols)

    for i in range(n):
        for j in range(n):
            if i < j:      # 上三角 → Pearson
                combined.iloc[i, j] = pearson_corr.iloc[i, j]
            elif i > j:    # 下三角 → Spearman
                combined.iloc[i, j] = spearman_corr.iloc[i, j]
            # i == j 保持 1.0

    # ---------- 4. 绘制热力图 ----------
    fig, ax = plt.subplots(figsize=(12, 10))

    mask_upper = np.tril(np.ones_like(combined, dtype=bool), k=0)
    mask_lower = np.triu(np.ones_like(combined, dtype=bool), k=0)

    # 上三角：Pearson
    sns.heatmap(
        combined, mask=mask_upper, cmap=cmap, center=0,
        annot=annot, fmt=fmt, linewidths=0.5,
        xticklabels=style_cols, yticklabels=style_cols,
        cbar=False, ax=ax, vmin=-1, vmax=1,
    )
    # 下三角：Spearman
    sns.heatmap(
        combined, mask=mask_lower, cmap=cmap, center=0,
        annot=annot, fmt=fmt, linewidths=0.5,
        xticklabels=style_cols, yticklabels=style_cols,
        cbar=False, ax=ax, vmin=-1, vmax=1,
    )

    # ---------- 5. 图例 ----------
    ax.text(
        0.98, 0.02,
        'Upper: Pearson\nLower: Spearman',
        transform=ax.transAxes, ha='right', va='bottom',
        fontsize=9, bbox=dict(boxstyle='round,pad=0.3',
                              facecolor='white', edgecolor='gray', alpha=0.9),
    )

    ax.set_title(f'Factor Return Correlation (last {window} days)', fontsize=14, pad=15)
    plt.tight_layout()

    return pearson_corr, spearman_corr, fig


def cal_rolling_corr(df, factor1, factor2, sd, ed, windows=(20, 40, 60)):
    """
    计算风格因子的滚动 Pearson 相关系数，绘制折线图。

    - 若 factor2 为具体因子名：factor1 vs factor2，多窗口（20/40/60d）
    - 若 factor2 为 "所有_XXd"：factor1 vs 其余所有因子，单窗口 XXd

    :param df:       因子收益率 DataFrame，DatetimeIndex
    :param factor1:  因子 1 列名
    :param factor2:  因子 2 列名 或 "所有_20d" / "所有_40d" / "所有_60d"
    :param sd:       起始日期
    :param ed:       结束日期
    :param windows:  多窗口模式的回溯窗口
    :return:         (rolling_corr_df, fig)
    """
    data = df.loc[sd:ed,STYLE_COLS]

    if isinstance(factor2, str) and factor2.startswith("所有_"):
        w = int(factor2.split("_")[1].replace("d", ""))
        others = [c for c in data.columns if c != factor1]
        results = {o: data[factor1].rolling(w).corr(data[o]) for o in others}
        title = f"{factor1} vs 所有因子  ({w}d 滚动相关)"
    else:
        sub = data[[factor1, factor2]].dropna()
        results = {f"{w}d": sub[factor1].rolling(w).corr(sub[factor2]) for w in windows}
        title = f"{factor1} vs {factor2}  —  Rolling Correlation"

    corr_df = pd.DataFrame(results).dropna(how="all")

    fig, ax = plt.subplots(figsize=(12, 4))
    for col in corr_df.columns:
        ax.plot(corr_df.index, corr_df[col], lw=1.2, label=col)
    ax.axhline(0, color="gray", ls="--", lw=0.6, alpha=0.6)
    ax.legend(loc="upper left", fontsize=7.5, ncol=2)
    ax.set_title(title)
    ax.set_ylabel("Pearson r")
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    plt.tight_layout()

    return corr_df, fig


def cal_style_beta(df: pd.DataFrame, df_kj, ed, window=20, style_cols=None):
    """
    计算每个风格因子收益率对宽基指数收益率的一元线性回归 Beta 系数。

    :param df:         因子收益率 DataFrame，DatetimeIndex，列为风格因子
    :param df_kj:      宽基指数日收益率，MultiIndex(date, order_book_id)，列含 'daily_return'
    :param ed:         数据截止日期
    :param window:     计算窗口（最近 N 个交易日）
    :param style_cols: 因子名称列表，默认 STYLE_COLS
    :return:           Beta 矩阵 DataFrame（行=因子，列=指数代码）
    """
    if style_cols is None:
        style_cols = STYLE_COLS

    kj_ids = list(INDEX_NAME_MAP.keys())

    # ---------- 1. 增量更新宽基指数收益率 ----------
    latest_date = df_kj.index.get_level_values("date").max()
    if ed > latest_date:
        temp = get_price(kj_ids, start_date=latest_date, end_date=ed,
                         frequency="1d", fields="close", adjust_type="pre",
                         skip_suspended=False, expect_df=True, market="cn")
        temp_rt = temp.groupby("order_book_id")["close"].pct_change().dropna()
        temp_rt = temp_rt.to_frame("daily_return")
        df_kj = pd.concat([df_kj, temp_rt]).sort_index()
        df_kj.to_pickle(f"{KJDIR}/宽基指数日收益率_2601_2607.pkl")

    # ---------- 2. 宽基收益率透视：行=日期，列=指数 ----------
    mkt = df_kj["daily_return"].unstack("order_book_id")
    available = [c for c in kj_ids if c in mkt.columns]
    mkt = mkt[available]

    # ---------- 3. 对齐日期 ----------
    style_data = df[style_cols].iloc[-window:]
    common = style_data.index.intersection(mkt.index)
    X = mkt.loc[common].values          # (T, n_idx)
    Y = style_data.loc[common].values   # (T, n_style)

    # ---------- 4. 向量化一元 Beta = Cov(X, Y) / Var(X) ----------
    Xc = X - X.mean(axis=0)
    Yc = Y - Y.mean(axis=0)
    cov_xy = Xc.T @ Yc                  # (n_idx, n_style)
    var_x = (Xc ** 2).sum(axis=0)       # (n_idx,)
    betas = cov_xy / var_x[:, None]     # (n_idx, n_style)

    # ---------- 5. 整理输出 ----------
    translate_cols = [INDEX_NAME_MAP.get(c, c) for c in available]
    result = pd.DataFrame(betas.T, index=style_cols, columns=translate_cols)
    result.index.name = "style_factor"
    result.columns.name = "index_code"
    return result


@st.fragment
def corr_beta_section(df_view, style_cols, ed, kj_dir):
    """fragment：corr_window 变化时仅重跑此函数，其余页面不动"""
    st.divider()
    st.subheader("风格因子相关性与 Beta 分析")

    corr_window = st.number_input(
        "计算窗口（交易日）", min_value=5, max_value=252, value=60, step=5,
        key="corr_beta_window",
    )

    _, _, corr_fig = cal_style_corr(df_view[style_cols], window=corr_window)

    kj_path = os.path.join(kj_dir, "宽基指数日收益率_2601_2607.pkl")
    beta_tbl = None
    if os.path.exists(kj_path):
        df_kj = pd.read_pickle(kj_path)
        beta_tbl = cal_style_beta(df_view[style_cols], df_kj, ed, window=corr_window)

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("##### 因子收益率相关性")
        st.caption("上三角 Pearson / 下三角 Spearman")
        st.pyplot(corr_fig)
        plt.close(corr_fig)
    with col_r:
        if beta_tbl is not None:
            st.markdown("##### 风格因子对宽基指数的 Beta 系数")
            styled_beta = beta_tbl.style.format("{:.3f}")
            styled_beta = styled_beta.bar(align=0, color=["#5fba7d", "#d65f5f"])
            html_beta = styled_beta.set_table_styles([
                {"selector": "td, th",
                 "props": [("padding", "5px 10px"), ("text-align", "right"), ("white-space", "nowrap")]},
                {"selector": "th",
                 "props": [("text-align", "left"), ("font-weight", "bold")]},
            ]).to_html()
            st.markdown(f"""<div style="overflow-x:auto; width:100%;">{html_beta}</div>""", unsafe_allow_html=True)
        else:
            st.warning("宽基指数收益率数据缺失，无法计算 Beta")


@st.fragment
def rolling_corr_section(df_view, style_cols, sd, ed):
    """fragment：切换因子时仅重跑此函数，其余页面不动"""
    st.divider()
    st.subheader("因子滚动相关性分析")

    factor2_opts = list(style_cols) + ["所有_20d", "所有_40d", "所有_60d"]

    c1, c2 = st.columns(2)
    with c1:
        factor1 = st.selectbox("因子 1", style_cols, key="rc_factor1")
    with c2:
        factor2 = st.selectbox("因子 2", factor2_opts, index=min(1, len(style_cols) - 1),
                               key="rc_factor2")

    _, fig = cal_rolling_corr(df_view, factor1, factor2, sd, ed)
    st.pyplot(fig)
    plt.close(fig)


if __name__ == "__main__":
    srcdir = "E:/SJTU/intern/gtht/barra/data_base/fac_ret/whole_mkt"
    kjdir = "E:/SJTU/intern/gtht/barra/data_base/index"
    df = pd.read_pickle(f"{srcdir}/factor_returns_20_2603.pkl") 
    pearson, spearman, fig =cal_style_corr(df, window=20, annot=True, fmt='.2f')
    #plt.show()
    df_kj = pd.read_pickle(f"{kjdir}/宽基指数日收益率_2601_2607.pkl")
    betas, df_check = cal_style_beta(df, df_kj, ed=pd.Timestamp("2026-07-17"), window=20)
    df_check.to_excel(f"{kjdir}/style_beta_check_2607.xlsx")
    print(betas)
