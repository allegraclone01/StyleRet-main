import os
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import Normalize
from matplotlib.ticker import FuncFormatter
import pickle
from rqdatac import *
import pickle
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ACPDIR = os.path.join(BASE_DIR, "data_base", "index_component_日频")
ARTDIR = os.path.join(BASE_DIR, "data_base", "stk_ret")
INDDIR = os.path.join(BASE_DIR, "data_base", "industry_component_日频")
MCPDIR = os.path.join(BASE_DIR, "data_base", "stk_mcp")

_UNIVERSE_INDEX = {"沪深300": "000300.XSHG", "中证500": "000905.XSHG", "中证1000": "000852.XSHG", "上证50": "000016.XSHG","中证2000":"932000.INDX"}

# ── 简易 parquet 缓存：同一文件整个会话只读一次 ──
_pq = {}
def _r(path):
    if path not in _pq:
        _pq[path] = pd.read_parquet(path) if os.path.exists(path) else pd.DataFrame()
    return _pq[path]

def _append_parquet(new_df, base_dir):
    """将 DataFrame(rows=dates, cols=stocks) 按季度追加到 base_dir 下的 parquet"""
    for q, df_q in new_df.groupby(new_df.index.to_period("Q")):
        path = os.path.join(base_dir, f"{q}.parquet")
        if os.path.exists(path):
            old = pd.read_parquet(path)
            n_old = len(old)
            all_cols = old.columns.union(df_q.columns)
            df_q = pd.concat([old, df_q], axis=0).reindex(columns=all_cols)
            df_q = df_q[~df_q.index.duplicated(keep="last")].sort_index()
            n_new = len(df_q)
            print(f"更新{n_new - n_old}条数据至 {path}（原{n_old}条）")
        df_q.to_parquet(path, engine="pyarrow", compression="zstd")
        _pq.pop(path, None)  # 写入后清除缓存，保证下次读取拿到最新数据


def update_Aret(end):
    """增量更新成分股、收益率、行业、市值四类数据至 end 日期"""
    # ① 成分股：检查增量
    df = pd.read_pickle(os.path.join(ACPDIR, "866011.RI_19_26D_dict.pkl"))
    df_2000 = pd.read_pickle(os.path.join(ACPDIR, "932000.INDX_20_26D_dict.pkl"))

    dates = sorted(df.keys())
    md = dates[-1]
    end_ts = pd.Timestamp(end)
    if end_ts <= md:
        return

    temp = index_weights("866011.RI", start_date=md, end_date=end, market="cn")
    temp_2000 = index_weights_ex("932000.INDX", start_date=md, end_date=end, market='cn')

    new_dates = sorted(d for d in temp.index.get_level_values(0).unique() if d > md)
    if not new_dates:
        return

    ret_rows, ind_rows, value_rows = {}, {}, {}
    for dt in new_dates:
        df[dt] = temp.loc[dt]["weight"]
        df_2000[dt] = temp_2000.loc[dt]["weight"]

        stk = df[dt].index.tolist()
        stk_fb = [s for s in stk if not s.endswith(".BJSE")]

        # ② 收益率
        ret_rows[dt] = get_price_change_rate(
            stk_fb, start_date=dt, end_date=dt, expect_df=True, market="cn").T[dt]

        # ③ 行业
        ind_rows[dt] = get_instrument_industry(
            stk, source="citics_2019", level=1, date=dt, market="cn")["first_industry_name"]
        

    # 保存成分股
    with open(os.path.join(ACPDIR, "866011.RI_19_26D_dict.pkl"), "wb") as f:
        pickle.dump(df, f)
    with open(os.path.join(ACPDIR, "932000.INDX_20_26D_dict.pkl"), "wb") as f:
        pickle.dump(df_2000, f)
        

    # 保存收益率 + 行业（按季度 parquet）
    _append_parquet(pd.DataFrame(ret_rows).T.sort_index().astype("float32"), ARTDIR)
    _append_parquet(pd.DataFrame(ind_rows).T.sort_index(), INDDIR)

    # ④ 市值：全量股票批量拉取，再按季度存
    all_stocks = sorted(set().union(*(sr.index for sr in df.values())))
    mcap = get_factor(all_stocks, "a_share_market_val_in_circulation",
                      start_date=md.strftime("%Y-%m-%d"), end_date=end,
                      universe=None, expect_df=True, market="cn")
    mcap_wide = mcap["a_share_market_val_in_circulation"].unstack(level=0).sort_index().astype("float32")
    _append_parquet(mcap_wide, MCPDIR)

# ---- 分域 + 数据加载 ----
def _quarter(date):
    return f"{date.year}Q{(date.month - 1) // 3 + 1}"


def _load_mcap(date):
    """读取单日流通市值 Series(stock→mcap)"""
    q = _quarter(date)
    sr = _r(os.path.join(MCPDIR, f"{q}.parquet")).loc[pd.Timestamp(date)]
    return sr.dropna()


def _get_universe(date, universe, weighted):
    """
    {stock: weight}，weighted=False 时全A 返回 None（不过滤）。
    权重来源：指数 → pickle 成分股权重；行业/全A → 流通市值。
    """
    ts = pd.Timestamp(date)

    # ---- 宽基指数 ----
    if universe in _UNIVERSE_INDEX:
        path = os.path.join(ACPDIR, f"{_UNIVERSE_INDEX[universe]}_20_26D_dict.pkl")
        with open(path, "rb") as f:
            sr = pickle.load(f).get(ts)
        if sr is None:
            return {}
        return {s: (float(sr[s]) if weighted else 1.0)
                for s in sr.index if float(sr[s]) > 0}

    # ---- 行业 ----
    if universe != "全A":
        path = os.path.join(INDDIR, f"{_quarter(ts)}.parquet")
        if not os.path.exists(path):
            return {}
        ind = _r(path)
        if ts not in ind.index:
            return {}
        stocks = ind.loc[ts]
        stocks = stocks[stocks == universe].index.tolist()
        if not weighted:
            return {s: 1.0 for s in stocks}
        mcap = _load_mcap(date)
        mcap = mcap[mcap.index.isin(stocks)]
        return {s: float(mcap[s]) for s in mcap.index if float(mcap[s]) > 0}

    # ---- 全A ----
    if not weighted:
        return None
    mcap = _load_mcap(date)
    return {s: float(mcap[s]) for s in mcap.index if float(mcap[s]) > 0}


def _w_stats(r, w=None):
    """(std, up_pct, down_pct)；w=None 则等权"""
    if w is None:
        up, dn = (r > 0).mean(), (r < 0).mean()
        return r.std(ddof=0), up, dn
    w = np.asarray(w, dtype=float)
    w_sum = w.sum()
    if w_sum == 0:
        return np.nan, np.nan, np.nan
    r_mean = np.average(r, weights=w)
    r_var = np.average((r - r_mean) ** 2, weights=w)
    up = np.average(r > 0, weights=w)
    dn = np.average(r < 0, weights=w)
    return np.sqrt(r_var), up, dn


def _load_ret(date):
    """读取单日全市场收益率 Series(stock→ret)，非交易日抛 KeyError"""
    q = _quarter(date)
    df = _r(os.path.join(ARTDIR, f"{q}.parquet"))
    ts = pd.Timestamp(date)
    if ts not in df.index:
        raise KeyError(f"{date.date()} 不是交易日")
    return df.loc[ts].dropna()


def _load_week_ret(date):
    """
    读取 date 所在 ISO 周的全部交易日收益率，验证 date 是周尾。
    返回 (dates_list, ret_df: rows=dates, cols=stocks)
    """
    ts = pd.Timestamp(date)
    iso = ts.isocalendar()[:2]
    frames, all_dates = [], []
    for offset in [0, -1]:
        qt = ts + pd.DateOffset(months=offset * 3)
        q = _quarter(qt)
        path = os.path.join(ARTDIR, f"{q}.parquet")
        if os.path.exists(path):
            df = _r(path)
            wk = [d for d in df.index if d.isocalendar()[:2] == iso]
            if wk:
                frames.append(df.loc[wk])
                all_dates.extend(wk)
    if not all_dates:
        raise KeyError(f"{date.date()} 所在周无交易日数据")
    all_dates = sorted(set(all_dates))
    if ts != max(all_dates):
        raise ValueError(f"{date.date()} 不是本周最后一个交易日（末交易日为 {max(all_dates).date()}）")
    return all_dates, pd.concat(frames)


def _daily_stats(date, universe="全A", weighted=False):
    """单日截面波动率（等权/加权，全A/指数/行业）"""
    sr = _load_ret(date)
    uni = _get_universe(date, universe, weighted)
    if uni is not None:
        common = [s for s in uni if s in sr.index]
        if not common:
            raise ValueError(f"{universe} 在 {date.date()} 无有效成分股")
        r, w = sr[common].values, [uni[s] for s in common]
    else:
        r, w = sr.values, None
    cs, up, dn = _w_stats(r, w)
    adr = up / dn if dn > 0 else np.nan

    print(f"finish {date}")
    return {"date": pd.Timestamp(date), "cs_vol": cs, "up_pct": up, "down_pct": dn,
            "adr": adr, "log_adr": np.log(adr) if (up > 0 and dn > 0) else np.nan}


def _weekly_stats(date, universe="全A", weighted=False):
    """单周截面波动率（仅周尾可算）"""
    wk_dates, ret_df = _load_week_ret(date)
    # 复合周收益
    r = (1 + ret_df.fillna(0)).prod() - 1
    r = r.dropna()
    uni = _get_universe(date, universe, weighted)
    if uni is not None:
        common = [s for s in uni if s in r.index]
        if not common:
            raise ValueError(f"{universe} 在 {date.date()} 无有效成分股")
        r_vals, w = r[common].values, ([uni[s] for s in common] if weighted else None)
    else:
        r_vals, w = r.values, None
    cs, up, dn = _w_stats(r_vals, w)
    adr = up / dn if dn > 0 else np.nan
    n = len(wk_dates)
    return {"date": pd.Timestamp(date), "cs_vol": cs, "up_pct": up, "down_pct": dn,
            "adr": adr, "log_adr": np.log(adr) if (up > 0 and dn > 0) else np.nan,
            "n_days": n, "short_week": n < 5}


def _trading_days(start, end):
    """返回指定日期区间内的交易日列表"""
    dates = pd.read_pickle("trading_dates.pkl")
    cal_p = [pd.Timestamp(d) for d in dates]
    cal = [d for d in cal_p if (start is None or d >= pd.Timestamp(start)) and (end is None or d <= pd.Timestamp(end))]
    return cal


def calc_cs_stats(freq="daily", start=None, end=None, universe="全A", weighted=False):
    """
    计算截面波动率 + 涨跌比 + ADR（向量化：批量加载 + 矩阵运算）。

    Parameters
    ----------
    freq  : "daily" | "weekly"
    start, end : str/Timestamp/None  日期区间，None=全量
    universe : "全A" / "沪深300" / "中证500" / "中证1000" / "上证50" / 行业名
    weighted : bool

    Returns
    -------
    pd.DataFrame  index=date
    """
    dates = sorted(_trading_days(start, end))
    if not dates:
        raise ValueError("日期范围内无交易日")
    dates_ts = pd.DatetimeIndex(dates)

    # ── 1. 批量加载全部收益率 → T×N 矩阵（_r 缓存：每季度文件只读一次）──
    quarters = sorted({f"{d.year}Q{(d.month - 1) // 3 + 1}" for d in dates_ts})
    ret_daily = pd.concat([_r(os.path.join(ARTDIR, f"{q}.parquet")) for q in quarters])
    ret_daily = ret_daily.sort_index()
    ret_daily = ret_daily[ret_daily.index.isin(dates_ts)]

    # ── 2. 日频直接用；周频复合 ──
    if freq == "weekly":
        week = ret_daily.index.to_period("W")
        n_days = ret_daily.notna().any(axis=1).groupby(week).sum()
        ret = (1 + ret_daily.fillna(0)).groupby(week).prod() - 1

        # 取每周最后一个交易日作为索引
        last_day = ret_daily.index.to_series().groupby(week).last()
        ret.index = pd.DatetimeIndex(last_day.values)
        n_days.index = ret.index

        # 只保留 dates_ts 中的周尾
        ret = ret.loc[ret.index.intersection(dates_ts)]
        n_days = n_days.loc[ret.index]
    else:
        ret = ret_daily.loc[ret_daily.index.intersection(dates_ts)]

    if ret.empty:
        raise ValueError("无有效收益率数据")

    # ── 3. 构建权重矩阵（0=不在域内），等权全A 则直接用 ret ──
    if universe == "全A" and not weighted:
        # 等权全市场：三行 pandas，一次算完所有日期！
        std = ret.std(axis=1, ddof=0).values
        up  = (ret > 0).mean(axis=1).values
        dn  = (ret < 0).mean(axis=1).values
        avg_ret = ret.mean(axis=1).values                     # 等权平均收益
    else:
        # 逐日收集权重（_get_universe 中 _r 已消除 I/O 瓶颈）
        w_rows = {}
        for d in ret.index:
            uni = _get_universe(d, universe, weighted)
            if uni is not None:
                w_rows[d] = uni
        W = pd.DataFrame(w_rows).T.reindex(columns=ret.columns).fillna(0.0)
        W = W.loc[ret.index]

        # 矩阵运算：加权标准差 = sqrt(Σ w·(r - r̄)² / Σ w)
        common = ret.columns.intersection(W.columns)
        R = ret[common].values.astype(np.float64)
        Wv = W[common].values.astype(np.float64)
        # 有效收益mask
        mask = ~np.isnan(R)
        Wv = Wv * mask
        # Wv = np.where(np.isnan(R), 0.0, Wv)               # 无收益 → 权重置 0
        w_sum = Wv.sum(axis=1)
        valid = w_sum > 0

        std = np.full(len(ret), np.nan)
        up  = np.full(len(ret), np.nan)
        dn  = np.full(len(ret), np.nan)
        avg_ret = np.full(len(ret), np.nan)                  # 加权平均收益
        if valid.any():
            Rv, Wvv, ws = R[valid], Wv[valid], w_sum[valid]
            R_clean = np.nan_to_num(Rv, nan=0.0)

            mean = (Wvv * R_clean).sum(axis=1) / ws
            avg_ret[valid] = mean                             # 保存加权平均收益
            diff = R_clean - mean[:, None]
            std[valid] = np.sqrt((Wvv * diff ** 2).sum(axis=1) / ws)
            up[valid]  = (Wvv * (Rv > 0)).sum(axis=1) / ws
            dn[valid]  = (Wvv * (Rv < 0)).sum(axis=1) / ws

    # ── 4. 组装结果 ──
    adr = np.where(dn > 0, up / dn, np.nan)
    result = pd.DataFrame({
        "cs_vol": std, "up_pct": up, "down_pct": dn,
        "adr": adr, "log_adr": np.where((up > 0) & (dn > 0), np.log(adr), np.nan),
        "ret": avg_ret,
    }, index=ret.index)
    result.index.name = "date"

    if freq == "weekly":
        result["n_days"] = n_days.values
        result["short_week"] = n_days.values < 5

    return result


def calc_vol_percentile(cs_vol, window=252):
    """
    滚动百分位排名：cs_vol[t] 在过去 window 个值中的分位数（含自身）。

    Parameters
    ----------
    cs_vol : pd.Series  date → cs_vol
    window : int  日频 252（≈1年），周频 52

    Returns
    -------
    pd.Series  date → pct_rank (0~1)
    """
    vals = cs_vol.values
    n = len(vals)
    pct = np.full(n, np.nan)
    for i in range(window - 1, n):
        hist = vals[i - window + 1 : i + 1]
        pct[i] = (hist <= vals[i]).mean()
    return pd.Series(pct, index=cs_vol.index, name="pct_rank")


def _fmt_dates(ax):
    """自适应密集日期标注：~18 刻度，短区间 %m-%d / 中区间 %Y-%m / 长区间 YYYYQn"""
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


def plot_vol_series(stats, pct_rank, freq="daily", ma_windows=None, full_vol=None):
    """
    三张独立图片：波动率（颜色编码）、历史排位、ADR。

    Parameters
    ----------
    ma_windows : list[int] or None  MA 窗口列表，如 [5, 20, 60]
    full_vol   : pd.Series or None  全样本波动率（用于 MA 计算），None 则用 stats

    Returns
    -------
    dict  {"vol": fig, "rank": fig, "adr": fig}
    """
    x = stats.index
    common = dict(figsize=(15, 4), constrained_layout=True)

    # ===== Fig 1: 绝对波动（颜色编码排位）=====
    fig1, ax1 = plt.subplots(**common)
    cmap = plt.cm.RdYlGn_r
    norm = Normalize(0, 1)
    y = stats["cs_vol"].values
    c = pct_rank.values

    for i in range(len(x) - 1):
        if not np.isnan(c[i]):
            ax1.fill_between(x[i:i+2], y[i:i+2], 0,
                             color=cmap(norm(c[i])), alpha=0.35, lw=0)
    ax1.plot(x, y, color="#333333", lw=0.9)

    # MA 均线（在全样本上计算，再截取展示区间，避免短区间 MA 大量空缺）
    if ma_windows:
        src = full_vol if full_vol is not None else stats["cs_vol"]
        colors = ["#27d3d6", "#1f77b4", "#662ca0"]
        for w, color in zip(ma_windows, colors):
            ma = src.rolling(w).mean().reindex(stats.index)
            ax1.plot(x, ma.values, color=color, lw=1.6, label=f"MA{w}")
        ax1.legend(fontsize=8, loc="upper left")

    ax1.set_ylabel("CS Volatility")
    # 右轴：累计净值（从 ret 实时计算，展示区间起点归一为 1）
    if "ret" in stats.columns:
        _nav = (1 + stats["ret"].fillna(0)).cumprod()
        _nav = _nav / _nav.iloc[0]
        ax1r = ax1.twinx()
        ax1r.plot(x, _nav.values, color="#ff0400", lw=1.8, alpha=0.8)
        ax1r.set_ylabel("NAV", color="#ff0400", fontsize=9)
        ax1r.tick_params(axis="y", labelcolor="#ff0400")
    ax1.set_title(f"Market Cross-Sectional Volatility ({freq})")
    ax1.set_ylim(0, None)
    ax1.grid(alpha=0.25)
    _fmt_dates(ax1)

    # ===== Fig 2: 历史排位 =====
    fig2, ax2 = plt.subplots(**common)
    ax2.fill_between(x, pct_rank.values, alpha=0.12, color="#ff7f0e")
    ax2.plot(x, pct_rank.values, color="#ff7f0e", lw=1.2)
    for level, ls in [(0.25, ":"), (0.50, "--"), (0.75, ":")]:
        ax2.axhline(level, color="gray", ls=ls, lw=0.8)
    ax2.set_ylabel("Percentile Rank")
    ax2.set_title(f"Volatility Percentile Rank ({freq})")
    ax2.set_ylim(0, 1)
    ax2.grid(alpha=0.25)
    _fmt_dates(ax2)

    # ===== Fig 3: log ADR =====
    fig3, ax3 = plt.subplots(**common)
    ax3.fill_between(x, stats["log_adr"], alpha=0.12, color="#2ca02c")
    ax3.plot(x, stats["log_adr"], color="#2ca02c", lw=1.2)
    ax3.axhline(0, color="gray", ls="--", lw=0.8)
    ax3.set_ylabel("log ADR")
    # 右轴：累计净值（从 ret 实时计算，展示区间起点归一为 1）
    if "ret" in stats.columns:
        _nav = (1 + stats["ret"].fillna(0)).cumprod()
        _nav = _nav / _nav.iloc[0]
        ax3r = ax3.twinx()
        ax3r.plot(x, _nav.values, color="#ff0400", lw=1.8, alpha=0.8)
        ax3r.set_ylabel("NAV", color="#ff0400", fontsize=9)
        ax3r.tick_params(axis="y", labelcolor="#ff0400")
    ax3.set_title(f"Advance/Decline Ratio ({freq})")
    ax3.grid(alpha=0.25)
    _fmt_dates(ax3)

    return {"vol": fig1, "rank": fig2, "adr": fig3}


def main(freq="daily", start=None, end=None, universe="全A", weighted=False, lookback=None):
    """
    一键计算 + 画图。排位基于 [start-lookback, end] 回溯窗口，避免全样本重算。

    Returns
    -------
    figs : dict  {"vol": fig, "rank": fig, "adr": fig}
    df   : pd.DataFrame  统计结果
    """
    if lookback is None:
        lookback = 252 if freq == "daily" else 52

    # 排位需要回溯窗口：起始日往前推足够长
    rank_start = None
    if start is not None:
        all_dates = _trading_days(None, None)
        idx = [i for i, d in enumerate(all_dates) if d >= pd.Timestamp(start)]
        if idx:
            rank_start = all_dates[max(0, idx[0] - 252)]

    df_full = calc_cs_stats(freq, rank_start, end, universe, weighted)
    pct_full = calc_vol_percentile(df_full["cs_vol"], window=lookback)

    # 截取用户指定的展示区间
    df = df_full.loc[pd.Timestamp(start or df_full.index[0]):pd.Timestamp(end or df_full.index[-1])]
    pct = pct_full.loc[df.index]

    figs = plot_vol_series(df, pct, freq, full_vol=df_full["cs_vol"])
    return figs, df_full


if __name__ == "__main__":
    desdir = "E:/SJTU/intern/gtht/barra/data_base/volatility/output"

    # u = "全A"
    # f = "weekly"
    # w = True
    # _tag = f"{'w' if w else 'ew'}"
    # print((f"计算截面波动率：freq={f}, universe={u}, weighted={w}"))
    # figs, stats = main(freq=f, start="2020-01-01", universe=u, weighted=w)
    # #存储
    # stats.to_pickle(f"{desdir}/{f}_{u}_{_tag}.pkl")
    # print(stats.tail())
    # plt.show()


    #批量计算
    universes = [] + list(_UNIVERSE_INDEX.keys()) + ['石油石化', '煤炭', '有色金属', '电力及公用事业', '钢铁', '基础化工', '建筑', '建材', '轻工制造',
       '机械', '电力设备及新能源', '国防军工', '汽车', '商贸零售', '消费者服务', '家电', '纺织服装',
       '医药', '食品饮料', '农林牧渔', '银行', '非银行金融', '房地产', '综合金融', '交通运输', '电子',
       '通信', '计算机', '传媒', '综合']
    # "全A"
    
    for u in universes:
        for f in ["daily", "weekly"]:
            for w in [False, True]:
                _tag = f"{'w' if w else 'ew'}"
                print(f"计算：universe={u}, freq={f}, weighted={w}")

                try:
                    figs, stats = main(freq=f, start="2020-01-01", universe=str(u), weighted=w)
                except Exception as e:
                    print(f"  × 跳过：{e}")
                    continue

                stats.to_pickle(f"{desdir}/{f}_{u}_{_tag}.pkl")
                print(stats.head())

                for name, fig in figs.items():
                    fig.savefig(f"{desdir}/plots/{f}_{u}_{_tag}_{name}.png", dpi=150, bbox_inches="tight")
                    plt.close(fig)

