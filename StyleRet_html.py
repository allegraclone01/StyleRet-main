import os, re
from datetime import timedelta
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import streamlit as st
from helpfunc_basis import *
from helpfunc_barra import *
from helpfunc_vol import *
from helpfunc_vol import _UNIVERSE_INDEX, INDUSTRY_UNIVERSES
from rqdatac import *

from matplotlib import font_manager
font_path = "fonts/SimHei.ttf"
font_manager.fontManager.addfont(font_path)
prop = font_manager.FontProperties(fname=font_path)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_FILE = os.path.join(BASE_DIR, "data_base", "fac_ret", "whole_mkt", "factor_returns_10_2608.pkl")
BASIS_DIR = os.path.join(BASE_DIR, "data_base", "basis","index_future_basis_data.pkl")
KJDIR = os.path.join(BASE_DIR, "data_base", "index")

plt.rcParams["axes.unicode_minus"] = False

def _setup_chinese_font():
    import matplotlib.font_manager as fm
    candidates = ["Microsoft YaHei", "SimHei", "WenQuanYi Micro Hei", "Noto Sans SC", "Noto Sans CJK SC"]
    for f in fm.fontManager.ttflist:
        if f.name in candidates:
            plt.rcParams["font.sans-serif"] = [f.name]
            return
    try:
        import urllib.request
        fp = "/tmp/NotoSansSC-Regular.otf"
        if not os.path.exists(fp):
            urllib.request.urlretrieve(
                "https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTF/SimplifiedChinese/NotoSansSC-Regular.otf",
                fp
            )
        fm.fontManager.addfont(fp)
        plt.rcParams["font.sans-serif"] = [fm.FontProperties(fname=fp).get_name()]
    except:
        pass

_setup_chinese_font()

RQ_OK = False
try:
    import rqdatac
    rqdatac.init(
    username = "license",
    password = "ZZ-u7ZWosqrntc3VY3TJzJLPsb-A0o4zehYoiNpDvIBXiwvRIOUmFe7medtMhwu4qiaNxqFSc6ONdGcGeVYgUVd-w5QKScPkmzBEmYVEt94lz9sQZoHwdtQXWWRGGrJqtr7ehiQACydlPS7RcPBfJrpyeTJFsGF1E1guZbpLnvU=XouX9YSi7Pcyo0rSLCMydvHs3nrVq6Rwjda-jI9H_gfGlp53ot0ZnIA6g-ZtvwPDAb62K38pHIqYYyTAyER7FBtZ5HumXzOrWW42LHpUn5-vbnLMxiwbimJ9ns41CaMbjpFEgNcfO52l5wiqDqFCkZNy_OKSDjepfa9GxHsLZZE="
)
    RQ_OK = True
except Exception as _rq_err:
    st.sidebar.error(f"rqdatac 初始化失败: {_rq_err}")


def load_cache():
    if not os.path.exists(CACHE_FILE):
        return pd.DataFrame()
    try:
        df = pd.read_pickle(CACHE_FILE)
        if isinstance(df, pd.DataFrame):
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            return df.sort_index()
    except:
        pass
    return pd.DataFrame()


def save_cache(df):
    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    pd.to_pickle(df, CACHE_FILE)


def fetch_from_api(start, end):
    from rqdatac import get_factor_return
    raw = get_factor_return(start, end, factors=None, universe="whole_market",
                            method="implicit", industry_mapping="sws_2021",
                            model="v1", market="cn")
    if raw is None or len(raw) == 0:
        return pd.DataFrame()
    df = pd.DataFrame(raw) if not isinstance(raw, pd.DataFrame) else raw
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.sort_index()


def get_data(sd, ed):
    _d = []
    df = load_cache()
    _d.append(f"缓存区间: {df.index.min().date()} ~ {df.index.max().date()}" if not df.empty else "缓存为空")

    if not df.empty:
        c_min, c_max = df.index.min(), df.index.max()
        _d.append(f"缓存区间: {c_min.date()} ~ {c_max.date()}")
        _d.append(f"sd >= c_min? {sd >= c_min}  |  ed <= c_max? {ed <= c_max}")
        if (sd >= c_min) and (ed <= c_max):
            _d.append("缓存完全覆盖 → 直接返回")
            return df, _d
        _d.append("缓存不足 → 需增量更新")
    else:
        _d.append("缓存为空")

    _d.append(f"RQ_OK (API可用)? {RQ_OK}")
    if not RQ_OK:
        _d.append("API不可用 → 返回现有缓存")
        return df if not df.empty else pd.DataFrame(), _d

    if not df.empty:
        if ed > c_max:
            fetch_start = c_max 
            fetch_end = ed 
        elif sd < c_min:
            fetch_start = sd 
            fetch_end = c_min 
        else:
            return df, _d
    else:
        fetch_start = sd 
        fetch_end = ed 
    _d.append(f"计划拉取: {fetch_start.date()} ~ {fetch_end.date()}")
    st.info(f"📡 正在提取新数据 ({fetch_start.date()} ~ {fetch_end.date()})…")
    try:
        new = fetch_from_api(fetch_start.strftime("%Y%m%d"), fetch_end.strftime("%Y%m%d"))
    except Exception as e:
        _d.append(f"API 异常: {e}")
        return df, _d

    _d.append(f"API 返回行数: {len(new)}")
    if new.empty:
        _d.append("API 返回空")
        return df, _d

    if not df.empty:
        df = pd.concat([df, new]).drop_duplicates(keep="last").sort_index()
        _d.append(f"合并后行数: {len(df)}")
    else:
        df = new.sort_index()
        _d.append(f"首次拉取行数: {len(df)}")
    try:
        save_cache(df)
        _d.append("缓存已保存 ✅")
    except Exception as e:
        _d.append(f"缓存写入失败: {e}")
    return df, _d

mapping = {"风格因子": "style", "行业因子": "industry"}
st.set_page_config(page_title="Barra 因子净值", layout="wide")
st.markdown("""
<style>
div[data-testid="metric-container"] label { font-size: 0.7rem !important; white-space: nowrap !important; }
div[data-testid="metric-container"] div { font-size: 0.8rem !important; }
</style>
""", unsafe_allow_html=True)
st.title("行情面板")
st.sidebar.header("配置")
st.session_state.sd = st.sidebar.date_input("起始", pd.Timestamp("2020-01-02"), max_value=pd.Timestamp("2036-03-25"))
st.session_state.ed = st.sidebar.date_input("结束", last_trading_day(), max_value=pd.Timestamp("2036-03-25"))
mode = st.sidebar.radio("模式", ["Barra大类综合", "Barra单因子详细", "基差成本监控", "全市场波动", "超额回撤复盘"])

sd = pd.Timestamp(st.session_state.sd)
ed = pd.Timestamp(st.session_state.ed)
with st.spinner("加载数据中..."):
    df_full, debug_log = get_data(sd, ed)
if df_full.empty:
    st.error("无可用数据")
    st.stop()

# ---------- 基差数据：读取 + 增量更新 ----------
def _load_basis():
    if os.path.exists(BASIS_DIR):
        try:
            bd = pd.read_pickle(BASIS_DIR)
            if isinstance(bd, pd.DataFrame) and not bd.empty:
                for _cn in ["date", "listed_date", "maturity_date"]:
                    if _cn in bd.columns:
                        bd[_cn] = pd.to_datetime(bd[_cn])
                return bd
        except:
            pass
    return pd.DataFrame()

df_basis = _load_basis()
if mode == "基差成本监控":
    if not df_basis.empty:
        b_max = df_basis["date"].max()
    else:
        b_max = sd - pd.Timedelta(days=1)
    if ed > b_max:
        try:
            _new = add_basis_data(b_max, ed)
            if isinstance(_new, pd.DataFrame) and not _new.empty:
                for _cn in ["date", "listed_date", "maturity_date"]:
                    if _cn in _new.columns:
                        _new[_cn] = pd.to_datetime(_new[_cn])
                # 确保 _new 也把 order_book_id 当列处理
                if _new.index.name == "order_book_id":
                    _new = _new.reset_index()
                df_basis = pd.concat([df_basis.reset_index(), _new], axis=0)
                df_basis = (df_basis.drop_duplicates(keep="last")
                                    .sort_values(["order_book_id", "date"])
                                    .set_index("order_book_id"))
                print(df_basis.tail())
                os.makedirs(os.path.dirname(BASIS_DIR), exist_ok=True)
                pd.to_pickle(df_basis, BASIS_DIR)
                print(f"基差数据更新完成: {df_basis['date'].max().date()}")
        except Exception as _be:
            st.warning(f"基差数据更新失败: {_be}")

df_view = df_full[(df_full.index >= sd) & (df_full.index <= ed)]
if df_view.empty:
    st.error("区间无数据")
    st.stop()

cols = [c for c in df_full.columns if str(c).lower() != "comovement"]
style_cols = [c for c in cols if not re.search(r"[\u4e00-\u9fff]", str(c))]
industry_cols = [c for c in cols if re.search(r"[\u4e00-\u9fff]", str(c))]

if mode != "基差成本监控":
    c1, c2, c3 = st.columns(3)
    c1.metric("开始时间", f"{df_view.index.min().date()}")
    c2.metric("结束时间", f"{df_view.index.max().date()}")
    c3.metric("交易日", len(df_view))


if mode == "Barra大类综合":
    cat = st.radio("类别", ["风格因子", "行业因子"], horizontal=True)
    target = style_cols if cat == "风格因子" else industry_cols
    if not target:
        st.stop()
    nav = (df_view[target] + 1).cumprod()
    nav = nav / nav.iloc[0]
    order = nav.iloc[-1].sort_values(ascending=False).index

    if cat == "风格因子":
        st.subheader("barra波动率")
        render_style_volatility_section(df_view, style_cols, sd, ed)
        picked_style_vol = st.multiselect(
            "查看多窗口时序波动率因子（最多2个）",
            style_cols,
            default=[],
            max_selections=2,
            key="style_vol_multi_factors",
        )
        if picked_style_vol:
            plot_factor_volatility_multiwindow(df_view, picked_style_vol, start=sd, end=ed)

    st.subheader("barra收益率")

    today_idx = nav.index
    today = today_idx[-1]

    # ⭐ week_start: 本周之前一周的最后一个交易日
    monday_nat = today - pd.Timedelta(days=today.weekday())
    prev_cand = today_idx[today_idx < monday_nat]
    week_start = prev_cand[-1] if len(prev_cand) > 0 else today

    # ⭐ month_start: 当月第一天的前一个交易日
    first_of_month = pd.Timestamp(year=today.year, month=today.month, day=1)
    prev_cand = today_idx[today_idx < first_of_month]
    month_start = prev_cand[-1] if len(prev_cand) > 0 else today
    curr_month = pd.Timestamp(year=today.year, month=today.month, day=1)
    curr_cand = today_idx[today_idx >= curr_month]
    curr_month_start = curr_cand[0] if len(curr_cand) > 0 else today

    latest_nav = nav.iloc[-1]
    if len(nav) >= 2:
        ret_1d = (nav.iloc[-1] / nav.iloc[-2] - 1)
    else:
        ret_1d = pd.Series(np.nan, index=nav.columns)

    nav_week = nav.loc[week_start:today]
    ret_1w = nav_week.iloc[-1] / nav_week.iloc[0] - 1

    nav_month = nav.loc[month_start:today]
    ret_1m = nav_month.iloc[-1] / nav_month.iloc[0] - 1
    cum = nav_month / nav_month.iloc[0]
    dd = (cum - cum.cummax()) / cum.cummax()
    dd_max = dd.min()
    month_ret = df_view.loc[curr_month_start:today, target]
    vol = month_ret.std()

    tbl = pd.DataFrame({
        "最新净值": latest_nav,
        "1日收益": ret_1d * 100,
        "本周累计": ret_1w * 100,
        "本月累计": ret_1m * 100,
        "本月最大回撤": dd_max * 100,
        "本月波动率": vol * 100,
    }).round(4)
    tbl = tbl.reindex(order)

    fig, ax = plt.subplots(figsize=(12, 6))
    for c in order:
        ax.plot(nav.index, nav[c], label=str(c), lw=1.2)
    ax.axhline(1, color="gray", ls="--", lw=0.6, alpha=0.6)
    ax.set_title(f"{mapping[cat]} nav")
    ax.legend(loc="upper left", bbox_to_anchor=(-0.15, 1), fontsize=7.5, ncol=1)
    plt.tight_layout(rect=[0.15, 0, 1, 1])
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    st.pyplot(fig)
    plt.close(fig)

    bar_cols = ["1日收益", "本周累计", "本月累计"]
    styled = tbl.style.format({
        "最新净值": "{:.4f}",
        **{c: "{:.2f}%" for c in bar_cols},
        "本月最大回撤": "{:.2f}%",
        "本月波动率": "{:.2f}%",
    }, na_rep="-")
    for c in bar_cols:
        styled = styled.bar(subset=[c], align="zero",
                            color=["#5fba7d","#d65f5f"], vmin=None, vmax=None)
    html = styled.set_table_styles([
        {"selector": "td, th", "props": [("padding", "5px 10px"), ("text-align", "right"), ("white-space", "nowrap")]},
        {"selector": "th", "props": [("text-align", "left"), ("font-weight", "bold")]},
    ]).to_html()
    st.markdown(f"""<div style="overflow-x:auto; width:100%;">{html}</div>""", unsafe_allow_html=True)

    if cat == "风格因子":
        corr_beta_section(df_view, style_cols, ed, KJDIR)
        rolling_corr_section(df_view, style_cols, sd, ed)

elif mode == "基差成本监控":
    if df_basis.empty:
        st.error("基差数据为空，请检查数据文件")
        st.stop()

    # ====== cal_fhds：分红点数调整（@st.cache_data 缓存，切换列不重算）======
    _prev_date = pd.Timestamp(ed)

    @st.cache_data(ttl=3600, show_spinner="正在计算分红调整…")
    def _compute_fhds(_dt_str):
        """缓存：对指定日期计算 cal_fhds + adj 指标，并入前日对比；返回 (_df, _detail_df)"""
        _dt = pd.Timestamp(_dt_str)
        _row = df_basis[df_basis["date"] == _dt].copy()
        if _row.empty:
            return pd.DataFrame(), pd.DataFrame()
        _df, _detail = cal_fhds(_dt, _row, return_detail=True)
        if _df.empty:
            return _df, pd.DataFrame()
        _df["adj_basis"] = _df["basis"] + _df["dividend_point"]
        _df["adj_abs_ratio"] = _df["adj_basis"] / _df["close_index"]
        _df["adj_ana_cost"] = _df["adj_abs_ratio"] / _df["residual_day"] * 365

        # 前一个交易日（不需要 detail，传默认 False）
        _prev_dates = sorted(df_basis[df_basis["date"] < _dt]["date"].unique())
        if _prev_dates:
            _row_p = df_basis[df_basis["date"] == _prev_dates[-1]].copy()
            if not _row_p.empty:
                _df_p = cal_fhds(_prev_dates[-1], _row_p)
                if not _df_p.empty:
                    _df_p["adj_basis"] = _df_p["basis"] + _df_p["dividend_point"]
                    _df_p["adj_abs_ratio"] = _df_p["adj_basis"] / _df_p["close_index"]
                    _df_p["adj_ana_cost"] = _df_p["adj_abs_ratio"] / _df_p["residual_day"] * 365
                    _df["prev_adj_ana_cost"] = _df_p["adj_ana_cost"]
                    _df["prev_adj_basis"] = _df_p["adj_basis"]
                    _df["adj_basis_chg"] = _df["adj_basis"] - _df_p["adj_basis"]
                    _df["adj_basis_chg_ratio"] = _df["adj_basis_chg"] / _df["close_index"]
        return _df, _detail

    _fhds_df, _fhds_detail = _compute_fhds(str(_prev_date))

    if not _fhds_df.empty:
        st.markdown(f"**分红调整后基差（日期: {_prev_date.date()}）**")

        # 列分组：日期列默认隐藏，其余默认显示；st.dataframe 自带横向滚动 + index 固定
        _groups = {
            "日期列": ["date", "listed_date", "maturity_date", "residual_day"],
            "原始基差": ["settlement", "close_index", "basis", "abs_ratio", "ana_cost"],
            "调整后指标": ["dividend_point", "adj_basis", "adj_abs_ratio", "adj_ana_cost"],
            "前日对比": ["prev_adj_ana_cost", "prev_adj_basis", "adj_basis_chg", "adj_basis_chg_ratio"],
        }
        _c1, _c2, _c3, _c4 = st.columns(4)
        _toggles = {}
        with _c1:
            _toggles["日期列"] = st.checkbox("日期列", value=False)
        with _c2:
            _toggles["原始基差"] = st.checkbox("原始基差", value=False)
        with _c3:
            _toggles["调整后指标"] = st.checkbox("调整后指标", value=True)
        with _c4:
            _toggles["前日对比"] = st.checkbox("前日对比", value=True)

        _show_cols = []
        for _gname, _gcols in _groups.items():
            if _toggles[_gname]:
                _show_cols.extend([c for c in _gcols if c in _fhds_df.columns])

        if _show_cols:
            _display = _fhds_df[_show_cols].copy()
            _pct_cols = {"abs_ratio", "ana_cost", "adj_abs_ratio", "adj_ana_cost", "adj_basis_chg_ratio"}
            _fmt = {}
            for _c in _display.columns:
                if _c in _pct_cols:
                    _fmt[_c] = "{:.4f}"
                elif pd.api.types.is_float_dtype(_display[_c]):
                    _fmt[_c] = "{:.4f}"
                elif pd.api.types.is_datetime64_any_dtype(_display[_c]):
                    _fmt[_c] = lambda x: x.strftime("%Y-%m-%d") if pd.notna(x) else "-"
            st.dataframe(
                _display.style.format(_fmt, na_rep="-"),
                use_container_width=True,
                height=400,
            )
        else:
            st.info("请至少勾选一组列")

    # 每个 order_book_id 的存续期是否与 [sd, ed] 有重叠
    def _has_overlap(sub):
        ld = sub["listed_date"].iloc[0]
        md = sub["maturity_date"].iloc[0]
        return not (md < sd or ld > ed)

    valid_ids = []
    for _id, _sub in df_basis.groupby(level="order_book_id", sort=False):
        if _has_overlap(_sub):
            valid_ids.append(str(_id))

    if not valid_ids:
        st.error("当前日期范围内没有可用的合约")
        st.stop()

    valid_ids = sorted(valid_ids)
    ic_ids = [x for x in valid_ids if x.startswith("IC")]
    default_id = ic_ids[-4] if len(ic_ids) >= 4 else (ic_ids[0] if ic_ids else valid_ids[0])
    sel_id = st.selectbox("选择合约 (order_book_id)", valid_ids, index=valid_ids.index(default_id))
    sub = df_basis[df_basis.index.get_level_values("order_book_id").astype(str) == sel_id].copy()
    sub = sub.sort_values("date").reset_index(drop=True)

    # 取该合约在 [sd, ed] 区间内的日期用于绘图
    plot_sub = sub[(sub["date"] >= sd) & (sub["date"] <= ed)].reset_index(drop=True)
    if plot_sub.empty:
        st.warning("所选合约在当前日期区间内没有数据")
        st.stop()

    fig, tbl_html = plot_basis_series(plot_sub, contract_id=sel_id,mode="adjust")
    st.pyplot(fig)
    plt.close(fig)
    st.markdown(tbl_html, unsafe_allow_html=True)


    # ====== 分红除权明细 ======
    if not _fhds_detail.empty:
        st.markdown("---")
        st.markdown(f"**分红除权明细（{_prev_date.date()}）**")
        _c1, _c2 = st.columns(2)
        with _c1:
            _pfx_opts = sorted(_fhds_detail["prefix"].unique())
            _sel_pfx = st.multiselect("指数", _pfx_opts, default=_pfx_opts, key="detail_pfx")
        with _c2:
            _q_opts = sorted(_fhds_detail["quarter"].unique())
            _sel_q = st.multiselect("季度", _q_opts, default=_q_opts, key="detail_q")
        _dtbl = _fhds_detail[
            _fhds_detail["prefix"].isin(_sel_pfx) & _fhds_detail["quarter"].isin(_sel_q)
        ].sort_values(["prefix", "ex_date"]).reset_index(drop=True)
        if not _dtbl.empty:
            st.dataframe(_dtbl.style.format({"dividend": "{:.4f}"}), use_container_width=True, height=300)
            _fig = plot_fhds_detail(_fhds_detail[_fhds_detail["quarter"].isin(_sel_q)], _sel_pfx)
            if _fig is not None:
                st.pyplot(_fig)
                plt.close(_fig)


    

elif mode == "全市场波动":
    CACHE_DIR = os.path.join(os.path.dirname(__file__), "data_base", "volatility", "output")
    os.makedirs(CACHE_DIR, exist_ok=True)

    _freq_key = "vol_freq"
    _universe_key = "vol_universe"
    _weighted_key = "vol_weighted"
    _freq = st.session_state.get(_freq_key, "daily")
    _uni_opts = ["全A", *list(_UNIVERSE_INDEX.keys()), *INDUSTRY_UNIVERSES]
    _universe = st.session_state.get(_universe_key, "全A")
    _weighted = st.session_state.get(_weighted_key, False)

    _tag = f"{'w' if _weighted else 'ew'}"
    _data_key = f"vol_data_{_freq}_{_universe}_{_tag}"       # 全量数据（与日期无关）

    

    #_view_key = f"vol_view_{_freq}_{_universe}_{_tag}_{sd.date()}_{ed.date()}_{_ma_tuple}"

    def _load_cached_vol(_u):
        _path = os.path.join(CACHE_DIR, f"{_freq}_{_u}_{_tag}.pkl")
        _key = f"vol_data_{_freq}_{_u}_{_tag}"
        if _key in st.session_state:
            return st.session_state[_key]
        if os.path.exists(_path):
            _df = pd.read_pickle(_path)
            st.session_state[_key] = _df
            return _df
        return pd.DataFrame()

    def _ensure_cached_vol(_u):
        _path = os.path.join(CACHE_DIR, f"{_freq}_{_u}_{_tag}.pkl")
        _key = f"vol_data_{_freq}_{_u}_{_tag}"
        _cached = _load_cached_vol(_u)
        _calc_start = (_cached.index.max() + pd.Timedelta(days=1)) if not _cached.empty else sd
        if _calc_start <= ed:
            new_df = calc_cs_stats(_freq, start=_calc_start, end=ed, universe=_u, weighted=_weighted)
            _cached = pd.concat([_cached, new_df])
            _cached = _cached[~_cached.index.duplicated(keep="last")].sort_index()
            _cached.to_pickle(_path)
        st.session_state[_key] = _cached
        return _cached

    summary_universes = ["全A", *list(_UNIVERSE_INDEX.keys()), *INDUSTRY_UNIVERSES]

    # ① 先统一更新所有分域缓存
    update_Aret(ed)
    with st.spinner(f"更新全市场波动缓存（{len(summary_universes)}个分域）…"):
        summary_cache = {u: _ensure_cached_vol(u) for u in summary_universes}
    cached = summary_cache[_universe]
    st.session_state[_data_key] = cached

    _summary = build_market_vol_snapshot(summary_cache, freq=_freq, as_of=ed, top_n=10)
    if not _summary.empty:
        st.markdown(f"**{pd.Timestamp(_summary['日期'].max()).date()} 市场波动概览**")
        _summary_num_cols = ["波动率", "历史分位数", "上涨比例", "下跌比例"]
        styled_summary = _summary.style.format({c: "{:.2%}" for c in _summary_num_cols}, na_rep="-")
        styled_summary = styled_summary.bar(subset=["波动率"], color="#5B8FF9")
        styled_summary = styled_summary.bar(subset=["历史分位数"], color="#d65f5f", vmin=0, vmax=1)
        html_summary = styled_summary.set_table_styles([
            {"selector": "td, th", "props": [("padding", "5px 10px"), ("text-align", "right"), ("white-space", "nowrap")]},
            {"selector": "th", "props": [("text-align", "left"), ("font-weight", "bold")]},
        ]).to_html()
        st.markdown(f"""<div style="overflow-x:auto; width:100%;">{html_summary}</div>""", unsafe_allow_html=True)
        st.markdown("---")

    # 用户选择（放在概览表后、子图前）
    _c1, _c2, _c3 = st.columns(3)
    with _c1:
        _freq = st.radio("频率", ["daily", "weekly"], horizontal=True, key=_freq_key)
    with _c2:
        _universe = st.selectbox("分域", _uni_opts, index=_uni_opts.index(_universe) if _universe in _uni_opts else 0, key=_universe_key)
    with _c3:
        _weighted = st.radio("加权", ["等权", "加权"], horizontal=True, index=1 if _weighted else 0, key="vol_weighted_label") == "加权"
        st.session_state[_weighted_key] = _weighted

    _tag = f"{'w' if _weighted else 'ew'}"
    _data_key = f"vol_data_{_freq}_{_universe}_{_tag}"
    # MA 均线选择
    _ma_opts = [5, 20, 60] if _freq == "daily" else [4, 13, 52]
    _ma_windows = st.multiselect("MA", _ma_opts, [], key=f"ma_{_data_key}")
    _ma_tuple = tuple(sorted(_ma_windows))
    _view_key = f"vol_view_{_freq}_{_universe}_{_tag}_{sd.date()}_{ed.date()}_{_ma_tuple}"
    cached = st.session_state.get(_data_key, cached if _universe == st.session_state.get(_universe_key, _universe) else pd.DataFrame())

    # ② 视图（figs + 表格）：命中则跳过画图，否则重新生成
    if _view_key in st.session_state:
        figs, _tbl, csv, _fname = st.session_state[_view_key]
        st.info("📋 配置未变，从内存直接展示")
    else:
        if _data_key not in st.session_state:
            with st.spinner(f"计算 {_universe} {_freq} {_tag} …"):
                cached = _ensure_cached_vol(_universe)
        else:
            cached = st.session_state[_data_key]

        _display = cached.loc[pd.Timestamp(sd):pd.Timestamp(ed)]
        if _display.empty:
            st.warning("所选区间无数据")
            st.stop()

        _lookback = 252 if _freq == "daily" else 52
        _pct = calc_vol_percentile(cached["cs_vol"], window=_lookback)
        _pct_display = _pct.loc[_display.index]

        figs = plot_vol_series(_display, _pct_display, _freq, ma_windows=_ma_windows or None, full_vol=cached["cs_vol"])
        _tbl = _display.reset_index()
        csv = _tbl.to_csv(index=False).encode("utf-8-sig")
        _fname = f"cs_vol_{_freq}_{_universe}_{_tag}_{sd.date()}_{ed.date()}.csv"
        st.session_state[_view_key] = (figs, _tbl, csv, _fname)

    # ── 展示 ──
    for _key, _title in [("vol", "绝对波动"), ("rank", "历史排位"), ("adr", "ADR")]:
        st.markdown(f"**{_title}**")
        st.pyplot(figs[_key])

    st.markdown("---")
    st.markdown("**统计指标**")
    _num_cols = _tbl.select_dtypes(include=[np.number]).columns.tolist()
    st.dataframe(_tbl.style.format({c: "{:.4f}" for c in _num_cols}, na_rep="-"),
                 use_container_width=True, height=400)
    st.download_button("下载 CSV", csv, _fname, "text/csv")

elif mode == "超额回撤复盘":
    from helpfunc_specificr import vol_pipeline, event_vol_analysis, plot_vol_summary, plot_vol_growth

    st.subheader("📉 超额回撤复盘")

    _cache_key = f"excess_dd_{sd.date()}_{ed.date()}"
    if _cache_key not in st.session_state:
        events = pd.read_excel(os.path.join(BASE_DIR, "comb", "outputs",
                                            "Alpha私募超额指数_回撤分析.xlsx"),
                               sheet_name="共性回撤期")
        summary = vol_pipeline(str(sd.date()), str(ed.date()))
        result = event_vol_analysis(events, summary, current_date=str(ed.date()))
        st.session_state[_cache_key] = (summary, result, events)
    summary, result, events = st.session_state[_cache_key]

    st.pyplot(plot_vol_summary(summary, events=events)); plt.close()
    st.pyplot(plot_vol_growth(summary)); plt.close()

    st.markdown("**事件触发前后波动率对比 (%)**")
    num_cols = [c for c in result.columns if c.startswith("前5_") or c.startswith("后5_")]
    s = result.style.format({c: "{:.2f}" for c in num_cols}, na_rep="-")
    s = s.bar(subset=["前5_avg_vol"], color="#5B8FF9")
    s = s.bar(subset=["前5_avg_vol_rank"], color="#d65f5f", vmin=0, vmax=100)
    for c in ["前5_vol_growth", "后5_vol_growth"]:
        s = s.bar(subset=[c], align="zero", color=["#5fba7d", "#d65f5f"])
    html_tbl = s.set_table_styles([
        {"selector": "td, th", "props": [("padding", "5px 10px"), ("text-align", "right"), ("white-space", "nowrap")]},
        {"selector": "th", "props": [("text-align", "left"), ("font-weight", "bold")]},
    ]).to_html()
    st.markdown(f"""<div style="overflow-x:auto; width:100%;">{html_tbl}</div>""", unsafe_allow_html=True)

    # --- 复盘 MD 文档展示 ---
    st.markdown("---")
    st.subheader("📄 复盘文档")
    st.caption("章节导览：一、数据现象  二、市场环境还原  三、原因分析  四、过程时间线  五、影响与启示  六、矛盾检查  七、附录")
    from helpfunc_specificr import render_summary_md
    render_summary_md(os.path.join(BASE_DIR, "comb", "end_input", "summary"))

else:
    sub_cat = st.radio("类型", ["风格因子", "行业因子"], horizontal=True)
    pool = style_cols if sub_cat == "风格因子" else industry_cols
    factor = st.selectbox("因子", pool) if pool else st.stop()
    ret = df_view[factor].dropna()
    nav = (ret + 1).cumprod()
    nav = nav / nav.iloc[0] #净值归1
    ref = df_full[(df_full.index >= sd) & (df_full.index <= ed)][factor].dropna()
    ref_vals = ref.values
    pct = ret.apply(lambda x: float((ref_vals < x).sum()) / len(ref_vals) * 100)

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(nav.index, nav, color="#1f77b4", lw=2, zorder=5)
    ax.axhline(1, color="gray", ls="--", lw=0.6, alpha=0.6)
    y_pad = (nav.max() - nav.min()) * 0.08
    ax.set_ylim(nav.min() - y_pad, nav.max() + y_pad)

    ax2 = ax.twinx()
    ax2.scatter(pct.index, pct, color="#f39c12", s=10, alpha=0.55, label="日收益分位数(%)")
    ax2.axhline(50, color="#f39c12", ls=":", lw=0.8, alpha=0.5)
    ax2.set_ylim(0, 100)
    ax2.set_ylabel("分位数(%)", color="#f39c12", fontsize=11)
    ax2.tick_params(axis="y", labelcolor="#f39c12")

    ax.set_title(f"{factor}", fontsize=14, fontweight="bold")
    ax.set_ylabel("净值", fontsize=11)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="best", framealpha=0.7)
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    st.pyplot(fig)
    plt.close(fig)

    row_labels, ret_list, pct_list, z_list = [], [], [], []
    vol_list, z250_list, z750_list, rank250_list, rank750_list = [], [], [], [], []
    hist_full = ref.values    # 2020-01-02 起的完整日收益
    hist_mean = hist_full.mean()
    hist_std = hist_full.std()

    vol_metrics = cal_factor_volatility(df_view, cols=style_cols) if sub_cat == "风格因子" else None
    if vol_metrics is not None:
        vol_df = vol_metrics["vol"]
        rank250_df = vol_metrics["rank_250d"]
        rank750_df = vol_metrics["rank_750d"]
        z250_df = vol_metrics["z_250d"]
        z750_df = vol_metrics["z_750d"]
    for k in range(1, 21):
        if len(ret) < k:
            break
        idx = ret.index[-k]
        r_val = ret.iloc[-k]
        p_val = float((hist_full < r_val).sum()) / len(hist_full) 
        z_val = (r_val - hist_mean) / hist_std if hist_std > 0 else np.nan
        row_labels.append(f"最近第{k}日 ({idx.strftime('%Y-%m-%d')})")
        ret_list.append(r_val * 100)
        pct_list.append(p_val)
        z_list.append(z_val)
        if sub_cat == "风格因子":
            vol_list.append(vol_df.loc[idx, factor])
            z250_list.append(z250_df.loc[idx, factor])
            z750_list.append(z750_df.loc[idx, factor])
            rank250_list.append(rank250_df.loc[idx, factor])
            rank750_list.append(rank750_df.loc[idx, factor])

    # 近20 / 近60日区间（累计收益）
    for window in [20, 60]:
        if len(nav) < window + 1:
            row_labels.append(f"近{window}日（数据不足）")
            ret_list.append(np.nan)
            pct_list.append(np.nan)
            z_list.append(np.nan)
            vol_list.append(np.nan)
            z250_list.append(np.nan)
            z750_list.append(np.nan)
            rank250_list.append(np.nan)
            rank750_list.append(np.nan)
            continue
        cur_ret = nav.iloc[-1] / nav.iloc[-(window + 1)] - 1

        # 以 ref 对应的净值序列做同窗口滚动累计收益，作为历史参照分布
        ref_nav = (ref + 1).cumprod()
        if len(ref_nav) > window:
            ref_idx = ref_nav.index
            rolling_rets = []
            for i in range(window, len(ref_nav)):
                if ref_idx[i] <= ret.index[-1]:
                    rolling_rets.append(ref_nav.iloc[i] / ref_nav.iloc[i - window] - 1)
            if len(rolling_rets) > 0:
                rarr = np.array(rolling_rets)
                cur_pct = float((rarr < cur_ret).sum()) / len(rarr)
                cur_z = (cur_ret - rarr.mean()) / rarr.std() if rarr.std() > 0 else np.nan
            else:
                cur_pct = np.nan
                cur_z = np.nan
        else:
            cur_pct = np.nan
            cur_z = np.nan

        row_labels.append(f"近{window}日")
        ret_list.append(cur_ret * 100)
        pct_list.append(cur_pct)
        z_list.append(cur_z)
        vol_list.append(np.nan)
        z250_list.append(np.nan)
        z750_list.append(np.nan)
        rank250_list.append(np.nan)
        rank750_list.append(np.nan)

    tbl_single = pd.DataFrame({
        "收益率(%)": ret_list,
        "收益率历史分位数(%)": pct_list,
        "收益率z值": z_list,
        **({
            "vol": vol_list,
            "z_250d": z250_list,
            "z_750d": z750_list,
            "波动率分位数250(%)": rank250_list,
            "波动率分位数750(%)": rank750_list,
        } if sub_cat == "风格因子" else {}),
    }, index=row_labels).round(4)

    bar_s_cols = ["收益率(%)"]
    pct_s_cols = ["收益率历史分位数(%)"] + (["波动率分位数250(%)", "波动率分位数750(%)"] if sub_cat == "风格因子" else [])
    fmt_map = {
        "收益率(%)": "{:.3f}%",
        "收益率历史分位数(%)": "{:.2%}",
        "收益率z值": "{:.3f}",
    }
    if sub_cat == "风格因子":
        fmt_map.update({
            "vol": "{:.2%}",
            "z_250d": "{:.3f}",
            "z_750d": "{:.3f}",
            "波动率分位数250(%)": "{:.2%}",
            "波动率分位数750(%)": "{:.2%}",
        })
    styled_single = tbl_single.style.format(fmt_map, na_rep="-")
    styled_single = styled_single.bar(subset=pct_s_cols, color="#d65f5f", vmin=0, vmax=1)
    html_s = styled_single.set_table_styles([
        {"selector": "td, th", "props": [("padding", "5px 10px"), ("text-align", "right"), ("white-space", "nowrap")]},
        {"selector": "th", "props": [("text-align", "left"), ("font-weight", "bold")]},
    ]).to_html()
    st.markdown(f"""<div style="overflow-x:auto; width:100%;">{html_s}</div>""", unsafe_allow_html=True)

with st.expander("📋 数据加载日志"):
    for line in debug_log:
        st.markdown(f"- {line}")