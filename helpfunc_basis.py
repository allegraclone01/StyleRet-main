from rqdatac import *
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
srcdir = os.path.join(BASE_DIR, "data_base", "basis","index_future_basics.pkl")

idxdir = os.path.join(BASE_DIR, "data_base", "index","股指宽基指数收盘价_2601_2607.pkl")
cmpdir = os.path.join(BASE_DIR, "data_base", "index_component_日频")
cachedir = os.path.join(BASE_DIR, "data_base", "basis", "output")

epsdir = os.path.join(BASE_DIR, "data_base", "basis","866011.RI_eps_24_25Q.pkl")
#df_eps = pd.read_pickle(epsdir)
dpsdir = os.path.join(BASE_DIR, "data_base", "basis","866011.RI_dps_22_25H.pkl")
#df_dps = pd.read_pickle(dpsdir)
timedir = os.path.join(BASE_DIR, "data_base", "basis","dividend_timeline.pkl")
#df_time = pd.read_pickle(timedir)
alldir = os.path.join(BASE_DIR, "data_base", "index_component_日频","866011.RI_20_26D_dict.pkl")
all_df = pd.read_pickle(alldir)
all_ids = list(all_df.values())[-1].index.tolist()

def last_trading_day(ref=None):
    """返回最近一个完整交易日。周一时返回上周五，周日时返回上周五，其余返回前一天。"""
    if ref is None:
        ref = pd.Timestamp.now().normalize()
    w = ref.weekday()  # 0=周一, 6=周日
    if w == 0:
        return ref - pd.Timedelta(days=3)
    if w == 6:
        return ref - pd.Timedelta(days=2)
    return ref - pd.Timedelta(days=1)


def add_basis_data(st,ed):
    df_future = all_instruments(type='Future')
    df_index = df_future[df_future["product"]=="Index"]
    df_index_real = df_index[df_index["maturity_date"]!="0000-00-00"]
    df_index_real.to_pickle(srcdir)

    contracts = df_index_real["order_book_id"].tolist()
    df_info = futures.get_basis(contracts, start_date=st, end_date=ed, fields=["settlement","close_index"], frequency='1d', dividend_adjusted=False, market='cn')
    df_info = df_info.reset_index(level=1)

    # 计算基础指标
    df_info["basis"] = df_info["settlement"] - df_info["close_index"]
    df_info["abs_ratio"] = df_info["basis"] / df_info["close_index"]

    df_index_real.set_index(["order_book_id"], inplace=True)
    df_info_m = df_info.merge(df_index_real[["listed_date","maturity_date"]], on=["order_book_id"], how="left")
    # 统一转为日期格式
    df_info_m["maturity_date"] = pd.to_datetime(df_info_m["maturity_date"])
    df_info_m["date"] = pd.to_datetime(df_info_m["date"])
    # 再计算间隔天数
    df_info_m["residual_day"] = np.where(
        (df_info_m["maturity_date"] - df_info_m["date"]).dt.days == 0,
        np.nan,
        (df_info_m["maturity_date"] - df_info_m["date"]).dt.days
    )

    df_info_m["ana_cost"] = df_info_m["abs_ratio"] / df_info_m["residual_day"] * 365

    return df_info_m

def get_dividend_payratio(quarter:str,stk:str,phase:str):
    """
    计算上一年的分红支付率
    由于分红率存在切换的趋势，因此没必要取前三年的平均，否则误差会持续存在两年
    """
    #pre_quarter = str(int(quarter[:4]) - 1) + quarter[4:]

    #提取报告期的净利润
    try:
        df_eps_current = df_eps.loc[(stk, quarter)]
        n_profit = df_eps_current["net_profit"]
    except:
        n_profit = np.nan
    #提取报告期的分红金额
    try:
        df_fh = df_time.loc[(stk, quarter)]
        df_fh = df_fh[df_fh["event_procedure"]==phase]
        fh = df_fh["amount"].values[0]
    except:
        fh = np.nan

    dividend_payratio = fh / n_profit

    return dividend_payratio if dividend_payratio else 0
    
def get_eps(quarter:str,stk:str):
    """
    获取指定报告期的eps
    """
    try:
        df_eps_current = df_eps.loc[(stk, quarter)] 
        eps = df_eps_current["basic_earnings_per_share"] 
    except:
        eps = np.nan
    return eps

def active_contract(dt):
    """
    获取当前日期后的所有活跃合约，包含合约代码和到期日期
    """
    df_index_real_new = pd.read_pickle(srcdir)
    df_index_real_new = df_index_real_new[pd.to_datetime(df_index_real_new["maturity_date"]) > pd.to_datetime(dt)] 
    print(f"{dt}当天存续的合约个数：{len(df_index_real_new)}") #今天交割的合约，期现价格已经收敛
    active_df = df_index_real_new[["order_book_id","maturity_date"]]
    return active_df

def get_info_d(c_id,dt):
    """
    获取指定合约在指定日期的成分股信息,包含成分股代码，成分股权重，成分股价格
    """
    # 合约前缀 -> 指数代码映射  IC:000905 IM:000852 IH:000016 IF:000300
    _prefix = str(c_id)[:2].upper()
    _idx_map = {"IC": "000905", "IM": "000852", "IH": "000016", "IF": "000300"}
    if _prefix not in _idx_map:
        return pd.DataFrame()
    _idx = _idx_map[_prefix] + ".XSHG"
    
    _d_str = pd.Timestamp(dt).strftime("%Y%m%d")
    df = index_weights_ex(_idx, start_date=_d_str, end_date=_d_str, market="cn").reset_index(level=0)

    ids = df.index.tolist() #dt天成分股
    df_price = get_price(ids, start_date=_d_str, end_date=_d_str, frequency='1d', fields=["close"], adjust_type='pre', skip_suspended=False, expect_df=True, time_slice=None, market='cn').reset_index(level=1)
    df = df.merge(df_price, on=["order_book_id"], how="left")
    
    #存量更新一下数据
    file_path_w = os.path.join(cmpdir, f"{_idx}_20_26D_dict.pkl")
    with open(file_path_w, 'rb') as f:
        df_w_old = pickle.load(f)
    if pd.Timestamp(dt) not in list(df_w_old.keys()):
        df_w_old[pd.Timestamp(dt)] = df["weight"]
        with open(file_path_w, 'wb') as f:
            pickle.dump(df_w_old, f)  # 这里的 df 就是你的字典变量名  
        
        file_path_p = os.path.join(cmpdir, f"{_idx}成分股价_26D_dict.pkl")
        with open(file_path_p, 'rb') as f:
            price_dict = pickle.load(f)
        price_dict[pd.Timestamp(dt)] = df_price["close"]
        with open(file_path_p, 'wb') as f:
            pickle.dump(price_dict, f)  # 这里的 df 就是你的字典变量名


    return ids, df if isinstance(df, pd.DataFrame) else pd.DataFrame()

def get_index_d(dt):
    """
    获取指定日期的指数收盘价格
    """
    df_index = get_price(["000016.XSHG","000300.XSHG", "000905.XSHG","000852.XSHG"], start_date=dt, end_date=dt, frequency='1d', fields=["close"], adjust_type='pre', skip_suspended=False, expect_df=True, time_slice=None, market='cn')

    #存量更新一下数据
    df_old = pd.read_pickle(idxdir)
    df_new = pd.concat([df_old,df_index],axis=0).drop_duplicates().sort_index()
    df_new.to_pickle(idxdir)
    print(f"指数收盘价多增{len(df_new)-len(df_old)}条")

    return df_index

def update_dps(dt):
    _dps_old = pd.read_pickle(dpsdir) if os.path.exists(dpsdir) else pd.DataFrame()
    #_date_col = "ex_dividend_date"
    # 确定增量起点
    if not _dps_old.empty: #and _date_col is not None:
        _start = pd.Timestamp(_dps_old.index.get_level_values("declaration_announcement_date").max()) + pd.Timedelta(days=1)
    else:
        _start = pd.Timestamp("2024-06-30")
    _end = pd.Timestamp(dt)
    if _start <= _end:
        _dps_new = get_dividend(all_ids,start_date=_start.strftime("%Y%m%d"),end_date=_end.strftime("%Y%m%d"),expect_df=True, market='cn')
        if isinstance(_dps_new, pd.DataFrame) and not _dps_new.empty:
            _n_old = len(_dps_old)
            _dps_old = _dps_old.drop_duplicates()  # 先清历史重复，避免干扰计数
            _dps_old = pd.concat([_dps_old, _dps_new], axis=0).drop_duplicates().sort_index()
            print(f"[update_dps] 拉取{len(_dps_new)}条, 净变化 {len(_dps_old) - _n_old:+d} 条, {_start.date()} ~ {_end.date()}")
    #存储
    _dps_old.to_pickle(dpsdir)
    return _dps_old

def update_eps(end_q):
    #增量更新 eps：读老数据 → 取老数据最大 quarter → 从下一季度拉到 end_q → 合并去重存回
    _eps_old = pd.read_pickle(epsdir) if os.path.exists(epsdir) else pd.DataFrame()

    _start_q = "2025q4"
    _new = get_pit_financials_ex(all_ids, ["basic_earnings_per_share"],start_quarter=_start_q, end_quarter=end_q,date=None, statements='latest', market='cn')

    if isinstance(_new, pd.DataFrame) and not _new.empty:
        _n_old = len(_eps_old)
        _eps_old = pd.concat([_eps_old, _new], axis=0)#
        mask = _eps_old.index.duplicated(keep="first")
        _eps_old = _eps_old[~mask].sort_index()
        print(f"[update_eps] +{len(_eps_old) - _n_old} 条, {_start_q} ~ {end_q}")
        _eps_old.to_pickle(epsdir)

    return _eps_old

def update_timeline(end_q):
    #增量更新 分红时间时间线：读老数据 → 取老数据最大 quarter → 从下一季度拉到 end_q → 合并去重存回
    _time_old = pd.read_pickle(timedir) if os.path.exists(timedir) else pd.DataFrame()

    _start_q = "2025q4"
    _new = get_dividend_amount(all_ids, start_quarter = _start_q, end_quarter = end_q, date = None, market = 'cn')

    if isinstance(_new, pd.DataFrame) and not _new.empty:
        _n_old = len(_time_old)
        _time_old = pd.concat([_time_old, _new], axis=0).drop_duplicates().sort_index()
        print(f"[update_timeline] +{len(_time_old) - _n_old} 条, {_start_q} ~ {end_q}")
        _time_old.to_pickle(timedir)

    return _time_old

def cal_fhds(dt, new, return_detail=False):

    # ===== 基础数据准备 =====
    active_df = active_contract(dt)
    if active_df.empty:
        return (pd.DataFrame(), pd.DataFrame()) if return_detail else pd.DataFrame()

    _y = pd.Timestamp(dt).year
    quarter_list = [f"{_y - 1}q2", f"{_y - 1}q4", f"{_y}q2"]

    _dps = update_dps(dt)
    _eps = update_eps(quarter_list[-1])
    _time = update_timeline(quarter_list[-1])

    global df_dps, df_eps, df_time
    df_dps = _dps
    df_eps = _eps
    df_time = _time

    _dt = pd.Timestamp(dt)
    idx_map = {"IC": "000905", "IM": "000852", "IH": "000016", "IF": "000300"}

    # ===== 提速1: 指数收盘价 dict（避免每合约扫 idx_prices.index）=====
    idx_prices = get_index_d(dt)
    _idx_prices = {}
    for _pfx, _code in idx_map.items():
        _k = _code + ".XSHG"
        try:
            _mask = idx_prices.index.get_level_values(0) == _k
            if _mask.any():
                _idx_prices[_pfx] = float(idx_prices.loc[_mask, "close"].iloc[0])
        except (KeyError, IndexError):
            pass

    # ===== 提速2: 按合约前缀分组，每个前缀只调用一次 get_info_d =====
    active_df["_prefix"] = active_df["order_book_id"].astype(str).str[:2].str.upper()
    _comp_cache = {}  # prefix -> {stk: (price, weight)}
    for _pfx in active_df["_prefix"].unique():
        if _pfx not in idx_map:
            continue
        _sample = active_df.loc[active_df["_prefix"] == _pfx, "order_book_id"].iloc[0]
        _ids, _cdf = get_info_d(_sample, dt)
        if _cdf.empty:
            continue
        # comp_df 的 index 就是 order_book_id
        _sw = {}
        for _s in _ids:
            try:
                _price = float(_cdf.loc[_s, "close"])
                _weight = float(_cdf.loc[_s, "weight"])
                if _price > 0 and _weight > 0:
                    _sw[_s] = (_price, _weight)
            except (KeyError, TypeError, ValueError):
                continue
        _comp_cache[_pfx] = _sw

    if not _comp_cache:
        _r = active_df[["order_book_id"]].copy()
        _r["dividend_point"] = 0.0
        daily_df = new.merge(_r, on="order_book_id", how="right")
        return (daily_df, pd.DataFrame()) if return_detail else daily_df

    # ===== 提速3: 收集所有涉及的股票，预过滤 & 预建 {stk: sub_df} =====
    _all_stks = set()
    for _sw in _comp_cache.values():
        _all_stks.update(_sw.keys())
    _all_stks = list(_all_stks)

    # 一次 isin 过滤，剩下只有相关股票的数据，之后 .xs(stk) 就是小表的快速查找
    _dps_f = _dps[_dps.index.get_level_values(0).isin(_all_stks)]
    _eps_f = _eps[_eps.index.get_level_values(0).isin(_all_stks)]
    _time_f = _time[_time.index.get_level_values(0).isin(_all_stks)]

    # 预构建 {stk: sub_df}，避免循环内反复 .xs()；不用 .unique()，Index.__contains__ 已是 O(1)
    _dps_dict = {s: _dps_f.xs(s, level=0, drop_level=True) for s in _all_stks if s in _dps_f.index.get_level_values(0)}
    _eps_dict = {s: _eps_f.xs(s, level=0, drop_level=True) for s in _all_stks if s in _eps_f.index.get_level_values(0)}
    _time_dict = {s: _time_f.xs(s, level=0, drop_level=True) for s in _all_stks if s in _time_f.index.get_level_values(0)}

    # ===== 提速4: 预计算每只股票 × 每个 quarter 的 (dividend, ex_date) =====
    # 关键改动：_time_sub / _dps_sub / _eps_sub 提到 _stk 外循环；
    # 用 loc[[q]] 永远返回 DataFrame，消除 Series→DataFrame 转换；
    # 所有数据访问改为单索引子表 lookup，不再对 _dps_f/_time_f 做 MultiIndex .loc
    _fhds_cache = {}  # (stk, q) -> (dividend, ex_date)
    for _stk in _all_stks:
        _time_sub = _time_dict.get(_stk)
        if _time_sub is None:
            continue
        _dps_sub = _dps_dict.get(_stk)
        _eps_sub = _eps_dict.get(_stk)

        for q in quarter_list:
            if q not in _time_sub.index:
                continue
            rows = _time_sub.loc[[q]]  # 始终 DataFrame

            events = rows["event_procedure"].tolist()
            if "方案实施" in events:
                _evt, _pred = "方案实施", False
            elif "决案" in events:
                _evt, _pred = "决案", True
            elif "预案" in events:
                _evt, _pred = "预案", True
            else:
                continue

            try:
                info_date = rows[rows["event_procedure"] == _evt]["info_date"].iloc[0]
                if not _pred:
                    # 方案实施：从预建 _dps_dict 查（单索引 lookup，避免 MultiIndex .loc）
                    if _dps_sub is None or info_date not in _dps_sub.index:
                        continue
                    row = _dps_sub.loc[info_date]
                    dividend = float(row["dividend_cash_before_tax"]) / 10
                    ex_date = pd.Timestamp(row["ex_dividend_date"])
                    _type_label = "①方案实施"
                else:
                    # 决案/预案：用去年同季 (event_date → ex_date) 的间隔预测
                    pre_q = str(int(q[:4]) - 1) + q[4:]
                    if pre_q not in _time_sub.index:
                        continue
                    pre_rows = _time_sub.loc[[pre_q]]
                    _pre_event_mask = pre_rows["event_procedure"] == _evt
                    if not _pre_event_mask.any():
                        continue
                    _pre_info = pd.Timestamp(pre_rows.loc[_pre_event_mask, "info_date"].iloc[0])

                    if _dps_sub is None:
                        continue
                    _pre_q_mask = _dps_sub["quarter"] == pre_q
                    if not _pre_q_mask.any():
                        continue
                    _pre_ex = pd.Timestamp(_dps_sub.loc[_pre_q_mask, "ex_dividend_date"].iloc[-1])

                    ex_date = pd.Timestamp(info_date) + (_pre_ex - _pre_info)
                    _type_label = f"②{_evt}"

                    # 若预测日期不晚于明天，用历史同季度 ex_date 重估（均需 +1 年投射到当前年份）
                    _cutoff = _dt + pd.Timedelta(days=1)
                    if ex_date <= _cutoff:
                        _y_q, _q_num = int(q[:4]), q[4:]
                        _same_qs = [f"{_y_q - i}{_q_num}" for i in range(1, 4)]  # 前三年同季度
                        _ex_same_q = pd.to_datetime(
                            _dps_sub[_dps_sub["quarter"].isin(_same_qs)]
                            .groupby("quarter")["ex_dividend_date"].last().dropna()
                        )  # 每季度只保留最后一条（防止多次分红的情况），确保每年至多一个 ex_date
                        if len(_ex_same_q) > 0:
                            tmp = _ex_same_q.apply(lambda x: x.replace(year=2000))
                            avg_doy = round(tmp.dt.dayofyear.mean())
                            _avg3 = (pd.Timestamp("2000-01-01") + pd.Timedelta(days=avg_doy - 1)).replace(year=_y)
                            if _avg3 > _cutoff:
                                ex_date = _avg3
                                _type_label = "③历史平均"
                        if ex_date <= _cutoff:
                            if _evt == "决案":
                                ex_date = info_date + pd.DateOffset(days=30)
                            elif _evt == "预案":
                                ex_date = info_date + pd.DateOffset(days=70)
                            #ex_date = _pre_ex + pd.DateOffset(years=1)
                            _type_label = "④同类平均"

                    # 分红金额 = 当前 eps × 支付率（都用当前季度 q）
                    if _eps_sub is None or q not in _eps_sub.index:
                        continue
                    _cur_row = _eps_sub.loc[q]
                    current_eps = float(_cur_row["basic_earnings_per_share"])
                    n_profit = float(_cur_row["net_profit"])
                    fh = float(rows.loc[rows["event_procedure"] == _evt, "amount"].iloc[0])
                    payratio = fh / n_profit
                    dividend = current_eps * payratio

                _fhds_cache[(_stk, q)] = (dividend, ex_date, _type_label)
            except (KeyError, TypeError, ValueError, IndexError, ZeroDivisionError):
                continue

    # ===== 提速5: 合约循环 -> 纯查表汇总 =====
    results = []
    for _idx, cont_row in active_df.iterrows():
        c_id = str(cont_row["order_book_id"])
        maturity = pd.Timestamp(cont_row["maturity_date"])
        _pfx = cont_row["_prefix"]

        _sw = _comp_cache.get(_pfx)
        idx_price = _idx_prices.get(_pfx)
        if _sw is None or idx_price is None or np.isnan(idx_price) or idx_price == 0:
            results.append({"order_book_id": c_id, "dividend_point": 0})
            continue

        fhds = 0.0
        for stk, (stk_price, stk_weight) in _sw.items():
            for q in quarter_list:
                cached = _fhds_cache.get((stk, q))
                if cached is None:
                    continue
                dividend, ex_date, _ = cached
                if ex_date <= _dt or ex_date > maturity:
                    continue
                if dividend and not np.isnan(dividend):
                    fhds += (dividend / stk_price) * stk_weight * idx_price

        results.append({"order_book_id": c_id, "dividend_point": fhds})

    df_result = pd.DataFrame(results).set_index("order_book_id")
    daily_df = new.merge(df_result, on="order_book_id", how="right")

    if return_detail:
        _detail_rows = []
        for _pfx, _sw in _comp_cache.items():
            for _stk, (_price, _) in _sw.items():
                for q in quarter_list:
                    cached = _fhds_cache.get((_stk, q))
                    if cached is None:
                        continue
                    _dividend, _ex_date, _type_label = cached
                    _detail_rows.append({
                        "stock": _stk, "prefix": _pfx, "quarter": q,
                        "type": _type_label, "dividend": _dividend, "ex_date": _ex_date,
                    })
        return daily_df, pd.DataFrame(_detail_rows)

    return daily_df


def plot_fhds_detail(detail_df, prefixes=None):
    """分红除权日历图：每指数一个子图，柱状（除权公司数，按①②类型堆叠）+ 折线（平均每股分红）"""
    if detail_df.empty:
        return None
    if prefixes is None:
        prefixes = sorted(detail_df["prefix"].unique())
    elif isinstance(prefixes, str):
        prefixes = [prefixes]

    _df = detail_df.copy()
    _df["ex_date_d"] = _df["ex_date"].dt.date
    _type_order = ["①方案实施", "②预案", "②决案", "③历史平均", "④同类平均"]
    _colors = {"①方案实施": "#2ca02c", "②决案": "#ff7f0e", "②预案": "#d62728", "③历史平均": "#9467bd", "④同类平均": "#8c564b"}

    fig, axes = plt.subplots(len(prefixes), 1, figsize=(14, 4.5 * len(prefixes)), sharex=False, constrained_layout=True)
    if len(prefixes) == 1:
        axes = [axes]

    for i, pfx in enumerate(prefixes):
        ax = axes[i]
        sub = _df[_df["prefix"] == pfx]
        if sub.empty:
            continue
        dates = sorted(sub["ex_date_d"].unique())

        bottom = np.zeros(len(dates))
        for t in _type_order:
            cnts = sub[sub["type"] == t].groupby("ex_date_d").size().reindex(dates, fill_value=0).values
            ax.bar(dates, cnts, bottom=bottom, label=t, color=_colors.get(t, "#999"), width=0.8)
            bottom = bottom + cnts

        avg_div = sub.groupby("ex_date_d")["dividend"].mean().reindex(dates)
        ax2 = ax.twinx()
        ax2.plot(dates, avg_div.values, "o-", color="#1f77b4", lw=2, markersize=6)

        ax.set_title(pfx, fontsize=12, fontweight="bold")
        ax.set_ylabel("除权公司数")
        ax2.set_ylabel("平均每股分红", color="#1f77b4")
        ax2.tick_params(axis="y", labelcolor="#1f77b4")
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(alpha=0.3)
        ax.tick_params(axis="x", rotation=45)

    return fig


def _get_fhds_cached(contract_id, dates, idx_close_df, weight_dict, price_dict,
                     df_dps, df_eps, df_time):
    """按合约缓存 cal_fhds_history 结果，只计算新增日期后合并。"""
    cache_path = os.path.join(cachedir, f"{contract_id}_dividend.pkl")

    if os.path.exists(cache_path):
        cached = pd.read_pickle(cache_path)
        old = set(pd.Timestamp(d) for d in cached['date'])
        new_dates = [d for d in dates if pd.Timestamp(d) not in old]
    else:
        cached = pd.DataFrame()
        new_dates = list(dates)

    if new_dates:
        new_result = cal_fhds_history(contract_id, new_dates, idx_close_df,
                                      weight_dict, price_dict, df_dps, df_eps, df_time)
        cached = pd.concat([cached, new_result], ignore_index=True).drop_duplicates()
        os.makedirs(cachedir, exist_ok=True)
        cached.to_pickle(cache_path)

    return cached


def plot_basis_series(df, contract_id=None, mode="origin", cols_per_seg=20):
    """
    单合约基差成本监控图 + 转置数据表 HTML。

    Parameters
    ----------
    df : pd.DataFrame
        mode="origin": 需含 date, abs_ratio, ana_cost
        mode="adjust": 额外需含 dividend_point, close_index, residual_day
    contract_id : str  合约代码，用于图表标题
    mode : str  "origin"=原始 | "adjust"=含分红调整
    cols_per_seg : int 数据表每段列数，默认 20

    Returns
    -------
    fig : matplotlib Figure
    tbl_html : str  HTML 字符串，可直接 st.markdown(..., unsafe_allow_html=True)
    """
    # ---- mode="adjust": 日期检查 & 分红计算 ----
    _dates_raw = sorted(df["date"].unique())
    if mode == "adjust" and pd.Timestamp(_dates_raw[0]) < pd.Timestamp("2026-01-01"):
        mode = "origin"

    if mode == "adjust":
        # 加载预存数据
        idx_map = {"IC": "000905", "IM": "000852", "IH": "000016", "IF": "000300"}
        _prefix = str(contract_id)[:2].upper()
        index = idx_map[_prefix] + ".XSHG"

        idx_close_df = pd.read_pickle(idxdir)

        file_path_w = os.path.join(cmpdir, f"{index}_20_26D_dict.pkl")
        with open(file_path_w, 'rb') as f:
            weight_dict = pickle.load(f)

        file_path_p = os.path.join(cmpdir, f"{index}成分股价_26D_dict.pkl")
        with open(file_path_p, 'rb') as f:
            price_dict = pickle.load(f)

        df_dps = pd.read_pickle(dpsdir)
        df_eps = pd.read_pickle(epsdir)
        df_time = pd.read_pickle(timedir)

        # 带缓存计算（首次全量，后续仅算增量）
        result = _get_fhds_cached(contract_id, _dates_raw, idx_close_df, weight_dict,
                                  price_dict, df_dps, df_eps, df_time)

        df = df.merge(result, on="date")
        df["adj_basis"] = df["basis"] + df["dividend_point"]
        df["adj_abs_ratio"] = df["adj_basis"] / df["close_index"]
        df["adj_ana_cost"] = df["adj_abs_ratio"] / df["residual_day"] * 365

    # ---- 绘图 ----
    dates = df["date"].values  # merge 之后取，保证 x/y 一致
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(dates, df["abs_ratio"].values * 100, color="#1f77b4", lw=2, label="abs_ratio (%)")
    ax.plot(dates, df["ana_cost"].values * 100, color="#ff7f0e", lw=2, label="ana_cost (%)")
    if mode == "adjust":
        ax.plot(dates, df["adj_abs_ratio"].values * 100, color="#1f77b4", lw=2,
                ls="--", label="adj_abs_ratio (%)")
        ax.plot(dates, df["adj_ana_cost"].values * 100, color="#ff7f0e", lw=2,
                ls="--", label="adj_ana_cost (%)")

    title = f"基差成本监控 — {contract_id}" if contract_id else "基差成本监控"
    if mode == "adjust":
        title += "（含分红调整）"
    ax.set_title(title)
    ax.set_xlabel("date")
    ax.set_ylabel("%")
    ax.legend(fontsize=9, loc="best")
    ax.grid(alpha=0.3)
    ax.axhline(0, color="black", lw=0.8)
    fig.autofmt_xdate()

    # ---- 转置数据表 ----
    if mode == "adjust":
        tbl_cols = ["date", "abs_ratio", "adj_abs_ratio", "ana_cost", "adj_ana_cost"]
        pct_cols = ["abs_ratio(%)", "adj_abs_ratio(%)", "ana_cost(%)", "adj_ana_cost(%)"]
    else:
        tbl_cols = ["date", "abs_ratio", "ana_cost"]
        pct_cols = ["abs_ratio(%)", "ana_cost(%)"]

    tbl = df[tbl_cols].copy()
    tbl["date"] = tbl["date"].dt.strftime("%Y-%m-%d")
    for raw_col, pct_col in zip(tbl_cols[1:], pct_cols):
        tbl[pct_col] = tbl[raw_col] * 100
    tbl = tbl[["date"] + pct_cols].round(3).tail(30)
    tbl = tbl.sort_values("date").reset_index(drop=True)

    _wide = tbl.set_index("date").T
    _wide.index.name = None
    cols = list(_wide.columns)

    seg_count = (len(cols) + cols_per_seg - 1) // cols_per_seg
    html_parts = []
    for si in range(seg_count):
        seg_cols = cols[si * cols_per_seg:(si + 1) * cols_per_seg]
        seg_html = (_wide[seg_cols].style.format("{:.3f}").set_table_styles([
            {"selector": "td, th",
             "props": [("padding", "3px 6px"), ("text-align", "right"),
                       ("font-size", "0.8rem"), ("white-space", "nowrap")]},
            {"selector": "th.row_heading",
             "props": [("text-align", "left"), ("font-weight", "bold")]},
        ]).to_html())
        html_parts.append(
            f'<div style="overflow-x:auto; width:100%; margin-bottom:8px;">{seg_html}</div>'
        )

    return fig, "\n".join(html_parts)


def cal_fhds_history(c_id, date_list, idx_close_df, weight_dicts, price_dicts,
                     df_dps, df_eps, df_time, return_detail=False):
    """
    计算单合约在历史日期序列上的每日分红点数（无未来信息泄露）

    站在每个 his_dt 时点，仅使用 his_dt 之前已知的数据（PIT 遮罩），
    预测成分股分红并汇总为当日 dividend_point。

    Parameters
    ----------
    c_id : str
        合约代码，如 'IF2306'，前2位确定指数前缀
    date_list : array-like
        历史日期序列，调用者保证在合约存续期内
    idx_close_df : pd.DataFrame
        MultiIndex=(order_book_id, date), columns=['close']
    weight_dicts : dict
        {pd.Timestamp: pd.Series(stock_id → weight)}，已对应当前指数
    price_dicts : dict
        {pd.Timestamp: pd.Series(stock_id → close)}，已对应当前指数
    df_dps : pd.DataFrame
        MultiIndex=(order_book_id, declaration_announcement_date)
        PIT: index level 1 <= his_dt
    df_eps : pd.DataFrame
        MultiIndex=(order_book_id, quarter)，含 rice_create_tm 列
        PIT: rice_create_tm <= his_dt
    df_time : pd.DataFrame
        MultiIndex=(order_book_id, quarter)，含 rice_create_tm 列
        PIT: rice_create_tm <= his_dt
    return_detail : bool

    Returns
    -------
    result_df : pd.DataFrame  columns=[date, dividend_point]
    detail_df : pd.DataFrame  (if return_detail=True)
    """

    print("开始计算历史分红点数")
    # ===== 0. 预处理 =====
    idx_map = {"IC": "000905", "IM": "000852", "IH": "000016", "IF": "000300"}
    _prefix = str(c_id)[:2].upper()
    if _prefix not in idx_map:
        raise ValueError(f"无法识别合约前缀: {_prefix}，应为 IC/IM/IH/IF 之一")
    idx_code = idx_map[_prefix] + ".XSHG"  # e.g. '000905.XSHG'

    # 取合约到期日（从 srcdir 缓存中读取）
    contract_df = pd.read_pickle(srcdir)
    contract_df['maturity_date'] = pd.to_datetime(contract_df['maturity_date'])
    maturity = pd.Timestamp(contract_df.set_index('order_book_id').loc[c_id, 'maturity_date'])

    # 收集该指数全时段出现过的所有成分股 → 预过滤三类 PIT 数据
    all_stks = set()
    for _s in weight_dicts.values():
        all_stks.update(_s.index.tolist())
    all_stks = list(all_stks)

    if not all_stks:
        base = [{'date': pd.Timestamp(d), 'dividend_point': 0.0} for d in date_list]
        return (pd.DataFrame(base), pd.DataFrame()) if return_detail else pd.DataFrame(base)

    dps_f = df_dps[df_dps.index.get_level_values(0).isin(all_stks)]
    eps_f = df_eps[df_eps.index.get_level_values(0).isin(all_stks)]
    time_f = df_time[df_time.index.get_level_values(0).isin(all_stks)]

    # 预建 per-stock 字典（全量数据，PIT 过滤在查表时进行）
    dps_dict = {s: dps_f.xs(s, level=0, drop_level=True)
                for s in all_stks if s in dps_f.index.get_level_values(0)}
    eps_dict = {s: eps_f.xs(s, level=0, drop_level=True)
                for s in all_stks if s in eps_f.index.get_level_values(0)}
    time_dict = {s: time_f.xs(s, level=0, drop_level=True)
                 for s in all_stks if s in time_f.index.get_level_values(0)}

    # ===== 主循环：逐日计算 =====
    all_results = []
    all_details = [] if return_detail else None

    for dt_val in date_list:
        his_dt = pd.Timestamp(dt_val) #+ pd.Timedelta(hours=23, minutes=59, seconds=59) #米筐录入可能有时分秒的延迟，不能以00:00:00为标准
        _y = his_dt.year
        quarter_list = [f"{_y - 1}q2", f"{_y - 1}q4", f"{_y}q2"]

        # ----- ① 指数收盘价 -----
        try:
            idx_price = float(idx_close_df.loc[(idx_code, his_dt), 'close'])
            if np.isnan(idx_price) or idx_price == 0:
                all_results.append({'date': his_dt, 'dividend_point': 0.0})
                continue
        except KeyError:
            all_results.append({'date': his_dt, 'dividend_point': 0.0})
            continue

        # ----- ② 当日成分股: weight ⋂ price -----
        if his_dt not in weight_dicts or his_dt not in price_dicts:
            all_results.append({'date': his_dt, 'dividend_point': 0.0})
            continue

        w_series = weight_dicts[his_dt]
        p_series = price_dicts[his_dt]
        common_stks = w_series.index.intersection(p_series.index)

        _comp = {}
        for stk in common_stks:
            _w = float(w_series[stk])
            _p = float(p_series[stk])
            if _w > 0 and _p > 0:
                _comp[stk] = (_p, _w)

        if not _comp:
            all_results.append({'date': his_dt, 'dividend_point': 0.0})
            continue

        # ----- ③④ PIT 过滤 + 分红预测（与 cal_fhds 一致的 ①②③④ 逻辑）-----
        _fhds_cache = {}
        _cutoff = his_dt + pd.Timedelta(days=1)

        for stk in _comp:
            ts = time_dict.get(stk)
            ds = dps_dict.get(stk)
            es = eps_dict.get(stk)
            if ts is None:
                continue

            for q in quarter_list:
                if q not in ts.index:
                    continue
                # PIT: 只看 his_dt 之前创建的 timeline 记录
                rows = ts.loc[[q]]
                rows = rows[rows['rice_create_tm'] <= his_dt + pd.Timedelta(hours=23, minutes=59, seconds=59)]
                if rows.empty:
                    continue

                events = rows['event_procedure'].tolist()
                if "方案实施" in events:
                    _evt, _pred = "方案实施", False
                elif "决案" in events:
                    _evt, _pred = "决案", True
                elif "预案" in events:
                    _evt, _pred = "预案", True
                else:
                    continue

                try:
                    info_date = rows[rows['event_procedure'] == _evt]['info_date'].iloc[0]

                    if not _pred:
                        # ---- ① 方案实施：取真实分红数据 ----
                        if ds is None or info_date not in ds.index:
                            continue
                        # PIT: 公告日必须 < his_dt
                        if info_date > his_dt:
                            continue
                        row = ds.loc[info_date]
                        dividend = float(row['dividend_cash_before_tax']) / 10
                        ex_date = pd.Timestamp(row['ex_dividend_date'])
                        _type_label = "①方案实施"

                    else:
                        # ---- ② 决案/预案：预测 ex_date ----
                        pre_q = str(int(q[:4]) - 1) + q[4:]
                        if pre_q not in ts.index:
                            continue    #去年同季度不分红，默认今年也不分红
                        pre_rows = ts.loc[[pre_q]]
                        pre_rows = pre_rows[pre_rows['rice_create_tm'] <= his_dt]
                        if pre_rows.empty:
                            continue
                        _pre_evt_mask = pre_rows['event_procedure'] == _evt
                        if not _pre_evt_mask.any():
                            continue
                        _pre_info = pd.Timestamp(pre_rows.loc[_pre_evt_mask, 'info_date'].iloc[0])

                        if ds is None:
                            continue
                        # PIT: DPS 公告日 < his_dt
                        pre_q_mask = (ds['quarter'] == pre_q) & (ds.index <= his_dt)
                        if not pre_q_mask.any():
                            continue
                        _pre_ex = pd.Timestamp(ds.loc[pre_q_mask, 'ex_dividend_date'].iloc[-1])

                        ex_date = pd.Timestamp(info_date) + (_pre_ex - _pre_info)
                        _type_label = f"②{_evt}"

                        # ---- ③ 历史平均 ----
                        if ex_date <= _cutoff:
                            _y_q, _q_num = int(q[:4]), q[4:]
                            _same_qs = [f"{_y_q - i}{_q_num}" for i in range(1, 4)]
                            dps_pit = ds[ds.index <= his_dt]
                            _ex_same_q = pd.to_datetime(
                                dps_pit[dps_pit['quarter'].isin(_same_qs)]
                                .groupby('quarter')['ex_dividend_date'].last().dropna()
                            )
                            if len(_ex_same_q) > 0:
                                tmp = _ex_same_q.apply(lambda x: x.replace(year=2000))
                                avg_doy = round(tmp.dt.dayofyear.mean())
                                _avg3 = (pd.Timestamp("2000-01-01") +
                                         pd.Timedelta(days=avg_doy - 1)).replace(year=_y)
                                if _avg3 > _cutoff:
                                    ex_date = _avg3
                                    _type_label = "③历史平均"

                            # ---- ④ 同类平均 ----
                            if ex_date <= _cutoff:
                                if _evt == "决案":
                                    ex_date = info_date + pd.DateOffset(days=30)
                                elif _evt == "预案":
                                    ex_date = info_date + pd.DateOffset(days=70)
                                _type_label = "④同类平均"

                        # ---- EPS: PIT 检查 ----
                        if es is None or q not in es.index:
                            continue
                        _cur_row = es.loc[q]
                        if isinstance(_cur_row, pd.DataFrame):
                            _cur_row = _cur_row.iloc[-1]
                        _rice = _cur_row['rice_create_tm']
                        if pd.isna(_rice) or pd.Timestamp(_rice) > his_dt + pd.Timedelta(hours=23, minutes=59, seconds=59):
                            continue
                        current_eps = float(_cur_row['basic_earnings_per_share'])
                        n_profit = float(_cur_row['net_profit'])
                        fh = float(rows.loc[rows['event_procedure'] == _evt, 'amount'].iloc[0])
                        payratio = fh / n_profit
                        dividend = current_eps * payratio

                    _fhds_cache[(stk, q)] = (dividend, ex_date, _type_label)
                    
                except (KeyError, TypeError, ValueError, IndexError, ZeroDivisionError):
                    continue
        
        # ----- ⑤ 合约加总 -----
        fhds = 0.0
        for stk, (stk_price, stk_weight) in _comp.items():
            for q in quarter_list:
                cached = _fhds_cache.get((stk, q))
                if cached is None:
                    continue
                dividend, ex_date, _ = cached
                if ex_date <= his_dt or ex_date > maturity:
                    continue
                if dividend and not np.isnan(dividend):
                    fhds += (dividend / stk_price) * stk_weight * idx_price

        all_results.append({'date': his_dt, 'dividend_point': fhds})
        print(f"完成计算{(his_dt,fhds)}")

        # ----- Detail -----
        if return_detail:
            for stk in _comp:
                for q in quarter_list:
                    cached = _fhds_cache.get((stk, q))
                    if cached is None:
                        continue
                    _dividend, _ex_date, _type_label = cached
                    all_details.append({
                        'date': his_dt, 'stock': stk, 'prefix': _prefix, 'quarter': q,
                        'type': _type_label, 'dividend': _dividend, 'ex_date': _ex_date,
                    })

    # ===== 输出 =====
    result_df = pd.DataFrame(all_results)
    print(result_df)
    if return_detail:
        #print(all_details)
        return result_df, pd.DataFrame(all_details)
    return result_df



