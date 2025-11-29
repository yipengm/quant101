"""
Utilities for working with A-share monthly data using Baostock.

当前目标（根据最新需求）：
- 明确两个概念：
  a. 5 月线：前 5 个月的平均价格（通常指收盘价的 5 期滚动平均）
  b. 30 月线：前 30 个月的平均价格
- 实现一个函数，计算并打印某只股票的 5 月线和 30 月线。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
import os
import time

import akshare as ak
import baostock as bs
import pandas as pd


def _safe_baostock_history(
    *,
    symbol: str,
    period: str,
    start_date: str,
    end_date: Optional[str],
    adjust: str,
    max_retries: int = 3,
    base_sleep: float = 2.0,
) -> pd.DataFrame:
    """
    使用 Baostock 获取历史 K 线数据，并增加保守的重试逻辑。

    注意：假设外层已经调用过 bs.login()。
    - symbol: 不带交易所前缀的代码，例如 "600519"
    - period: "monthly" / "weekly"
    - start_date: "YYYYMMDD"
    - end_date: "YYYYMMDD" 或 None（None 表示当前日期）
    - adjust: "qfq" / "hfq" / 其它（不复权）
    """
    last_err: Optional[Exception] = None

    # Baostock 代码需要带前缀
    if symbol.startswith("6"):
        bs_code = f"sh.{symbol}"
    else:
        bs_code = f"sz.{symbol}"

    # 日期格式转换为 YYYY-MM-DD
    def _fmt(d: Optional[str]) -> Optional[str]:
        if d is None or d == "":
            return None
        if "-" in d:
            return d
        # 20200101 -> 2020-01-01
        return f"{d[0:4]}-{d[4:6]}-{d[6:8]}"

    start_d = _fmt(start_date)
    end_d = _fmt(end_date) or time.strftime("%Y-%m-%d")

    # 周期映射
    if period == "monthly":
        frequency = "m"
    elif period == "weekly":
        frequency = "w"
    else:
        frequency = "d"

    # 复权标志映射
    if adjust == "qfq":
        adjustflag = "2"
    elif adjust == "hfq":
        adjustflag = "1"
    else:
        adjustflag = "3"

    fields = ",".join(
        [
            "date",
            "code",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "amount",
            "adjustflag",
        ]
    )

    for i in range(max_retries):
        try:
            rs = bs.query_history_k_data_plus(
                bs_code,
                fields,
                start_date=start_d,
                end_date=end_d,
                frequency=frequency,
                adjustflag=adjustflag,
            )
            if rs.error_code != "0":
                raise RuntimeError(f"Baostock query failed: {rs.error_code}, {rs.error_msg}")

            data_list = []
            while rs.error_code == "0" and rs.next():
                data_list.append(rs.get_row_data())

            df = pd.DataFrame(data_list, columns=rs.fields)
            return df
        except Exception as e:
            last_err = e
            sleep_time = base_sleep * (i + 1)
            print(
                f"[retry {i+1}/{max_retries}] {bs_code} {period} failed: {e}. "
                f"Sleeping {sleep_time:.1f}s..."
            )
            time.sleep(sleep_time)

    # 多次重试失败，抛出最后一次异常
    if last_err is not None:
        raise last_err
    raise RuntimeError("Unknown error in _safe_baostock_history")


def refresh_all_a_share_codes(filepath: str = "a_share_codes.csv") -> pd.DataFrame:
    """
    从 AkShare 拉取最新的全部 A 股代码 + 名称，并保存到本地 CSV。

    只在需要更新映射表时调用，避免每次筛选都访问远端接口。
    """
    df = ak.stock_info_a_code_name()
    df = df[["code", "name"]].copy()
    df.to_csv(filepath, index=False)
    print(f"Saved {len(df)} A-share codes to {filepath}")
    return df


def get_target_share_codes(filepath: str = "a_share_codes.csv") -> pd.DataFrame:
    """
    获取“待筛选股票池”的代码列表（代码 + 名称）。

    默认从本地 CSV 读取（filepath 可配置），若不存在则自动从 AkShare
    拉取全部 A 股并保存一份。
    """
    if os.path.exists(filepath):
        df = pd.read_csv(filepath, dtype={"code": str, "name": str})
        return df[["code", "name"]]

    # 本地没有缓存时，调用一次远端接口并写入本地
    return refresh_all_a_share_codes(filepath=filepath)


def compute_monthly_ma(
    code: str,
    adjust: str = "qfq",
    months_window: int = 60,
) -> pd.DataFrame:
    """
    计算并打印某只股票最近两期的 5 月线和 30 月线。

    - 5 月线本月：最近 5 个月收盘价的平均
    - 5 月线上月：倒数第 2~6 个月收盘价的平均
    - 30 月线本月：最近 30 个月收盘价的平均
    - 30 月线上月：倒数第 2~31 个月收盘价的平均

    为了减少 API 调用次数，这里一次性获取最近 `months_window` 个月
    （默认 60 个月，大约 5 年）的月线数据，再从中计算上述两个参数。

    返回包含 date, code, close 的 DataFrame，并在控制台打印上述 4 个值。
    """

    df = fetch_recent_monthly_closes(
        code=code,
        adjust=adjust,
        months_window=months_window,
    )

    if df.empty:
        print(f"No monthly data returned for {code}.")
        return df

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["close"] = df["close"].astype(float)
    df.sort_values("date", inplace=True)
    df.set_index("date", inplace=True)

    n = len(df)
    if n < 6:
        print(
            f"\nNot enough monthly data to compute current & previous 5-month MAs for {code}. "
            f"Got only {n} months."
        )
        df.reset_index(inplace=True)
        return df[["date", "code", "close"]]

    # 5 月线：本月 vs 上月
    ma5_current = df["close"].iloc[-5:].mean()
    ma5_prev = df["close"].iloc[-6:-1].mean()

    # 30 月线需要至少 31 个月数据
    if n < 31:
        print(
            f"\nNot enough monthly data to compute current & previous 30-month MAs for {code}. "
            f"Got only {n} months."
        )
        ma30_current = float("nan")
        ma30_prev = float("nan")
    else:
        ma30_current = df["close"].iloc[-30:].mean()
        ma30_prev = df["close"].iloc[-31:-1].mean()

    last_date = df.index[-1].date()
    prev_date = df.index[-2].date()
    print(f"\nFor {code}, using data up to {last_date} (current month) and {prev_date} (previous month):")
    print(f"  5-month MA this month   (last 5 months):       {ma5_current:.2f}")
    print(f"  5-month MA last month   (months -2 to -6):     {ma5_prev:.2f}")

    if pd.notna(ma30_current) and pd.notna(ma30_prev):
        print(f"  30-month MA this month (last 30 months):      {ma30_current:.2f}")
        print(f"  30-month MA last month (months -2 to -31):    {ma30_prev:.2f}")
    else:
        print("  30-month MAs: not enough data to compute current & previous values.")

    # 可选：也打印出用于计算的最后几行，帮助检查
    print("\nLast 8 monthly closes used for reference:")
    print(df[["code", "close"]].tail(8))

    df.reset_index(inplace=True)
    return df[["date", "code", "close"]]


def fetch_recent_monthly_closes(
    code: str,
    adjust: str = "qfq",
    months_window: int = 60,
) -> pd.DataFrame:
    """
    帮助函数：获取某股票最近 months_window 个月的月线收盘价。

    返回 index 为日期、包含列 ['code', 'close'] 的 DataFrame，已按日期排序。

    注意：调用方需要确保在调用前已经完成 bs.login()，
    本函数内部不再重复登录 / 登出。
    """
    # 计算起始日期：当前日期往前推 months_window 个月
    today = pd.Timestamp.today().normalize()
    start_date_ts = today - pd.DateOffset(months=months_window)
    start_date_str = start_date_ts.strftime("%Y%m%d")

    df = _safe_baostock_history(
        symbol=code,
        period="monthly",
        start_date=start_date_str,
        end_date=None,
        adjust=adjust,
    )

    if df.empty:
        return df

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["close"] = df["close"].astype(float)
    df.sort_values("date", inplace=True)
    df.set_index("date", inplace=True)
    return df[["code", "close"]]


def is_interesting_stock(
    code: str,
    adjust: str = "qfq",
    months_window: int = 60,
    verbose: bool = True,
) -> dict:
    """
    根据最近两期 5 月线 / 30 月线的关系，判断该股票是否为“interesting stock”。

    判定条件：
      - ma5_last_month <= ma30_last_month
      - 且 ma5_this_month > ma30_this_month

    返回字典，包含：
      - interesting: bool
      - ma5: float (本月)
      - ma5_prev: float (上月)
      - ma10: float (本月)
      - ma10_prev: float (上月)
      - ma20: float (本月)
      - ma20_prev: float (上月)
      - ma30: float (本月)
      - ma30_prev: float (上月)
    """
    df = fetch_recent_monthly_closes(
        code=code,
        adjust=adjust,
        months_window=months_window,
    )

    n = len(df)
    if n < 31:
        if verbose:
            print(
                f"\n[is_interesting_stock] Not enough monthly data for {code}: "
                f"need at least 31 months, got {n}."
            )
        return {
            "interesting": False,
            "ma5_cross_ma30": False,
            "ma5up": None,
            "ma10up": None,
            "ma20up": None,
            "ma30up": None,
            "ma5": None,
            "ma5_prev": None,
            "ma10": None,
            "ma10_prev": None,
            "ma20": None,
            "ma20_prev": None,
            "ma30": None,
            "ma30_prev": None,
        }

    # 5 月线：本月 vs 上月
    ma5 = df["close"].iloc[-5:].mean()
    ma5_prev = df["close"].iloc[-6:-1].mean()

    # 10 月线：本月 vs 上月（需要至少 11 个月数据）
    ma10 = df["close"].iloc[-10:].mean() if n >= 11 else None
    ma10_prev = df["close"].iloc[-11:-1].mean() if n >= 11 else None

    # 20 月线：本月 vs 上月（需要至少 21 个月数据）
    ma20 = df["close"].iloc[-20:].mean() if n >= 21 else None
    ma20_prev = df["close"].iloc[-21:-1].mean() if n >= 21 else None

    # 30 月线：本月 vs 上月
    ma30 = df["close"].iloc[-30:].mean()
    ma30_prev = df["close"].iloc[-31:-1].mean()

    ma5_cross_ma30 = (ma5_prev <= ma30_prev) and (ma5 > ma30)

    # 计算 maXup：maX_today > maX_yesterday
    ma5up = ma5 > ma5_prev
    ma10up = (ma10 is not None and ma10_prev is not None) and (ma10 > ma10_prev)
    ma20up = (ma20 is not None and ma20_prev is not None) and (ma20 > ma20_prev)
    ma30up = ma30 > ma30_prev

    # interesting: 所有条件都必须为 True
    interesting = ma5_cross_ma30 and ma5up and ma10up and ma20up and ma30up

    if verbose:
        last_date = df.index[-1].date()
        prev_date = df.index[-2].date()
        print(f"\n[is_interesting_stock] {code} up to {last_date} / {prev_date}:")
        ma10_prev_str = f"{ma10_prev:.2f}" if ma10_prev is not None else "N/A"
        ma20_prev_str = f"{ma20_prev:.2f}" if ma20_prev is not None else "N/A"
        ma10_str = f"{ma10:.2f}" if ma10 is not None else "N/A"
        ma20_str = f"{ma20:.2f}" if ma20 is not None else "N/A"
        print(f"  ma5_prev={ma5_prev:.2f}, ma10_prev={ma10_prev_str}, ma20_prev={ma20_prev_str}, ma30_prev={ma30_prev:.2f}")
        print(f"  ma5={ma5:.2f}, ma10={ma10_str}, ma20={ma20_str}, ma30={ma30:.2f}")
        print(f"  ma5 cross ma30={ma5_cross_ma30}, ma5up={ma5up}, ma10up={ma10up}, ma20up={ma20up}, ma30up={ma30up}")
        print(f"  interesting={interesting}")

    return {
        "interesting": interesting,
        "ma5_cross_ma30": ma5_cross_ma30,
        "ma5up": ma5up,
        "ma10up": ma10up,
        "ma20up": ma20up,
        "ma30up": ma30up,
        "ma5": ma5,
        "ma5_prev": ma5_prev,
        "ma10": ma10,
        "ma10_prev": ma10_prev,
        "ma20": ma20,
        "ma20_prev": ma20_prev,
        "ma30": ma30,
        "ma30_prev": ma30_prev,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Compute 5-month and 30-month moving averages (5月线/30月线) "
            "for one or multiple A-shares using Baostock monthly data. "
            "The end date is always \"today\"; we fetch the latest ~60 months "
            "and compute 5-month / 30-month averages from that window."
        )
    )
    parser.add_argument(
        "--code",
        type=str,
        default=None,
        help=(
            'Single stock code without exchange prefix, e.g. "600519" for 贵州茅台. '
            "If omitted, codes will be read from --codes-file."
        ),
    )
    parser.add_argument(
        "--codes-file",
        type=str,
        default="a_share_codes.csv",
        help=(
            "CSV file path for the stock code:name universe "
            '(must contain at least a "code" column). '
            'Used when --code is not provided. Default: "a_share_codes.csv".'
        ),
    )
    parser.add_argument(
        "--max-stocks",
        type=int,
        default=None,
        help=(
            "When using --codes-file, only process the first N stocks "
            "from that file. Default: all stocks in the file."
        ),
    )
    args = parser.parse_args()

    # 情况 1：指定单只股票（--code），直接计算该股的 5 月线和 30 月线
    if args.code:
        lg = bs.login()
        if lg.error_code != "0":
            raise RuntimeError(f"Baostock login failed: {lg.error_code}, {lg.error_msg}")
        try:
            compute_monthly_ma(
                code=args.code,
            )
        finally:
            bs.logout()
    else:
        # 情况 2：未指定 --code，从 --codes-file 中读取代码列表，
        # 只处理前 max_stocks 只股票
        codes_df = get_target_share_codes(filepath=args.codes_file)
        if args.max_stocks is not None:
            codes_df = codes_df.head(args.max_stocks)

        print(
            f"Processing {len(codes_df)} stocks from {args.codes_file} "
            f"(max-stocks={args.max_stocks})."
        )

        results = []

        lg = bs.login()
        if lg.error_code != "0":
            raise RuntimeError(f"Baostock login failed: {lg.error_code}, {lg.error_msg}")

        try:
            for _, row in codes_df.iterrows():
                code = str(row["code"])
                name = str(row.get("name", ""))

                print(f"\n=== {code} {name} ===")
                result_dict = is_interesting_stock(code=code, verbose=True)

                results.append(
                    {
                        "code": code,
                        "name": name,
                        "interesting": result_dict["interesting"],
                        "ma5 cross ma30": result_dict["ma5_cross_ma30"],
                        "ma5up": result_dict["ma5up"],
                        "ma10up": result_dict["ma10up"],
                        "ma20up": result_dict["ma20up"],
                        "ma30up": result_dict["ma30up"],
                        "ma5": result_dict["ma5"],
                        "ma5_prev": result_dict["ma5_prev"],
                        "ma10": result_dict["ma10"],
                        "ma10_prev": result_dict["ma10_prev"],
                        "ma20": result_dict["ma20"],
                        "ma20_prev": result_dict["ma20_prev"],
                        "ma30": result_dict["ma30"],
                        "ma30_prev": result_dict["ma30_prev"],
                    }
                )
                # 适当放慢节奏，避免对远端服务器压力过大
                #time.sleep(0.3)
        finally:
            bs.logout()

        if results:
            today_str = pd.Timestamp.today().strftime("%Y%m%d")
            output_file = f"result_{today_str}.csv"
            df_results = pd.DataFrame(results)
            # 确保列顺序：code, name, interesting, ma5 cross ma30, ma5up, ma10up, ma20up, ma30up, ma5, ma5_prev, ma10, ma10_prev, ma20, ma20_prev, ma30, ma30_prev
            column_order = [
                "code",
                "name",
                "interesting",
                "ma5 cross ma30",
                "ma5up",
                "ma10up",
                "ma20up",
                "ma30up",
                "ma5",
                "ma5_prev",
                "ma10",
                "ma10_prev",
                "ma20",
                "ma20_prev",
                "ma30",
                "ma30_prev",
            ]
            df_results = df_results[column_order]
            df_results.to_csv(output_file, index=False)
            print(f"\nSaved {len(results)} rows to {output_file}")
        else:
            print("\nNo stocks processed; no result file generated.")

