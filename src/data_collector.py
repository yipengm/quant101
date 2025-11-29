"""
Data Collector: Fetch raw monthly K-line data for all target stocks.
Saves to data/raw/all_monthly_data.csv.
"""
import os
import time
import pandas as pd
import baostock as bs
import akshare as ak
from tqdm import tqdm
from typing import Optional

# Reuse robust fetching logic
def _safe_baostock_history(
    *,
    symbol: str,
    period: str,
    start_date: str,
    end_date: Optional[str],
    adjust: str,
    max_retries: int = 3,
    base_sleep: float = 0.5, # Faster sleep for batch
) -> pd.DataFrame:
    
    if symbol.startswith("6"):
        bs_code = f"sh.{symbol}"
    else:
        bs_code = f"sz.{symbol}"

    # 20200101 -> 2020-01-01
    def _fmt(d: Optional[str]) -> Optional[str]:
        if d is None or d == "": return None
        if "-" in d: return d
        return f"{d[0:4]}-{d[4:6]}-{d[6:8]}"

    start_d = _fmt(start_date)
    end_d = _fmt(end_date) or time.strftime("%Y-%m-%d")

    # period map
    if period == "monthly": frequency = "m"
    elif period == "weekly": frequency = "w"
    else: frequency = "d"

    # adjust map: qfq=2, hfq=1, else=3
    if adjust == "qfq": adjustflag = "2"
    elif adjust == "hfq": adjustflag = "1"
    else: adjustflag = "3"

    fields = "date,code,open,high,low,close,volume,amount,adjustflag"

    last_err = None
    for i in range(max_retries):
        try:
            rs = bs.query_history_k_data_plus(
                bs_code, fields,
                start_date=start_d, end_date=end_d,
                frequency=frequency, adjustflag=adjustflag
            )
            if rs.error_code != "0":
                raise RuntimeError(f"Baostock error: {rs.error_code} {rs.error_msg}")
            
            data_list = []
            while rs.error_code == "0" and rs.next():
                data_list.append(rs.get_row_data())
            
            return pd.DataFrame(data_list, columns=rs.fields)
        except Exception as e:
            last_err = e
            time.sleep(base_sleep * (i + 1))
    
    if last_err: raise last_err
    return pd.DataFrame()

def get_target_codes(filepath: str) -> list[str]:
    if os.path.exists(filepath):
        df = pd.read_csv(filepath, dtype={"code": str})
        return df["code"].tolist()
    else:
        # Fallback: fetch all A shares
        print("Codes file not found, fetching all A-shares from AkShare...")
        df = ak.stock_info_a_code_name()
        return df["code"].tolist()

def collect_data(codes_file: str, output_file: str, months: int = 60, max_n: Optional[int] = None):
    codes = get_target_codes(codes_file)
    if max_n is not None:
        codes = codes[:max_n]
        print(f"Debug mode: limiting to first {max_n} stocks.")
    
    print(f"Collecting data for {len(codes)} stocks...")

    # Calculate start date
    today = pd.Timestamp.today().normalize()
    start_date_ts = today - pd.DateOffset(months=months)
    start_date_str = start_date_ts.strftime("%Y%m%d")

    all_data = []
    
    lg = bs.login()
    if lg.error_code != "0":
        print(f"Login failed: {lg.error_msg}")
        return

    try:
        pbar = tqdm(codes)
        for code in pbar:
            pbar.set_description(f"Fetching {code}")
            try:
                df = _safe_baostock_history(
                    symbol=code,
                    period="monthly",
                    start_date=start_date_str,
                    end_date=None,
                    adjust="qfq"
                )
                if not df.empty:
                    # Keep clean code without prefix for consistency if needed
                    # Baostock returns sh.600519, we might want just 600519
                    # But let's keep raw for now or strip 'sh.'/'sz.'? 
                    # Your stock_filter.py uses raw codes (600519).
                    # Baostock result 'code' column has 'sh.600519'.
                    # Let's standardize to 600519 to match your codes file.
                    df["code"] = df["code"].apply(lambda x: x.split(".")[-1])
                    
                    # Convert numeric columns
                    cols = ["open", "high", "low", "close", "volume", "amount"]
                    for c in cols:
                        df[c] = pd.to_numeric(df[c])
                        
                    all_data.append(df)
            except Exception as e:
                pass # Skip error stocks
            
            # Rate limit slightly
            # time.sleep(0.05) 
    finally:
        bs.logout()

    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        # Sort by code and date
        final_df["date"] = pd.to_datetime(final_df["date"])
        final_df = final_df.sort_values(["code", "date"])
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        final_df.to_csv(output_file, index=False)
        print(f"Saved {len(final_df)} rows to {output_file}")
    else:
        print("No data collected.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--codes-file", default="data/codenameDB/a_share_codes.csv")
    parser.add_argument("--output-file", default="data/raw/all_monthly_data.csv")
    parser.add_argument("--months", type=int, default=60)
    parser.add_argument("--max-stocks", type=int, default=None, help="Debug: limit number of stocks")
    args = parser.parse_args()

    # Quick hack to slice codes if max-stocks is set (by modifying get_target_codes behavior or just slicing list)
    # I'll just do it here manually if needed or let user control via file.
    # Actually, let's implement the slicing logic in main for safety.
    
    codes = get_target_codes(args.codes_file)
    if args.max_stocks:
        codes = codes[:args.max_stocks]
        # Monkey patch the get function or just pass codes?
        # Easier to refactor collect_data to accept list of codes.
    
    # Refactoring collect_data to take list of codes would be cleaner, but for now I'll just overwrite the file read logic?
    # No, I'll just call the function as is, and if max-stocks is needed I'll rely on user providing a small file OR
    # I will duplicate logic slightly. Let's just run it.
    
    # Re-implementing call to support max_stocks properly
    # Let's modify collect_data signature slightly in next step if needed, but for now
    # I'll stick to the requested functionality.
    
    collect_data(args.codes_file, args.output_file, args.months, args.max_stocks)

