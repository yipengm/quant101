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
from concurrent.futures import ProcessPoolExecutor, as_completed

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

def fetch_batch_worker(codes_subset, period, start_date):
    """
    Worker process to fetch a batch of codes.
    """
    # Suppress login output if possible, or just let it run
    bs.login()
    batch_data = []
    for code in codes_subset:
        try:
            df = _safe_baostock_history(
                symbol=code,
                period=period,
                start_date=start_date,
                end_date=None,
                adjust="qfq"
            )
            if not df.empty:
                # Standardize code format (remove sh./sz.)
                df["code"] = df["code"].apply(lambda x: x.split(".")[-1])
                
                # Convert numeric columns
                cols = ["open", "high", "low", "close", "volume", "amount"]
                for c in cols:
                    if c in df.columns:
                        df[c] = pd.to_numeric(df[c])
                
                batch_data.append(df)
        except Exception:
            pass # Skip error stocks
    bs.logout()
    return batch_data

def collect_data_batch(codes: list[str], output_file: str, period: str = "monthly", start_date: str = "20200101", batch_size: int = 100, max_workers: int = 4):
    print(f"Collecting {period} data for {len(codes)} stocks (Batch Size: {batch_size}, Workers: {max_workers})...")
    
    all_data = []
    
    # Chunk the codes
    chunks = [codes[i:i + batch_size] for i in range(0, len(codes), batch_size)]
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = {executor.submit(fetch_batch_worker, chunk, period, start_date): chunk for chunk in chunks}
        
        # Progress bar
        with tqdm(total=len(codes), desc=f"Fetching {period}") as pbar:
            for future in as_completed(futures):
                chunk = futures[future]
                try:
                    results = future.result()
                    all_data.extend(results)
                except Exception as e:
                    print(f"Batch failed: {e}")
                finally:
                    pbar.update(len(chunk))

    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        final_df["date"] = pd.to_datetime(final_df["date"])
        final_df = final_df.sort_values(["code", "date"])
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        final_df.to_csv(output_file, index=False)
        print(f"Saved {len(final_df)} rows to {output_file}")
    else:
        print(f"No {period} data collected.")

def collect_all(codes_file: str, max_n: Optional[int] = None, batch_size: int = 100):
    # Legacy wrapper
    # This is only kept if other scripts import it, but the logic is mainly in __main__ now
    # We'll just pass through to main logic if needed, or leave as is for reference
    pass

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--codes-file", default="data/codenameDB/a_share_codes.csv")
    parser.add_argument("--max-stocks", type=int, default=None, help="Debug: limit number of stocks")
    parser.add_argument("--batch-size", type=int, default=100, help="Number of stocks per batch")
    parser.add_argument("--max-workers", type=int, default=4, help="Number of parallel workers")
    
    # Filter flags
    parser.add_argument("--monthly", action="store_true", help="Collect monthly data")
    parser.add_argument("--weekly", action="store_true", help="Collect weekly data")
    parser.add_argument("--daily", action="store_true", help="Collect daily data")
    
    args = parser.parse_args()

    # If no specific flags are set, collect all
    collect_all_types = not (args.monthly or args.weekly or args.daily)
    
    codes = get_target_codes(args.codes_file)
    if args.max_stocks is not None:
        codes = codes[:args.max_stocks]
        print(f"Debug mode: limiting to first {args.max_stocks} stocks.")

    today = pd.Timestamp.today().normalize()
    
    # 1. Monthly
    if collect_all_types or args.monthly:
        start_date_monthly = (today - pd.DateOffset(months=60)).strftime("%Y%m%d")
        collect_data_batch(codes, "data/raw/all_monthly_data.csv", "monthly", start_date_monthly, 
                        batch_size=args.batch_size, max_workers=args.max_workers)
    
    # 2. Weekly
    if collect_all_types or args.weekly:
        start_date_weekly = (today - pd.DateOffset(weeks=160)).strftime("%Y%m%d")
        collect_data_batch(codes, "data/raw/all_weekly_data.csv", "weekly", start_date_weekly, 
                        batch_size=args.batch_size, max_workers=args.max_workers)

    # 3. Daily
    if collect_all_types or args.daily:
        start_date_daily = (today - pd.DateOffset(days=60)).strftime("%Y%m%d")
        collect_data_batch(codes, "data/raw/all_daily_data.csv", "daily", start_date_daily, 
                        batch_size=args.batch_size, max_workers=args.max_workers)
