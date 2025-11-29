"""
Simple example: fetch A-share daily historical prices using Baostock.

参考 Baostock 官方文档: `http://www.baostock.com/mainContent?file=home.md`

Before running:
    pip install -r scripts/requirements.txt

Then run:
    python src/get_akshare_sample.py
"""

from __future__ import annotations

import baostock as bs
import pandas as pd


def fetch_sample_history() -> None:
    """
    Fetch and print sample daily K-line data for an A-share stock using Baostock.

    Example: Kweichow Moutai (code "sh.600519").
    """
    # Adjust these parameters as you like
    code = "sh.600519"  # 贵州茅台，在 Baostock 里需要带交易所前缀
    start_date = "2020-01-01"
    end_date = "2025-12-31"

    # 登录系统
    lg = bs.login()
    if lg.error_code != "0":
        raise RuntimeError(f"Baostock login failed: {lg.error_code}, {lg.error_msg}")

    try:
        # 参考官网 query_history_k_data_plus 示例
        fields = ",".join(
            [
                "date",
                "code",
                "open",
                "high",
                "low",
                "close",
                "preclose",
                "volume",
                "amount",
                "adjustflag",  # 复权标志：1=后复权，2=前复权，3=不复权
            ]
        )

        rs = bs.query_history_k_data_plus(
            code,
            fields,
            start_date=start_date,
            end_date=end_date,
            frequency="d",
            adjustflag="2",  # 前复权
        )

        if rs.error_code != "0":
            raise RuntimeError(f"Baostock query failed: {rs.error_code}, {rs.error_msg}")

        data_list = []
        while rs.error_code == "0" and rs.next():
            data_list.append(rs.get_row_data())

        df = pd.DataFrame(data_list, columns=rs.fields)

    finally:
        # 登出系统
        bs.logout()

    print(f"Fetched {len(df)} rows for {code} from {start_date} to {end_date}")
    print("Head:")
    print(df.head())

    # Optional: save to CSV for later use
    import os
    output_dir = "data/raw"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_file = f"{output_dir}/{code.replace('.', '')}_daily_{start_date.replace('-', '')}_{end_date.replace('-', '')}.csv"
    df.to_csv(output_file, index=False)
    print(f"\nData saved to: {output_file}")


if __name__ == "__main__":
    fetch_sample_history()
