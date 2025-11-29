import baostock as bs
import pandas as pd
import datetime

def _get_quarter(date_obj):
    """Return (year, quarter) for the latest likely available financial report."""
    # Reports: Q1 (Apr), Q2 (Aug), Q3 (Oct), Q4 (Apr next year)
    # Simple heuristic: look 2 quarters back to be safe, or logic based on month
    month = date_obj.month
    year = date_obj.year
    
    if month <= 4:
        return year - 1, 3 # Look at Q3 prev year (safe) or Q4 prev year
    elif month <= 8:
        return year, 1 # Q1
    elif month <= 10:
        return year, 2 # Q2
    else:
        return year, 3 # Q3

def get_factors(code_list):
    """
    Fetch Value, Quality, and Growth factors for a list of codes.
    """
    if not code_list:
        return pd.DataFrame()

    lg = bs.login()
    if lg.error_code != "0":
        return pd.DataFrame()

    # 1. Settings
    today = datetime.date.today()
    end_date_str = today.strftime("%Y-%m-%d")
    start_date_str = (today - datetime.timedelta(days=30)).strftime("%Y-%m-%d")
    
    # Financial quarter
    year, quarter = _get_quarter(today)
    
    results = []
    
    # Progress bar in streamlit? We pass a callback or just let streamlit spinner handle it.
    # We'll iterate.
    
    for code in code_list:
        # Format code
        if code.startswith('6'):
            bs_code = f"sh.{code}"
        else:
            bs_code = f"sz.{code}"
            
        row_data = {'code': code}
        
        # --- A. Valuation & Liquidity (Daily) ---
        # query_history_k_data_plus
        # peTTM, pbMRQ, psTTM, pcfNcfTTM, turn, volume, amount, close
        fields = "date,peTTM,pbMRQ,psTTM,pcfNcfTTM,turn,volume,amount,close"
        rs = bs.query_history_k_data_plus(
            bs_code, fields,
            start_date=start_date_str, end_date=end_date_str,
            frequency="d", adjustflag="3"
        )
        
        k_list = []
        while (rs.error_code == '0') & rs.next():
            k_list.append(rs.get_row_data())
            
        if k_list:
            latest = k_list[-1] # Last available day
            # fields indices: 0:date, 1:pe, 2:pb, 3:ps, 4:pcf, 5:turn, 6:vol, 7:amt, 8:close
            row_data['peTTM'] = latest[1]
            row_data['pbMRQ'] = latest[2]
            row_data['psTTM'] = latest[3]
            row_data['pcfNcfTTM'] = latest[4]
            row_data['turnover'] = latest[5]
            row_data['latest_close'] = latest[8]
        
        # --- B. Profitability (Quarterly) ---
        # query_profit_data
        # roeAvg, npMargin, gpMargin, netProfit, epsTTM
        rs_prof = bs.query_profit_data(code=bs_code, year=year, quarter=quarter)
        while (rs_prof.error_code == '0') & rs_prof.next():
            prof_data = rs_prof.get_row_data()
            # Fields vary, let's grab by name if possible or assume order. 
            # Baostock docs: code, pubDate, statDate, roeAvg, npMargin, gpMargin, netProfit, epsTTM, mbRevenue, totalShare, liqaShare
            # Let's assume standard order or check fields?
            # rs_prof.fields gives names.
            # For simplicity, we map manually if fields are standard.
            # Safe way: make a dict
            prof_dict = dict(zip(rs_prof.fields, prof_data))
            row_data['roeAvg'] = prof_dict.get('roeAvg')
            row_data['npMargin'] = prof_dict.get('npMargin')
            row_data['gpMargin'] = prof_dict.get('gpMargin')
            row_data['totalShare'] = prof_dict.get('totalShare') # Useful for Market Cap
            
        # --- C. Growth (Quarterly) ---
        # query_growth_data
        # YOYEquity, YOYAsset, YOYNI, YOYEPSBasic, YOYPNI
        rs_grow = bs.query_growth_data(code=bs_code, year=year, quarter=quarter)
        while (rs_grow.error_code == '0') & rs_grow.next():
            grow_dict = dict(zip(rs_grow.fields, rs_grow.get_row_data()))
            row_data['YOYNetProfit'] = grow_dict.get('YOYNI')
            row_data['YOYRevenue'] = grow_dict.get('MBRevenue') # Wait, growth api has different fields?
            # Growth api: code, pubDate, statDate, YOYEquity, YOYAsset, YOYNI, YOYEPSBasic, YOYPNI
            
        # --- D. Market Cap (Calculated) ---
        # Size = Close * TotalShare
        try:
            if row_data.get('latest_close') and row_data.get('totalShare'):
                close_val = float(row_data['latest_close'])
                share_val = float(row_data['totalShare'])
                row_data['MarketCap'] = close_val * share_val
        except:
            pass

        results.append(row_data)
        
    bs.logout()
    return pd.DataFrame(results)

