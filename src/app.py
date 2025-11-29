import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="Quant101 Dynamic Filter", layout="wide")
st.title("Quant101 Dynamic Stock Filter")

RAW_DATA_PATH = "data/raw/all_monthly_data.csv"

@st.cache_data
def load_raw_data(filepath):
    if not os.path.exists(filepath):
        return None
    # Read only necessary columns to save memory if file is huge
    df = pd.read_csv(filepath, dtype={"code": str})
    df['date'] = pd.to_datetime(df['date'])
    
    # Load stock names if available
    codes_path = "data/codenameDB/a_share_codes.csv"
    if os.path.exists(codes_path):
        codes_df = pd.read_csv(codes_path, dtype={"code": str})
        code_map = dict(zip(codes_df['code'], codes_df['name']))
        df['name'] = df['code'].map(code_map)
    else:
        df['name'] = ""
        
    return df

df_raw = load_raw_data(RAW_DATA_PATH)

if df_raw is None:
    st.error(f"Raw data file not found at `{RAW_DATA_PATH}`.")
    st.info("Please run `python src/data_collector.py` to fetch data first.")
    st.stop()

st.sidebar.header("Dynamic Strategy Settings")

# 1. Select MAs to calculate
ma_options = [5, 10, 20, 30, 60]
selected_mas = st.sidebar.multiselect(
    "Select MAs to calculate & require Trending Up",
    options=ma_options,
    default=[5, 30]
)

# 2. Special Conditions
check_golden_cross = st.sidebar.checkbox("Require MA5 > MA30 (Golden Cross condition)", value=True)

if st.sidebar.button("Calculate & Filter"):
    with st.spinner("Calculating metrics on raw data..."):
        # Processing
        # We need to process by group. For performance on large df, 
        # sorting and then rolling is essential.
        
        df = df_raw.sort_values(['code', 'date']).copy()
        
        # Calculate MAs
        # To avoid slow groupby().apply(), we can use transform or direct rolling 
        # if we ensure boundaries (but rolling across groups is dangerous).
        # Safe way: groupby code
        
        grouped = df.groupby('code')
        
        # Calculate selected MAs
        # We always calculate 5 and 30 if golden cross is checked
        mas_to_calc = set(selected_mas)
        if check_golden_cross:
            mas_to_calc.add(5)
            mas_to_calc.add(30)
            
        for ma in mas_to_calc:
            col_name = f"ma{ma}"
            # transform is faster than apply for this
            df[col_name] = grouped['close'].transform(lambda x: x.rolling(window=ma).mean())
            
            # Calculate Up trend (slope)
            # Slope > 0 means Current > Previous
            # We need previous value
            df[f"{col_name}_prev"] = grouped[col_name].shift(1)
            df[f"{col_name}_up"] = df[col_name] > df[f"{col_name}_prev"]

        # Now we only care about the LATEST data point for each stock
        # Get last row per group
        df_latest = df.groupby('code').tail(1).copy()
        
        # Apply Filters
        filter_mask = pd.Series(True, index=df_latest.index)
        
        # Filter 1: Trending Up
        for ma in selected_mas:
            col_up = f"ma{ma}_up"
            # We must ensure it's not NaN (enough data)
            has_data = df_latest[f"ma{ma}"].notna()
            filter_mask = filter_mask & has_data & df_latest[col_up]
            
        # Filter 2: Golden Cross (MA5 > MA30 AND MA5_prev <= MA30_prev) 
        # User logic in prev chat was: ma5_prev <= ma30_prev AND ma5 > ma30
        if check_golden_cross:
            cond_cross = (df_latest['ma5_prev'] <= df_latest['ma30_prev']) & \
                         (df_latest['ma5'] > df_latest['ma30'])
            filter_mask = filter_mask & cond_cross
            
        result_df = df_latest[filter_mask].copy()
        
        # Formatting output
        cols_to_show = ['code', 'name', 'date', 'close']
        # Add relevant metric columns
        for ma in sorted(list(mas_to_calc)):
            cols_to_show.extend([f"ma{ma}", f"ma{ma}_prev", f"ma{ma}_up"])
            
        st.success(f"Found {len(result_df)} stocks matching criteria.")
        st.dataframe(result_df[cols_to_show].reset_index(drop=True), use_container_width=True)
else:
    st.info("Select metrics sidebar and click 'Calculate & Filter' to start.")
    st.write("Raw Data Preview:", df_raw.head())
