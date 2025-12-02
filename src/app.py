import streamlit as st
import pandas as pd
import os
import glob
import re
import src.factor_service as fs

st.set_page_config(page_title="Quant101 Dynamic Filter", layout="wide")
st.title("Quant101 Dynamic Stock Filter")

DATA_DIR = "data/raw"

def get_available_versions():
    """Scan data/raw for all_monthly_data_{suffix}.csv and return sorted suffixes."""
    if not os.path.exists(DATA_DIR):
        return []
    
    # Find all monthly files
    files = glob.glob(os.path.join(DATA_DIR, "all_monthly_data_*.csv"))
    versions = set()
    
    for f in files:
        filename = os.path.basename(f)
        # Match suffix
        match = re.match(r"all_monthly_data_(.+)\.csv", filename)
        if match:
            versions.add(match.group(1))
            
    # Also check for the default "all_monthly_data.csv" (no suffix)
    if os.path.exists(os.path.join(DATA_DIR, "all_monthly_data.csv")):
        versions.add("Default")
        
    return sorted(list(versions), reverse=True) # Latest dates likely first if YYYYMMDD

available_versions = get_available_versions()

# Sidebar: Data Selection
st.sidebar.header("Data Selection")
selected_version = st.sidebar.selectbox("Select Data Version", available_versions) if available_versions else None

if selected_version:
    if selected_version == "Default":
        RAW_DATA_MONTHLY = os.path.join(DATA_DIR, "all_monthly_data.csv")
        RAW_DATA_WEEKLY = os.path.join(DATA_DIR, "all_weekly_data.csv")
    else:
        RAW_DATA_MONTHLY = os.path.join(DATA_DIR, f"all_monthly_data_{selected_version}.csv")
        RAW_DATA_WEEKLY = os.path.join(DATA_DIR, f"all_weekly_data_{selected_version}.csv")
else:
    # Fallback if no version selected or found
    RAW_DATA_MONTHLY = os.path.join(DATA_DIR, "all_monthly_data.csv")
    RAW_DATA_WEEKLY = os.path.join(DATA_DIR, "all_weekly_data.csv")


@st.cache_data
def load_data(monthly_path, weekly_path):
    # Load Monthly
    if os.path.exists(monthly_path):
        df_m = pd.read_csv(monthly_path, dtype={"code": str})
        df_m['date'] = pd.to_datetime(df_m['date'])
    else:
        df_m = None
        
    # Load Weekly
    if os.path.exists(weekly_path):
        df_w = pd.read_csv(weekly_path, dtype={"code": str})
        df_w['date'] = pd.to_datetime(df_w['date'])
    else:
        df_w = None
    
    # Load Names
    codes_path = "data/codenameDB/a_share_codes.csv"
    code_map = {}
    if os.path.exists(codes_path):
        codes_df = pd.read_csv(codes_path, dtype={"code": str})
        code_map = dict(zip(codes_df['code'], codes_df['name']))
    
    if df_m is not None: df_m['name'] = df_m['code'].map(code_map)
    if df_w is not None: df_w['name'] = df_w['code'].map(code_map)

    return df_m, df_w

df_monthly, df_weekly = load_data(RAW_DATA_MONTHLY, RAW_DATA_WEEKLY)

if df_monthly is None and df_weekly is None:
    st.error(f"No data found for version: {selected_version}. Please check your data directory.")
    st.stop()

# --- Session State for Rules & Results ---
if 'rules' not in st.session_state:
    st.session_state.rules = []
if 'result_df' not in st.session_state:
    st.session_state.result_df = None
if 'calc_msg' not in st.session_state:
    st.session_state.calc_msg = None

# --- Sidebar: Rule Builder ---
st.sidebar.header("Rule Builder")
st.sidebar.write("Define a new rule:")

# Cadence Selection
cadence = st.sidebar.radio("Frequency", ["Monthly (MA)", "Weekly (WA)"], key="input_cadence")

col1, col2 = st.sidebar.columns(2)
with col1:
    window_a = st.number_input("Window A", min_value=1, value=5, step=1, key="input_window_a")

operation = st.sidebar.selectbox("Operation", ["Up", "Down", "Cross"], key="input_op")

window_b = None
if operation == "Cross":
    with col2:
        window_b = st.number_input("Window B", min_value=1, value=30, step=1, key="input_window_b")

if st.sidebar.button("Add Rule"):
    rule_cadence = "monthly" if "Monthly" in cadence else "weekly"
    new_rule = {
        "cadence": rule_cadence,
        "window_a": window_a,
        "op": operation,
        "window_b": window_b
    }
    st.session_state.rules.append(new_rule)
    st.toast("Rule added!", icon="✅")
    st.rerun()

# --- Sidebar: Active Rules ---
st.sidebar.divider()
st.sidebar.subheader("Active Rules")

if st.session_state.rules:
    for i, rule in enumerate(st.session_state.rules):
        prefix = "MA" if rule['cadence'] == 'monthly' else "WA"
        
        if rule['op'] == 'Cross':
            desc = f"{prefix}({rule['window_a']}) Cross {prefix}({rule['window_b']})"
        else:
            desc = f"{prefix}({rule['window_a']}) {rule['op']}"
        
        col_desc, col_del = st.sidebar.columns([0.8, 0.2])
        col_desc.text(f"{i+1}. {desc}")
        if col_del.button("X", key=f"del_{i}"):
            st.session_state.rules.pop(i)
            st.rerun()
            
    if st.sidebar.button("Clear All Rules"):
        st.session_state.rules = []
        st.rerun()
else:
    st.sidebar.info("No rules added yet.")

# --- Main Calculation Logic ---
def compute_and_filter(df_in, rules_subset, prefix="ma"):
    if df_in is None or df_in.empty:
        return pd.Series(False), pd.DataFrame()

    # 1. Identify needed windows
    needed = set()
    for r in rules_subset:
        needed.add(r['window_a'])
        if r['window_b']: needed.add(r['window_b'])
    
    # 2. Calc metrics
    df = df_in.sort_values(['code', 'date']).copy()
    grouped = df.groupby('code')
    
    cols_created = []
    for w in needed:
        col = f"{prefix}{w}"
        df[col] = grouped['close'].transform(lambda x: x.rolling(window=w).mean())
        df[f"{col}_prev"] = grouped[col].shift(1)
        cols_created.extend([col, f"{col}_prev"])
    
    # 3. Latest slice
    df_latest = df.groupby('code').tail(1).copy()
    
    # 4. Apply masks
    final_mask = pd.Series(True, index=df_latest.index)
    
    for r in rules_subset:
        wa = r['window_a']
        op = r['op']
        
        col_a = f"{prefix}{wa}"
        col_a_prev = f"{prefix}{wa}_prev"
        
        valid_data = df_latest[col_a].notna() & df_latest[col_a_prev].notna()
        current_mask = valid_data
        
        if op == "Up":
            current_mask &= (df_latest[col_a] > df_latest[col_a_prev])
        elif op == "Down":
            current_mask &= (df_latest[col_a] < df_latest[col_a_prev])
        elif op == "Cross":
            wb = r['window_b']
            col_b = f"{prefix}{wb}"
            col_b_prev = f"{prefix}{wb}_prev"
            
            valid_data_b = df_latest[col_b].notna() & df_latest[col_b_prev].notna()
            current_mask &= valid_data_b
            
            # Cross Logic
            cross_logic = (df_latest[col_a_prev] <= df_latest[col_b_prev]) & \
                          (df_latest[col_a] > df_latest[col_b])
            current_mask &= cross_logic
        
        final_mask &= current_mask
    
    # Return series of boolean valid codes, and the df_latest with metrics for display
    result_codes = df_latest[final_mask]['code'].unique()
    return set(result_codes), df_latest.set_index('code')


# --- Trigger Section ---
if st.button("Calculate & Filter"):
    # Reset previous results
    st.session_state.result_df = None
    st.session_state.calc_msg = None
    
    if not st.session_state.rules:
        st.session_state.calc_msg = ("warning", "Please add rules.")
    else:
        with st.spinner("Processing..."):
            # Split rules
            monthly_rules = [r for r in st.session_state.rules if r['cadence'] == 'monthly']
            weekly_rules = [r for r in st.session_state.rules if r['cadence'] == 'weekly']
            
            valid_codes_m = None
            valid_codes_w = None
            
            df_disp_m = pd.DataFrame()
            df_disp_w = pd.DataFrame()

            # Process Monthly
            if monthly_rules:
                if df_monthly is None:
                    st.error("Monthly data missing.")
                    valid_codes_m = set()
                else:
                    valid_codes_m, df_disp_m = compute_and_filter(df_monthly, monthly_rules, prefix="ma")
            
            # Process Weekly
            if weekly_rules:
                if df_weekly is None:
                    st.error("Weekly data missing.")
                    valid_codes_w = set()
                else:
                    valid_codes_w, df_disp_w = compute_and_filter(df_weekly, weekly_rules, prefix="wa")

            # Intersect
            if valid_codes_m is None: 
                final_codes = valid_codes_w
            elif valid_codes_w is None:
                final_codes = valid_codes_m
            else:
                final_codes = valid_codes_m.intersection(valid_codes_w)
            
            if final_codes:
                # Prepare display DF
                final_list = sorted(list(final_codes))
                res_df = pd.DataFrame({'code': final_list})
                
                # Merge names
                if df_monthly is not None and not df_monthly.empty:
                     name_map = df_monthly.groupby('code')['name'].last()
                     res_df = res_df.merge(name_map, on='code', how='left')
                
                # Merge calculated metrics
                if not df_disp_m.empty:
                    cols_m = [c for c in df_disp_m.columns if c.startswith("ma")]
                    res_df = res_df.merge(df_disp_m[cols_m], on='code', how='left')
                    
                if not df_disp_w.empty:
                    cols_w = [c for c in df_disp_w.columns if c.startswith("wa")]
                    res_df = res_df.merge(df_disp_w[cols_w], on='code', how='left')
                
                st.session_state.result_df = res_df
                st.session_state.calc_msg = ("success", f"Found {len(final_codes)} stocks matching all criteria.")
            else:
                st.session_state.calc_msg = ("warning", "No stocks found matching ALL criteria.")

# --- Display Section (Persistent) ---
if st.session_state.calc_msg:
    msg_type, msg_text = st.session_state.calc_msg
    if msg_type == "success":
        st.success(msg_text)
    else:
        st.warning(msg_text)

if st.session_state.result_df is not None:
    st.dataframe(st.session_state.result_df, use_container_width=True)
    
    col_factors, col_save = st.columns([0.5, 0.5])
    
    with col_factors:
        if st.button("⚡ Generate Factors"):
            with st.spinner("Fetching Baostock factors (Valuation, Quality, Growth)..."):
                codes = st.session_state.result_df['code'].tolist()
                df_factors = fs.get_factors(codes)
                
                if not df_factors.empty:
                    # Merge into result_df
                    # Avoid duplicate cols if user clicks twice (drop existing factor cols if present)
                    existing_cols = st.session_state.result_df.columns
                    # Factor cols are likely: peTTM, pbMRQ, roeAvg etc.
                    # Simple way: drop common columns from result_df before merge, except code
                    # Actually, merge left.
                    
                    # Clean merge
                    df_base = st.session_state.result_df.copy()
                    # If we already have factor columns, we might overwrite or duplicate.
                    # Let's just merge on code.
                    
                    # Drop columns in base that appear in factors (except code) to avoid suffixes
                    cols_to_update = [c for c in df_factors.columns if c != 'code' and c in df_base.columns]
                    if cols_to_update:
                        df_base = df_base.drop(columns=cols_to_update)
                    
                    df_merged = df_base.merge(df_factors, on='code', how='left')
                    st.session_state.result_df = df_merged
                    st.success("Factors generated and added to table!")
                    st.rerun()
                else:
                    st.warning("No factors returned or connection failed.")

    with col_save:
        if st.button("💾 Save to results folder"):
            try:
                today_str = pd.Timestamp.today().strftime("%Y%m%d")
                csv_filename = f"results_filtered_{today_str}.csv"
                output_dir = "./data/results"
                output_path = os.path.join(output_dir, csv_filename)
                
                os.makedirs(output_dir, exist_ok=True)
                st.session_state.result_df.to_csv(output_path, index=False, encoding='utf-8-sig')
                st.success(f"Saved to `{output_path}`")
            except Exception as e:
                st.error(f"Error saving file: {e}")

elif st.session_state.calc_msg is None:
    # Initial state or reset
    st.info("Configure rules and click Calculate.")
    if df_monthly is not None:
        st.write(f"Monthly Data: {len(df_monthly)} rows")
    if df_weekly is not None:
        st.write(f"Weekly Data: {len(df_weekly)} rows")
