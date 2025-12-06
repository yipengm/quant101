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
        RAW_DATA_DAILY = os.path.join(DATA_DIR, "all_daily_data.csv")
    else:
        RAW_DATA_MONTHLY = os.path.join(DATA_DIR, f"all_monthly_data_{selected_version}.csv")
        RAW_DATA_WEEKLY = os.path.join(DATA_DIR, f"all_weekly_data_{selected_version}.csv")
        RAW_DATA_DAILY = os.path.join(DATA_DIR, f"all_daily_data_{selected_version}.csv")
else:
    # Fallback
    RAW_DATA_MONTHLY = os.path.join(DATA_DIR, "all_monthly_data.csv")
    RAW_DATA_WEEKLY = os.path.join(DATA_DIR, "all_weekly_data.csv")
    RAW_DATA_DAILY = os.path.join(DATA_DIR, "all_daily_data.csv")


@st.cache_data
def load_data(monthly_path, weekly_path, daily_path):
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
        
    # Load Daily
    if os.path.exists(daily_path):
        df_d = pd.read_csv(daily_path, dtype={"code": str})
        df_d['date'] = pd.to_datetime(df_d['date'])
    else:
        df_d = None
    
    # Load Names
    codes_path = "data/codenameDB/a_share_codes.csv"
    code_map = {}
    if os.path.exists(codes_path):
        codes_df = pd.read_csv(codes_path, dtype={"code": str})
        code_map = dict(zip(codes_df['code'], codes_df['name']))
    
    if df_m is not None: df_m['name'] = df_m['code'].map(code_map)
    if df_w is not None: df_w['name'] = df_w['code'].map(code_map)
    if df_d is not None: df_d['name'] = df_d['code'].map(code_map)

    return df_m, df_w, df_d

df_monthly, df_weekly, df_daily = load_data(RAW_DATA_MONTHLY, RAW_DATA_WEEKLY, RAW_DATA_DAILY)

if df_monthly is None and df_weekly is None and df_daily is None:
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
cadence = st.sidebar.radio("Frequency", ["Monthly (MA)", "Weekly (WA)", "Daily (DA)"], key="input_cadence")

col1, col2 = st.sidebar.columns(2)
with col1:
    window_a = st.number_input("Window A", min_value=1, value=5, step=1, key="input_window_a")
    coeff_a = st.number_input("Coeff A (α)", value=1.0, step=0.01, key="input_coeff_a")

operation = st.sidebar.selectbox("Operation", ["Up", "Down", "Cross", "Greater Than", "Less Than"], key="input_op")

window_b = None
coeff_b = 1.0
if operation in ["Cross", "Greater Than", "Less Than"]:
    with col2:
        window_b = st.number_input("Window B", min_value=1, value=30, step=1, key="input_window_b")
        coeff_b = st.number_input("Coeff B (β)", value=1.0, step=0.01, key="input_coeff_b")

if st.sidebar.button("Add Rule"):
    if "Monthly" in cadence:
        rule_cadence = "monthly"
    elif "Weekly" in cadence:
        rule_cadence = "weekly"
    else:
        rule_cadence = "daily"
        
    new_rule = {
        "cadence": rule_cadence,
        "window_a": window_a,
        "coeff_a": coeff_a,
        "op": operation,
        "window_b": window_b,
        "coeff_b": coeff_b
    }
    st.session_state.rules.append(new_rule)
    st.toast("Rule added!", icon="✅")
    st.rerun()

# --- Sidebar: Active Rules ---
st.sidebar.divider()
st.sidebar.subheader("Active Rules")

if st.session_state.rules:
    for i, rule in enumerate(st.session_state.rules):
        if rule['cadence'] == 'monthly':
            prefix = "MA"
        elif rule['cadence'] == 'weekly':
            prefix = "WA"
        else:
            prefix = "DA"
        
        str_a = f"{rule['coeff_a']}*{prefix}({rule['window_a']})" if rule['coeff_a'] != 1.0 else f"{prefix}({rule['window_a']})"
        
        if rule['op'] in ['Cross', 'Greater Than', 'Less Than']:
            str_b = f"{rule['coeff_b']}*{prefix}({rule['window_b']})" if rule['coeff_b'] != 1.0 else f"{prefix}({rule['window_b']})"
            desc = f"{str_a} {rule['op']} {str_b}"
        else:
            desc = f"{str_a} {rule['op']}"
        
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
        ca = r.get('coeff_a', 1.0)
        cb = r.get('coeff_b', 1.0)
        
        col_a = f"{prefix}{wa}"
        col_a_prev = f"{prefix}{wa}_prev"
        
        # Base validation: A must exist
        valid_data = df_latest[col_a].notna() & df_latest[col_a_prev].notna()
        current_mask = valid_data
        
        # Values adjusted by coefficients
        val_a_curr = df_latest[col_a] * ca
        val_a_prev = df_latest[col_a_prev] * ca
        
        if op == "Up":
            # For Up/Down, we compare the adjusted value to its own previous adjusted value
            # (Technically ca cancels out if positive, but strict implementation uses it)
            current_mask &= (val_a_curr > val_a_prev)
        elif op == "Down":
            current_mask &= (val_a_curr < val_a_prev)
            
        elif op in ["Cross", "Greater Than", "Less Than"]:
            wb = r['window_b']
            col_b = f"{prefix}{wb}"
            col_b_prev = f"{prefix}{wb}_prev"
            
            # Validation: B must exist
            valid_data_b = df_latest[col_b].notna()
            if op == "Cross":
                valid_data_b &= df_latest[col_b_prev].notna()
            current_mask &= valid_data_b
            
            val_b_curr = df_latest[col_b] * cb
            val_b_prev = df_latest[col_b_prev] * cb if op == "Cross" else None
            
            if op == "Cross":
                # Cross Logic: (Prev A <= Prev B) AND (Curr A > Curr B)
                cross_logic = (val_a_prev <= val_b_prev) & (val_a_curr > val_b_curr)
                current_mask &= cross_logic
                
            elif op == "Greater Than":
                current_mask &= (val_a_curr > val_b_curr)
                
            elif op == "Less Than":
                current_mask &= (val_a_curr < val_b_curr)
        
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
            daily_rules = [r for r in st.session_state.rules if r['cadence'] == 'daily']
            
            valid_codes_m = None
            valid_codes_w = None
            valid_codes_d = None
            
            df_disp_m = pd.DataFrame()
            df_disp_w = pd.DataFrame()
            df_disp_d = pd.DataFrame()

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

            # Process Daily
            if daily_rules:
                if df_daily is None:
                    st.error("Daily data missing.")
                    valid_codes_d = set()
                else:
                    valid_codes_d, df_disp_d = compute_and_filter(df_daily, daily_rules, prefix="da")

            # Intersect
            # Logic: Start with the first non-None set, then intersect with others
            final_codes = None
            
            sets_to_intersect = []
            if valid_codes_m is not None: sets_to_intersect.append(valid_codes_m)
            if valid_codes_w is not None: sets_to_intersect.append(valid_codes_w)
            if valid_codes_d is not None: sets_to_intersect.append(valid_codes_d)
            
            if sets_to_intersect:
                final_codes = sets_to_intersect[0]
                for s in sets_to_intersect[1:]:
                    final_codes = final_codes.intersection(s)
            
            if final_codes:
                # Prepare display DF
                final_list = sorted(list(final_codes))
                res_df = pd.DataFrame({'code': final_list})
                
                # Merge names (Prefer Monthly, then Weekly, then Daily)
                name_source = df_monthly if df_monthly is not None else (df_weekly if df_weekly is not None else df_daily)
                if name_source is not None and not name_source.empty:
                     name_map = name_source.groupby('code')['name'].last()
                     res_df = res_df.merge(name_map, on='code', how='left')
                
                # Merge calculated metrics
                if not df_disp_m.empty:
                    cols_m = [c for c in df_disp_m.columns if c.startswith("ma")]
                    res_df = res_df.merge(df_disp_m[cols_m], on='code', how='left')
                    
                if not df_disp_w.empty:
                    cols_w = [c for c in df_disp_w.columns if c.startswith("wa")]
                    res_df = res_df.merge(df_disp_w[cols_w], on='code', how='left')

                if not df_disp_d.empty:
                    cols_d = [c for c in df_disp_d.columns if c.startswith("da")]
                    res_df = res_df.merge(df_disp_d[cols_d], on='code', how='left')
                
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
    if df_daily is not None:
        st.write(f"Daily Data: {len(df_daily)} rows")
