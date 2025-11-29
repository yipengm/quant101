import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="Quant101 Dynamic Filter", layout="wide")
st.title("Quant101 Dynamic Stock Filter")

RAW_DATA_MONTHLY = "data/raw/all_monthly_data.csv"
RAW_DATA_WEEKLY = "data/raw/all_weekly_data.csv"

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
    st.error(f"No data found. Please run the data collector.")
    st.stop()

# --- Session State for Rules ---
if 'rules' not in st.session_state:
    st.session_state.rules = []

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


if st.button("Calculate & Filter"):
    if not st.session_state.rules:
        st.warning("Please add rules.")
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
                st.success(f"Found {len(final_codes)} stocks matching all criteria.")
                
                # Prepare display DF
                # We primarily show Monthly info, plus Weekly info if available
                final_list = sorted(list(final_codes))
                
                # Fetch row data for display
                res_df = pd.DataFrame({'code': final_list})
                
                # Merge names
                if df_monthly is not None and not df_monthly.empty:
                     # Get name map from monthly df
                     name_map = df_monthly.groupby('code')['name'].last()
                     res_df = res_df.merge(name_map, on='code', how='left')
                
                # Merge calculated metrics
                if not df_disp_m.empty:
                    cols_m = [c for c in df_disp_m.columns if c.startswith("ma")]
                    res_df = res_df.merge(df_disp_m[cols_m], on='code', how='left')
                    
                if not df_disp_w.empty:
                    cols_w = [c for c in df_disp_w.columns if c.startswith("wa")]
                    res_df = res_df.merge(df_disp_w[cols_w], on='code', how='left')
                
                st.dataframe(res_df, use_container_width=True)
                
            else:
                st.warning("No stocks found matching ALL criteria.")
else:
    st.info("Configure rules and click Calculate.")
    if df_monthly is not None:
        st.write(f"Monthly Data: {len(df_monthly)} rows")
    if df_weekly is not None:
        st.write(f"Weekly Data: {len(df_weekly)} rows")
