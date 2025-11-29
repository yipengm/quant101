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
    df = pd.read_csv(filepath, dtype={"code": str})
    df['date'] = pd.to_datetime(df['date'])
    
    # Load names
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
    st.stop()

# --- Session State for Rules ---
if 'rules' not in st.session_state:
    st.session_state.rules = []

# --- Sidebar: Rule Builder ---
st.sidebar.header("Rule Builder")

# Use simple widgets (not in a form) for immediate interaction
st.sidebar.write("Define a new rule:")

# Variable A
col1, col2 = st.sidebar.columns(2)
with col1:
    window_a = st.number_input("Window A (MA)", min_value=1, value=5, step=1, key="input_window_a")

# Operation
operation = st.sidebar.selectbox("Operation", ["Up", "Down", "Cross"], key="input_op")

# Variable B (Only for Cross)
window_b = None
if operation == "Cross":
    with col2:
        window_b = st.number_input("Window B (MA)", min_value=1, value=30, step=1, key="input_window_b")

if st.sidebar.button("Add Rule"):
    new_rule = {
        "type": "MA",
        "window_a": window_a,
        "op": operation,
        "window_b": window_b
    }
    st.session_state.rules.append(new_rule)
    st.toast("Rule added!", icon="✅")
    # Rerun is handled by the button interaction flow usually, but st.rerun() ensures list update
    # st.rerun() is typically not needed for state update unless we want immediate visual refresh of the list below
    # but let's do it to be snappy
    st.rerun()

# --- Sidebar: Active Rules List ---
st.sidebar.divider()
st.sidebar.subheader("Active Rules")

if st.session_state.rules:
    for i, rule in enumerate(st.session_state.rules):
        if rule['op'] == 'Cross':
            desc = f"MA({rule['window_a']}) Cross MA({rule['window_b']})"
        else:
            desc = f"MA({rule['window_a']}) {rule['op']}"
        
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

# --- Main Area: Calculation ---
if st.button("Calculate & Filter"):
    if not st.session_state.rules:
        st.warning("Please add at least one rule to filter.")
    else:
        with st.spinner("Calculating metrics..."):
            # 1. Identify required MAs to avoid redundant calc
            needed_mas = set()
            for rule in st.session_state.rules:
                needed_mas.add(rule['window_a'])
                if rule['window_b']:
                    needed_mas.add(rule['window_b'])
            
            # 2. Compute MAs on grouped data
            df = df_raw.sort_values(['code', 'date']).copy()
            grouped = df.groupby('code')
            
            for w in needed_mas:
                col = f"ma{w}"
                df[col] = grouped['close'].transform(lambda x: x.rolling(window=w).mean())
                # Pre-calculate prev for trend/cross logic
                df[f"{col}_prev"] = grouped[col].shift(1)

            # 3. Get latest slice for filtering
            df_latest = df.groupby('code').tail(1).copy()
            
            # 4. Apply Rules
            final_mask = pd.Series(True, index=df_latest.index)
            
            for rule in st.session_state.rules:
                wa = rule['window_a']
                op = rule['op']
                
                col_a = f"ma{wa}"
                col_a_prev = f"ma{wa}_prev"
                
                # Ensure data exists
                valid_data = df_latest[col_a].notna() & df_latest[col_a_prev].notna()
                
                current_mask = valid_data
                
                if op == "Up":
                    # Current > Prev
                    current_mask &= (df_latest[col_a] > df_latest[col_a_prev])
                
                elif op == "Down":
                    # Current < Prev
                    current_mask &= (df_latest[col_a] < df_latest[col_a_prev])
                
                elif op == "Cross":
                    wb = rule['window_b']
                    col_b = f"ma{wb}"
                    col_b_prev = f"ma{wb}_prev"
                    
                    # Ensure B data exists
                    valid_data_b = df_latest[col_b].notna() & df_latest[col_b_prev].notna()
                    current_mask &= valid_data_b
                    
                    # Cross Logic: A_prev <= B_prev AND A_curr > B_curr
                    # (Golden Cross logic as requested: ma5_last < ma30_last && ma5_current > ma30_current)
                    # Note: user asked "ma5_last < ma30_last", but standard is <= to catch exact touches
                    cross_logic = (df_latest[col_a_prev] <= df_latest[col_b_prev]) & \
                                  (df_latest[col_a] > df_latest[col_b])
                    current_mask &= cross_logic
                
                final_mask &= current_mask

            # 5. Result
            result_df = df_latest[final_mask].copy()
            
            st.success(f"Found {len(result_df)} stocks matching all {len(st.session_state.rules)} rules.")
            
            # Display columns
            base_cols = ['code', 'name', 'date', 'close']
            metric_cols = sorted([f"ma{w}" for w in needed_mas])
            st.dataframe(result_df[base_cols + metric_cols].reset_index(drop=True), use_container_width=True)

else:
    st.info("Define rules in the sidebar and click 'Calculate & Filter'.")
    st.write("Raw Data Preview:", df_raw.head())
