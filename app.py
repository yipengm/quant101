import streamlit as st
import pandas as pd
import glob
import os

st.set_page_config(page_title="Quant101 Stock Filter", layout="wide")

st.title("Quant101 Stock Filter Results")

# 1. Find available result files
csv_files = sorted(glob.glob("result_*.csv"), reverse=True)

if not csv_files:
    st.warning("No 'result_*.csv' files found in the current directory.")
    st.stop()

# 2. Sidebar: Select file to load
selected_file = st.sidebar.selectbox("Select Result File", csv_files)

@st.cache_data
def load_data(filepath):
    df = pd.read_csv(filepath, dtype={"code": str})
    return df

if selected_file:
    df = load_data(selected_file)
    st.sidebar.success(f"Loaded {len(df)} rows from {selected_file}")

    # 3. Sidebar: Filter metrics
    st.sidebar.header("Filter Metrics")
    
    # Define the exact columns user wants to filter by
    # Removed "interesting" as requested, focused on individual components
    target_filter_cols = ["ma5 cross ma30", "ma5up", "ma10up", "ma20up", "ma30up"]
    
    # Check which columns actually exist in the dataframe
    available_cols = [c for c in target_filter_cols if c in df.columns]
    
    # Warn if some expected columns are missing (e.g. if using an old CSV)
    missing_cols = set(target_filter_cols) - set(available_cols)
    if missing_cols:
        st.sidebar.warning(f"Columns not found in CSV: {', '.join(missing_cols)}")
        st.sidebar.info("Please re-run 'stock_filter.py' to generate a new result CSV with these metrics.")
    
    selected_metrics = []
    for col in available_cols:
        # Default value False means "don't care" / "show all"
        if st.sidebar.checkbox(f"{col}", value=False):
            selected_metrics.append(col)
            
    # 4. Apply Filters
    filtered_df = df.copy()
    
    for metric in selected_metrics:
        # Ensure we only keep rows where the column is True
        # Handle potential string 'True'/'False' or booleans
        if filtered_df[metric].dtype == object:
             filtered_df = filtered_df[filtered_df[metric].astype(str) == "True"]
        else:
             filtered_df = filtered_df[filtered_df[metric] == True]

    # 5. Display Results
    st.subheader(f"Filtered Results ({len(filtered_df)} stocks)")
    
    if selected_metrics:
        st.write(f"Filtering by: **{' AND '.join(selected_metrics)}**")
    else:
        st.info("No filters applied. Showing all stocks.")

    st.dataframe(filtered_df, use_container_width=True)
    
    # Optional: Download filtered results
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        "Download Filtered CSV",
        csv,
        "filtered_results.csv",
        "text/csv",
        key='download-csv'
    )
