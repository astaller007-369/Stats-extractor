import streamlit as st
import pandas as pd
import io

# Import the OCR extraction processing logic from our custom parser file
from ocr_parser import process_screenshot

# Set up page layout configs
st.set_page_config(page_title="Football Stats Extractor", page_icon="⚽", layout="wide")

if 'master_pivoted_data' not in st.session_state:
    st.session_state.master_pivoted_data = []

st.title("⚽ Advanced Football Match Stats Compiler")
st.markdown("Upload screenshots from your games, specify individual match dates on the fly, and download a cleanly pivoted flat matrix spreadsheet database.")

with st.sidebar:
    st.header("⚙️ App Reset Options")
    if st.button("AI Reset / Clear All Entries", type="secondary"):
        st.session_state.master_pivoted_data = []
        st.rerun()

uploaded_files = st.file_uploader(
    "Drag and drop or select match statistics images:", 
    type=["png", "jpg", "jpeg"], 
    accept_multiple_files=True
)

if uploaded_files:
    st.write("### 📅 Assign Dates to Uploaded Images")
    file_date_mapping = {}
    
    for index, file in enumerate(uploaded_files):
        col1, col2 = st.columns(2)
        with col1:
            st.text(f"📄 {file.name}")
        with col2:
            chosen_date = st.date_input(
                f"Match Date for file #{index+1}", 
                value=pd.to_datetime("today"), 
                key=f"date_picker_{file.name}_{index}"
            )
            file_date_mapping[file.name] = chosen_date
            
    st.markdown("---")
    
    if st.button("🚀 Process Batch & Append to Database", type="primary", use_container_width=True):
        new_matches_count = 0
        
        for file in uploaded_files:
            bytes_data = file.read()
            match_date = file_date_mapping[file.name].strftime('%Y-%m-%d')
            
            # Run OCR extraction matrix logic
            match_data_record = process_screenshot(bytes_data)
            
            # Prepend date row item parameter to line up with sequence constraints
            final_ordered_row = {"date": match_date}
            final_ordered_row.update(match_data_record)
            
            st.session_state.master_pivoted_data.append(final_ordered_row)
            new_matches_count += 1
            
        st.success(f"Success! Correctly processed and merged {new_matches_count} football match logs into your main database.")

if st.session_state.master_pivoted_data:
    # Explicit target sequence configuration requirement structure definition (Excluding status)
    target_header_sequence = [
        "date", "home_team", "away_team", "goals_home", "goals_away",
        "expected_goals_home", "expected_goals_away", "goalkeeper_saves_home", "goalkeeper_saves_away",
        "big_chances_home", "big_chances_away", "shots_on_target_home", "shots_on_target_away",
        "touches_in_penalty_area_home", "touches_in_penalty_area_away", "corner_kicks_home", "corner_kicks_away",
        "fouled_in_final_third_home", "fouled_in_final_third_away", "accurate_crosses_home", "accurate_crosses_away",
        "aerial_duels_percentage_home", "aerial_duels_percentage_away", "accurate_long_balls_home", "accurate_long_balls_away",
        "final_third_entries_home", "final_third_entries_away", "dribbles_percentage_home", "dribbles_percentage_away",
        "tackles_won_percentage_home", "tackles_won_percentage_away", "ground_duels_percentage_home", "ground_duels_percentage_away"
    ]
    
    df_pivoted = pd.DataFrame(st.session_state.master_pivoted_data)
    
    # Enforce strict missing key safety fallback logic
    for col in target_header_sequence:
        if col not in df_pivoted.columns:
            df_pivoted[col] = "0"
            
    # Force re-index sequence ordering
    df_pivoted = df_pivoted[target_header_sequence]
    
    st.write("### 🕒 Active Database Output View (Strict Sequential Row Layout)")
    st.dataframe(df_pivoted, use_container_width=True)
    
    st.markdown("---")
    st.write("### 💾 Export Options")
    
    down_col1, down_col2 = st.columns(2)
    with down_col1:
        custom_filename = st.text_input(
            "Name your file before downloading:", 
            value="football_match_master_database"
        )
        final_filename = custom_filename.strip() if custom_filename.endswith(".csv") else f"{custom_filename.strip()}.csv"
            
    with down_col2:
        csv_buffer = io.StringIO()
        df_pivoted.to_csv(csv_buffer, index=False, encoding='utf-8')
        csv_data = csv_buffer.getvalue()
        
        st.write("<div style='padding-top:24px;'></div>", unsafe_allow_html=True)
        st.download_button(
            label="📥 Download Structured CSV File",
            data=csv_data,
            file_name=final_filename,
            mime="text/csv",
            use_container_width=True
        )
        