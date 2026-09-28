import streamlit as st
import pandas as pd
import io

# Import the core OCR handler function from our background parser file
from ocr_parser import process_screenshot_batch

# Set up page layout configurations
st.set_page_config(page_title="Football Match Compiler", page_icon="⚽", layout="wide")

# Initialize global historical memory across multi-batch uploads
if 'master_database' not in st.session_state:
    st.session_state.master_database = []

st.title("⚽ Multi-Match Database Compiler")
st.markdown("Upload files batch-by-batch. Configure team names and a match date for the active batch, commit it to history, and move on to your next match setup.")

# Sidebar Configuration Control Panel
with st.sidebar:
    st.header("⚙️ Database Operations")
    if st.button("🧹 Wipe Entire Session History", type="secondary", use_container_width=True):
        st.session_state.master_database = []
        st.rerun()

st.write("---")
st.subheader("📥 Current Match Batch Configuration")

# Match Setup Field Inputs
meta_col1, meta_col2, meta_col3 = st.columns(3)
with meta_col1:
    match_date = st.date_input("Match Date:", value=pd.to_datetime("today"))
with meta_col2:
    home_name_input = st.text_input("Home Team Name:", placeholder="e.g., Barnsley")
with meta_col3:
    away_name_input = st.text_input("Away Team Name:", placeholder="e.g., Port Vale")

# File Upload Entry point targeting the current active match configuration
uploaded_files = st.file_uploader(
    f"Upload all screenshots belonging to the match between {home_name_input if home_name_input else 'Home'} vs {away_name_input if away_name_input else 'Away'}:", 
    type=["png", "jpg", "jpeg"], 
    accept_multiple_files=True,
    key="match_uploader"
)

if uploaded_files:
    if not home_name_input or not away_name_input:
        st.error("⚠️ Please fill in both Home and Away team names before processing this batch.")
    else:
        if st.button("🚀 Process & Commit Match to Master Database", type="primary", use_container_width=True):
            # Collect file streams into array payloads
            bytes_list = [file.read() for file in uploaded_files]
            
            # Send the collective images down to the engine to return one compiled dataset dictionary 
            extracted_metrics = process_screenshot_batch(bytes_list)
            
            # Build the finalized custom row row mapping layout tracking structural text fields
            finalized_match_row = {
                "date": match_date.strftime('%Y-%m-%d'),
                "home_team": home_name_input.strip(),
                "away_team": away_name_input.strip(),
                "goals_home": extracted_metrics.get("goals_home", "0"),
                "goals_away": extracted_metrics.get("goals_away", "0"),
                "expected_goals_home": extracted_metrics.get("expected_goals_home", "0"),
                "expected_goals_away": extracted_metrics.get("expected_goals_away", "0"),
                "goalkeeper_saves_home": extracted_metrics.get("goalkeeper_saves_home", "0"),
                "goalkeeper_saves_away": extracted_metrics.get("goalkeeper_saves_away", "0"),
                "big_chances_home": extracted_metrics.get("big_chances_home", "0"),
                "big_chances_away": extracted_metrics.get("big_chances_away", "0"),
                "shots_on_target_home": extracted_metrics.get("shots_on_target_home", "0"),
                "shots_on_target_away": extracted_metrics.get("shots_on_target_away", "0"),
                "touches_in_penalty_area_home": extracted_metrics.get("touches_in_penalty_area_home", "0"),
                "touches_in_penalty_area_away": extracted_metrics.get("touches_in_penalty_area_away", "0"),
                "corner_kicks_home": extracted_metrics.get("corner_kicks_home", "0"),
                "corner_kicks_away": extracted_metrics.get("corner_kicks_away", "0"),
                "fouled_in_final_third_home": extracted_metrics.get("fouled_in_final_third_home", "0"),
                "fouled_in_final_third_away": extracted_metrics.get("fouled_in_final_third_away", "0"),
                "accurate_crosses_home": extracted_metrics.get("accurate_crosses_home", "0"),
                "accurate_crosses_away": extracted_metrics.get("accurate_crosses_away", "0"),
                "aerial_duels_percentage_home": extracted_metrics.get("aerial_duels_percentage_home", "0%"),
                "aerial_duels_percentage_away": extracted_metrics.get("aerial_duels_percentage_away", "0%"),
                "accurate_long_balls_home": extracted_metrics.get("accurate_long_balls_home", "0"),
                "accurate_long_balls_away": extracted_metrics.get("accurate_long_balls_away", "0"),
                "final_third_entries_home": extracted_metrics.get("final_third_entries_home", "0"),
                "final_third_entries_away": extracted_metrics.get("final_third_entries_away", "0"),
                "dribbles_percentage_home": extracted_metrics.get("dribbles_percentage_home", "0%"),
                "dribbles_percentage_away": extracted_metrics.get("dribbles_percentage_away", "0%"),
                "tackles_won_percentage_home": extracted_metrics.get("tackles_won_percentage_home", "0%"),
                "tackles_won_percentage_away": extracted_metrics.get("tackles_won_percentage_away", "0%"),
                "ground_duels_percentage_home": extracted_metrics.get("ground_duels_percentage_home", "0%"),
                "ground_duels_percentage_away": extracted_metrics.get("ground_duels_percentage_away", "0%")
            }
            
            # Commit entry row record down to long-term app storage
            st.session_state.master_database.append(finalized_match_row)
            st.success(f"Successfully compiled and saved match data for: {home_name_input} vs {away_name_input}!")
            
            # Clear file entry point arrays to prepare cleanly for next set of screenshot configurations
            st.rerun()

st.write("---")

# Step 2: Global Database Render Interface Framework Area
if st.session_state.master_database:
    st.subheader("🕒 Persistent Historical Master Database")
    
    # Enforce sequence layout constraints configuration logic rule array
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
    
    df_master = pd.DataFrame(st.session_state.master_database)
    df_master = df_master[target_header_sequence]
    
    st.dataframe(df_master, use_container_width=True)
    
    # File Naming and Download Section
    st.markdown("### 💾 Export Database")
    down_col1, down_col2 = st.columns(2)
    with down_col1:
        custom_filename = st.text_input("Name your file before downloading:", value="football_match_master_database")
        final_filename = custom_filename.strip() if custom_filename.endswith(".csv") else f"{custom_filename.strip()}.csv"
        
    with down_col2:
        csv_buffer = io.StringIO()
        df_master.to_csv(csv_buffer, index=False, encoding='utf-8')
        csv_data = csv_buffer.getvalue()
        
        st.write("<div style='padding-top:24px;'></div>", unsafe_allow_html=True)
        st.download_button(
            label="📥 Download Complete CSV Dataset",
            data=csv_data,
            file_name=final_filename,
            mime="text/csv",
            use_container_width=True
        )
else:
    st.info("The master database is currently empty. Configure a match above, upload your screenshots, and save it to begin compiling your list.")
