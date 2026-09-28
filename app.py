import streamlit as st
import cv2
import numpy as np
import pytesseract
import pandas as pd
import re
import io

# =========================================================
# SEGMENT 1: ADVANCED IMAGE PREPROCESSING & TEXT ENGINE LOGIC
# =========================================================

def standardize_value(val_string):
    """Converts commas to decimals and strips non-numeric symbols."""
    if not val_string:
        return "0"
    val_cleaned = val_string.replace(',', '.')
    val_cleaned = re.sub(r'[^\d\.]', '', val_cleaned)
    return val_cleaned if val_cleaned else "0"

def parse_just_completed_raw(raw_line, keyword):
    """Isolates the left/right blocks and takes ONLY the completed integer volume."""
    parts = re.split(re.escape(keyword), raw_line, flags=re.IGNORECASE)
    if len(parts) < 2:
        return "0", "0"
    left_side = parts[0].strip()
    right_side = parts[1].strip()
    
    left_raw_match = re.search(r'(\d+)(?:/\d+)?', left_side)
    right_raw_match = re.search(r'(\d+)(?:/\d+)?', right_side)
    
    home_raw = left_raw_match.group(1) if left_raw_match else "0"
    away_raw = right_raw_match.group(1) if right_raw_match else "0"
    return home_raw, away_raw

def parse_percentage_only(raw_line, keyword):
    """Extracts only percentage metrics while maintaining team isolation."""
    parts = re.split(re.escape(keyword), raw_line, flags=re.IGNORECASE)
    if len(parts) < 2:
        return "0%", "0%"
    left_side = parts[0].strip()
    right_side = parts[1].strip()
    left_pct = re.search(r'(\d+%)', left_side)
    right_pct = re.search(r'(\d+%)', right_side)
    return (left_pct.group(1) if left_pct else "0%", right_pct.group(1) if right_pct else "0%")

def parse_standard_metric(raw_line, keyword):
    """Extracts basic integer or decimal counters safely."""
    parts = re.split(re.escape(keyword), raw_line, flags=re.IGNORECASE)
    if len(parts) < 2:
        return "0", "0"
    
    left_part = parts[0].strip().split()
    right_part = parts[1].strip().split()
    
    home_val = standardize_value(left_part[-1]) if left_part else "0"
    away_val = standardize_value(right_part[0]) if right_part else "0"
    return home_val, away_val

def process_screenshot_batch(image_bytes_list):
    """
    Loops through files in a match batch, applies 2x spatial scaling 
    and character whitelisting to eliminate character misread errors.
    """
    compiled_data = {}
    
    for img_bytes in image_bytes_list:
        file_bytes = np.asarray(bytearray(img_bytes), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        
        # UPGRADE A: Scale up image dimension sizes x2 to isolate micro font edges
        height, width = image.shape[:2]
        image_scaled = cv2.resize(image, (width * 2, height * 2), interpolation=cv2.INTER_CUBIC)
        
        gray = cv2.cvtColor(image_scaled, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # UPGRADE B: Enforce character whitelist mapping configuration rules
        # Restricts engine choices to prevent alphanumeric cross-mutation errors (e.g. 5 vs 6)
        custom_config = (
            r'--oem 3 --psm 6 '
            r'-c tessedit_char_whitelist="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,%() " '
            r'-c load_system_dawg=0 -c load_freq_dawg=0'
        )
        
        raw_text = pytesseract.image_to_string(thresh, config=custom_config)
        lines = raw_text.split('\n')
        
        # Extract goals if the score block is present
        for line in lines[:8]:
            match_score = re.search(r'(\d+)\s+(?:FT|AET|PEN)\s+(\d+)', line, flags=re.IGNORECASE)
            if match_score:
                compiled_data["goals_home"] = match_score.group(1)
                compiled_data["goals_away"] = match_score.group(2)
                break
                
        # Scrape lines for stats metrics mapping components across images
        for line in lines:
            if "expected goals" in line.lower() and "open play" not in line.lower() and "set play" not in line.lower() and "on target" not in line.lower():
                compiled_data["expected_goals_home"], compiled_data["expected_goals_away"] = parse_standard_metric(line, "Expected goals (xG)")
            if "keeper saves" in line.lower():
                compiled_data["goalkeeper_saves_home"], compiled_data["goalkeeper_saves_away"] = parse_standard_metric(line, "Keeper saves")
            if "big chances" in line.lower():
                compiled_data["big_chances_home"], compiled_data["big_chances_away"] = parse_standard_metric(line, "Big chances")
            if "shots on target" in line.lower():
                compiled_data["shots_on_target_home"], compiled_data["shots_on_target_away"] = parse_standard_metric(line, "Shots on target")
            if "touches in opposition box" in line.lower():
                compiled_data["touches_in_penalty_area_home"], compiled_data["touches_in_penalty_area_away"] = parse_standard_metric(line, "Touches in opposition box")
            if "corners" in line.lower():
                compiled_data["corner_kicks_home"], compiled_data["corner_kicks_away"] = parse_standard_metric(line, "Corners")
            if "fouled in final third" in line.lower():
                compiled_data["fouled_in_final_third_home"], compiled_data["fouled_in_final_third_away"] = parse_standard_metric(line, "Fouled in final third")
            if "accurate crosses" in line.lower():
                compiled_data["accurate_crosses_home"], compiled_data["accurate_crosses_away"] = parse_just_completed_raw(line, "Accurate crosses")
            if "aerial duels won" in line.lower():
                compiled_data["aerial_duels_percentage_home"], compiled_data["aerial_duels_percentage_away"] = parse_percentage_only(line, "Aerial duels won")
            if "accurate long balls" in line.lower():
                compiled_data["accurate_long_balls_home"], compiled_data["accurate_long_balls_away"] = parse_just_completed_raw(line, "Accurate long balls")
            if "final third entries" in line.lower():
                compiled_data["final_third_entries_home"], compiled_data["final_third_entries_away"] = parse_standard_metric(line, "Final third entries")
            if "successful dribbles" in line.lower():
                compiled_data["dribbles_percentage_home"], compiled_data["dribbles_percentage_away"] = parse_percentage_only(line, "Successful dribbles")
            if "tackles won" in line.lower():
                compiled_data["tackles_won_percentage_home"], compiled_data["tackles_won_percentage_away"] = parse_percentage_only(line, "Tackles won")
            if "ground duels won" in line.lower():
                compiled_data["ground_duels_percentage_home"], compiled_data["ground_duels_percentage_away"] = parse_percentage_only(line, "Ground duels won")

    return compiled_data
# =========================================================
# SEGMENT 2: STREAMLIT APP USER INTERFACE & PERSISTENT MEMORY
# =========================================================

# Initialize global historical memory across multi-batch uploads
if 'master_database' not in st.session_state:
    st.session_state.master_database = []

st.title("⚽ Multi-Match Database Compiler")
st.markdown("Upload files batch-by-batch. Configure team names and a match date for the active batch, commit it to history, and move on to your next match setup.")

# Sidebar Configuration Control Panel
with st.sidebar:
    st.header("⚙️ Database Operations")
    if st.button("Core Reset / Wipe All Entries", type="secondary", use_container_width=True):
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
            
            # Build the finalized custom row layout with tracking fields
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

# Global Database Render Interface Framework Area
if st.session_state.master_database:
    st.subheader("🕒 Persistent Historical Master Database")
    
    # Enforce strict snake_case horizontal order sequence constraints
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
    
    # Fill any missing metrics dynamically to avoid display glitches
    for col in target_header_sequence:
        if col not in df_master.columns:
            df_master[col] = "0"
            
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
    # =========================================================
# SEGMENT 2: STREAMLIT APP USER INTERFACE & PERSISTENT MEMORY
# =========================================================

# Initialize global historical memory across multi-batch uploads
if 'master_database' not in st.session_state:
    st.session_state.master_database = []

st.title("⚽ Multi-Match Database Compiler")
st.markdown("Upload files batch-by-batch. Configure team names and a match date for the active batch, commit it to history, and move on to your next match setup.")

# Sidebar Configuration Control Panel
with st.sidebar:
    st.header("⚙️ Database Operations")
    if st.button("Core Reset / Wipe All Entries", type="secondary", use_container_width=True):
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
            
            # Build the finalized custom row layout with tracking fields
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

# Global Database Render Interface Framework Area
if st.session_state.master_database:
    st.subheader("🕒 Persistent Historical Master Database")
    
    # Enforce strict snake_case horizontal order sequence constraints
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
    
    # Fill any missing metrics dynamically to avoid display glitches
    for col in target_header_sequence:
        if col not in df_master.columns:
            df_master[col] = "0"
            
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
