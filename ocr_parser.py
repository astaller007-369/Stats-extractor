import cv2
import numpy as np
import pytesseract
import re

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
        return ("0", "0")
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
        return ("0%", "0%")
    left_side = parts[0].strip()
    right_side = parts[1].strip()
    left_pct = re.search(r'(\d+%)', left_side)
    right_pct = re.search(r'(\d+%)', right_side)
    return (left_pct.group(1) if left_pct else "0%", right_pct.group(1) if right_pct else "0%")

def parse_standard_metric(raw_line, keyword):
    """Extracts basic integer or decimal counters safely."""
    parts = re.split(re.escape(keyword), raw_line, flags=re.IGNORECASE)
    if len(parts) < 2:
        return ("0", "0")
    home_val = standardize_value(parts[0].strip().split()[-1]) if parts[0].strip() else "0"
    away_val = standardize_value(parts[1].strip().split()[0]) if parts[1].strip() else "0"
    return (home_val, away_val)

def process_screenshot(image_bytes):
    """Processes a single screenshot and extracts raw strings, headers, and metadata metrics."""
    file_bytes = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    custom_config = r'--oem 3 --psm 6'
    raw_text = pytesseract.image_to_string(thresh, config=custom_config)
    lines = raw_text.split('\n')
    
    # Initialize basic layout metadata defaults
    home_team, away_team = "Home Team", "Away Team"
    goals_home, goals_away = "0", "0"
    
    # High-precision extraction for the scoreline header banner
    for line in lines[:5]:
        match_score = re.search(r'(\d+)\s+(?:FT|AET|PEN)\s+(\d+)', line, flags=re.IGNORECASE)
        if match_score:
            goals_home = match_score.group(1)
            goals_away = match_score.group(2)
            
            team_parts = re.split(r'\d+\s+(?:FT|AET|PEN)\s+\d+', line, flags=re.IGNORECASE)
            if len(team_parts) >= 2:
                home_team = team_parts[0].strip()
                away_team = team_parts[1].strip()
            break

    extracted = {}
    
    # Process standard text line extraction
    for line in lines:
        if "expected goals (xg)" in line.lower() and "open play" not in line.lower() and "set play" not in line.lower() and "on target" not in line.lower():
            extracted["xg_home"], extracted["xg_away"] = parse_standard_metric(line, "Expected goals (xG)")
        if "keeper saves" in line.lower():
            extracted["saves_home"], extracted["saves_away"] = parse_standard_metric(line, "Keeper saves")
        if "big chances" in line.lower():
            extracted["big_home"], extracted["big_away"] = parse_standard_metric(line, "Big chances")
        if "shots on target" in line.lower():
            extracted["sot_home"], extracted["sot_away"] = parse_standard_metric(line, "Shots on target")
        if "touches in opposition box" in line.lower():
            extracted["touches_home"], extracted["touches_away"] = parse_standard_metric(line, "Touches in opposition box")
        if "corners" in line.lower():
            extracted["corners_home"], extracted["corners_away"] = parse_standard_metric(line, "Corners")
        if "fouled in final third" in line.lower():
            extracted["fouled_home"], extracted["fouled_away"] = parse_standard_metric(line, "Fouled in final third")
        if "accurate crosses" in line.lower():
            extracted["cross_home"], extracted["cross_away"] = parse_just_completed_raw(line, "Accurate crosses")
        if "aerial duels won" in line.lower():
            extracted["aerial_home"], extracted["aerial_away"] = parse_percentage_only(line, "Aerial duels won")
        if "accurate long balls" in line.lower():
            extracted["long_home"], extracted["long_away"] = parse_just_completed_raw(line, "Accurate long balls")
        if "final third entries" in line.lower():
            extracted["entries_home"], extracted["entries_away"] = parse_standard_metric(line, "Final third entries")
        if "successful dribbles" in line.lower():
            extracted["dribble_home"], extracted["dribble_away"] = parse_percentage_only(line, "Successful dribbles")
        if "tackles won" in line.lower():
            extracted["tackles_home"], extracted["tackles_away"] = parse_percentage_only(line, "Tackles won")
        if "ground duels won" in line.lower():
            extracted["ground_home"], extracted["ground_away"] = parse_percentage_only(line, "Ground duels won")

    return {
        "home_team": home_team,
        "away_team": away_team,
        "goals_home": goals_home,
        "goals_away": goals_away,
        "expected_goals_home": extracted.get("xg_home", "0"),
        "expected_goals_away": extracted.get("xg_away", "0"),
        "goalkeeper_saves_home": extracted.get("saves_home", "0"),
        "goalkeeper_saves_away": extracted.get("saves_away", "0"),
        "big_chances_home": extracted.get("big_home", "0"),
        "big_chances_away": extracted.get("big_away", "0"),
        "shots_on_target_home": extracted.get("sot_home", "0"),
        "shots_on_target_away": extracted.get("sot_away", "0"),
        "touches_in_penalty_area_home": extracted.get("touches_home", "0"),
        "touches_in_penalty_area_away": extracted.get("touches_away", "0"),
        "corner_kicks_home": extracted.get("corners_home", "0"),
        "corner_kicks_away": extracted.get("corners_away", "0"),
        "fouled_in_final_third_home": extracted.get("fouled_home", "0"),
        "fouled_in_final_third_away": extracted.get("fouled_away", "0"),
        "accurate_crosses_home": extracted.get("cross_home", "0"),
        "accurate_crosses_away": extracted.get("cross_away", "0"),
        "aerial_duels_percentage_home": extracted.get("aerial_home", "0%"),
        "aerial_duels_percentage_away": extracted.get("aerial_away", "0%"),
        "accurate_long_balls_home": extracted.get("long_home", "0"),
        "accurate_long_balls_away": extracted.get("long_away", "0"),
        "final_third_entries_home": extracted.get("entries_home", "0"),
        "final_third_entries_away": extracted.get("entries_away", "0"),
        "dribbles_percentage_home": extracted.get("dribble_home", "0%"),
        "dribbles_percentage_away": extracted.get("dribble_away", "0%"),
        "tackles_won_percentage_home": extracted.get("tackles_home", "0%"),
        "tackles_won_percentage_away": extracted.get("tackles_away", "0%"),
        "ground_duels_percentage_home": extracted.get("ground_home", "0%"),
        "ground_duels_percentage_away": extracted.get("ground_away", "0%")
    }
    