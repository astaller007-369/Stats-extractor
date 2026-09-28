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
        return "0", "0"
    left_side = parts.strip()
    right_side = parts.strip()
    
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
    left_side = parts.strip()
    right_side = parts.strip()
    left_pct = re.search(r'(\d+%)', left_side)
    right_pct = re.search(r'(\d+%)', right_side)
    return (left_pct.group(1) if left_pct else "0%", right_pct.group(1) if right_pct else "0%")

def parse_standard_metric(raw_line, keyword):
    """Extracts basic integer or decimal counters safely."""
    parts = re.split(re.escape(keyword), raw_line, flags=re.IGNORECASE)
    if len(parts) < 2:
        return "0", "0"
    home_val = standardize_value(parts.strip().split()[-1]) if parts.strip() else "0"
    away_val = standardize_value(parts.strip().split()) if parts.strip() else "0"
    return home_val, away_val

def process_screenshot_batch(image_bytes_list):
    """
    Loops through all files within a specific match batch and safely aggregates 
    all found information down into a single master dictionary block.
    """
    compiled_data = {}
    
    for img_bytes in image_bytes_list:
        file_bytes = np.asarray(bytearray(img_bytes), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        custom_config = r'--oem 3 --psm 6'
        raw_text = pytesseract.image_to_string(thresh, config=custom_config)
        lines = raw_text.split('\n')
        
        # Pull high precision baseline goals metrics if the score layout block is parsed
        for line in lines[:5]:
            match_score = re.search(r'(\d+)\s+(?:FT|AET|PEN)\s+(\d+)', line, flags=re.IGNORECASE)
            if match_score:
                compiled_data["goals_home"] = match_score.group(1)
                compiled_data["goals_away"] = match_score.group(2)
                break
                
        # Scrape and extract structural line stats metrics mapping components across images
        for line in lines:
            if "expected goals (xg)" in line.lower() and "open play" not in line.lower() and "set play" not in line.lower() and "on target" not in line.lower():
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
    
