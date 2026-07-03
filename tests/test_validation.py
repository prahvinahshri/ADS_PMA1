import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "src")))

from data_utils import clean_data
from validation import check_duplicates, check_missing_values, check_value_range

def make_sample():
    return pd.DataFrame({
        "ID": [1, 2, 3, 4],
        "Arrival Delay": [5.0, np.nan, 12.0, 0.0],
        "Cleanliness": [3, 4, 0, 5],
    })

def test_check_missing_values_detects_nulls():
    missing = check_missing_values(make_sample())
    assert "Arrival Delay" in missing.index
    assert missing["Arrival Delay"] == 1

def test_check_duplicates_counts_correctly():
    df = make_sample()
    assert check_duplicates(df) == 0
    df_dup = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    assert check_duplicates(df_dup) == 1

def test_check_value_range_flags_invalid_ratings():
    invalid = check_value_range(make_sample(), "Cleanliness", 1, 5)
    assert len(invalid) == 1
    assert invalid.iloc[0]["Cleanliness"] == 0

def test_clean_data_fixes_both_issues():
    cleaned = clean_data(make_sample())
    assert cleaned["Arrival Delay"].isnull().sum() == 0
    assert cleaned["Cleanliness"].between(1, 5).all()
