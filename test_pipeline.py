import pandas as pd
import numpy as np
import pytest
from dataset import get_esports_data
from ml_pipeline import prepare_data

def test_scaled_features_mean_variance():
    """
    Verify that the scaled features maintain a mean of approximately 0 and a variance of 1.
    """
    df = get_esports_data()
    # We use all numeric columns for K-Means
    columns_to_scale = ['Average_Survival_Time_Min', 'Average_Damage_Dealt', 'Kills_Per_Match', 'Assists_Per_Match']
    scaled_df, player_ids, scaler = prepare_data(df, columns_to_scale)
    
    # Check mean is approx 0
    means = scaled_df.mean()
    assert np.allclose(means, 0, atol=1e-7), f"Means are not close to zero: {means}"
    
    # Check variance is approx 1
    # Note: StandardScaler uses ddof=0, pandas uses ddof=1 by default. We match ddof=0.
    variances = scaled_df.var(ddof=0)
    assert np.allclose(variances, 1, atol=1e-7), f"Variances are not close to one: {variances}"

def test_kmeans_input_excludes_player_id():
    """
    Verify that the shape of the input data matrix fed into K-Means completely
    excludes the string-based 'Player_ID' column.
    """
    df = get_esports_data()
    columns_to_scale = ['Average_Survival_Time_Min', 'Average_Damage_Dealt', 'Kills_Per_Match', 'Assists_Per_Match']
    scaled_df, player_ids, scaler = prepare_data(df, columns_to_scale)
    
    # Check that 'Player_ID' is not in the columns
    assert 'Player_ID' not in scaled_df.columns, "'Player_ID' is still present in the input matrix!"
    
    # Check that all remaining columns are strictly numeric types (no string-based columns)
    is_all_numeric = all(pd.api.types.is_numeric_dtype(scaled_df[col]) for col in scaled_df.columns)
    assert is_all_numeric, "Input data matrix contains non-numeric (string-based) columns."
