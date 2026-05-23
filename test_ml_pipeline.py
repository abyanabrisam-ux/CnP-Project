import pandas as pd
import numpy as np
import pytest
from dataset import get_esports_data
from ml_pipeline import prepare_data

def test_data_scaling_mean_variance():
    """
    Verify that the data scaling step outputs values with a mean of ~0 and a variance of ~1.
    """
    df = get_esports_data()
    scaled_df, player_ids, scaler = prepare_data(df)
    
    # Check mean is close to 0
    means = scaled_df.mean()
    assert np.allclose(means, 0, atol=1e-7), f"Means are not zero: {means}"
    
    # Check variance is close to 1
    # Note: StandardScaler calculates variance using ddof=0, pandas var() uses ddof=1 by default
    # So we use ddof=0 for pandas var() to match StandardScaler
    variances = scaled_df.var(ddof=0)
    assert np.allclose(variances, 1, atol=1e-7), f"Variances are not one: {variances}"

def test_training_dataframe_excludes_player_id():
    """
    Verify that the training dataframe strictly excludes the 'Player_ID' column.
    """
    df = get_esports_data()
    scaled_df, player_ids, scaler = prepare_data(df)
    
    assert 'Player_ID' not in scaled_df.columns, "'Player_ID' is still present in the scaled training data."
    assert player_ids is not None, "Player IDs were not successfully isolated."
    assert len(player_ids) == len(scaled_df), "Number of extracted player IDs does not match number of training samples."
