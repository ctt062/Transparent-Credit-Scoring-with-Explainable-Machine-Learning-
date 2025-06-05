import os
import pandas as pd

def load_data(train_file_path='train.csv', test_file_path='test.csv'):
    """
    Loads training and testing datasets from specified CSV file paths.

    Checks for the existence of the files and prints their shapes upon successful loading.

    Parameters:
    ----------
    train_file_path : str, optional
        The file path to the training data CSV file. 
        Defaults to 'train.csv' in the current working directory.
    test_file_path : str, optional
        The file path to the test data CSV file. 
        Defaults to 'test.csv' in the current working directory.

    Returns:
    -------
    tuple
        A tuple containing two pandas DataFrames: (train_df, test_df).
        - train_df: DataFrame loaded from train_file_path.
        - test_df: DataFrame loaded from test_file_path.

    Raises:
    ------
    FileNotFoundError
        If either the train_file_path or test_file_path does not exist.
    """
    if not os.path.exists(train_file_path):
        raise FileNotFoundError(f"Training file not found: {train_file_path}. Please ensure 'train.csv' is correctly pathed.")
    if not os.path.exists(test_file_path):
        raise FileNotFoundError(f"Test file not found: {test_file_path}. Please ensure 'test.csv' is correctly pathed.")
    
    print("✅ Dataset files found.")

    train_df = pd.read_csv(train_file_path)
    test_df = pd.read_csv(test_file_path)
    
    print("Train shape:", train_df.shape)
    print("Test shape:", test_df.shape)
    
    return train_df, test_df