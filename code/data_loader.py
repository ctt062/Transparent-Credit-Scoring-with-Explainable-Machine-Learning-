import os
import pandas as pd

def load_data(train_file_path='train.csv', test_file_path='test.csv'):
    # These paths are now expected to be absolute or correctly relative from where main.py is run
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