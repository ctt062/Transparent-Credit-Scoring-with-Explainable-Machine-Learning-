import os
import pandas as pd

def load_data(train_file_path='train.csv', test_file_path='test.csv'):
    if not os.path.exists(train_file_path) or not os.path.exists(test_file_path):
        raise FileNotFoundError(f"Ensure '{train_file_path}' and '{test_file_path}' are in the project root.")
    print("✅ Dataset files found.")

    train_df = pd.read_csv(train_file_path)
    test_df = pd.read_csv(test_file_path)
    
    print("Train shape:", train_df.shape)
    print("Test shape:", test_df.shape)
    
    return train_df, test_df