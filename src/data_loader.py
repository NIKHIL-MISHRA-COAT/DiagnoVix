import pandas as pd
import os

def load_csv_data(file_path: str):
    """Loads data from a CSV file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Error: CSV file not found at {file_path}")
    try:
        df = pd.read_csv(file_path)
        print(f"Successfully loaded data from {file_path}. Shape: {df.shape}")
        return df
    except Exception as e:
        raise Exception(f"Error loading CSV file {file_path}: {e}")



def load_data(data_path: str = None, target_column: str = None):
    """Loads data from CSV and validates target column."""
    if data_path:
        df = load_csv_data(data_path)
    else:
        raise ValueError("Error: CSV data_path  must be provided.")

    if target_column:
        if target_column not in df.columns:
            raise ValueError(f"Error: Target column '{target_column}' not found in the loaded dataset. Available columns: {df.columns.tolist()}")
    else:
        print("Warning: No target column specified. Some operations might be limited.")
        # Attempt to infer target if not provided (could be the last column, but risky)
        # For this project, we'll require it to be specified for clarity.
        raise ValueError("Error: Target column must be specified for diagnosis tasks.")
        
    return df