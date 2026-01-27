import pandas as pd
import numpy as np
import os
import glob
from pathlib import Path

def load_data(folder_name="Modelled"):
    """
    Load all CSV files from the specified data folder.
    
    Parameters:
    -----------
    folder_name : str
        Name of the folder within the package's data directory (default: "Modelled")
    
    Returns:
    --------
    dict
        Dictionary with dataset names as keys and pandas DataFrames as values
    """

    folder_path = r"C:\Users\inigo\OneDrive\Documents\Fourth_Year\Computations\Dissertation\DiffusionRWR_model_repo\DiffusionRWR_model_package\data\Modelled"
    
    # Convert to string for compatibility with glob
    folder_path = str(folder_path)
    
    # Get all CSV files
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    
    print(f"Number of CSV files found: {len(csv_files)}")
    if csv_files:
        print("\nCSV files found:")
        for f in csv_files:
            print(f"  - {os.path.basename(f)}")
    
    # Import all CSV files into a dictionary
    data_dict = {}
    
    for file in csv_files:
        name = os.path.splitext(os.path.basename(file))[0]
        df = pd.read_csv(file)
        
        # Set first column as index if it contains gene names
        if df.columns[0] == 'Unnamed: 0' or 'gene' in df.columns[0].lower():
            df.set_index(df.columns[0], inplace=True)
        
        # Filter columns to only keep integer time points (0.0, 1.0, 2.0, 3.0, 4.0)
        integer_cols = [col for col in df.columns if float(col) % 1 == 0]
        df = df[integer_cols]
        print(f"  Filtered to integer time points: {integer_cols}")
        
        # Row-wise standardization (z-score normalization)
        df_standardized = df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1), axis=0)
        
        data_dict[name] = df_standardized
        print(f"Imported '{name}' with shape: {df.shape} (row-wise standardized)")
    
    print(f"\nSuccessfully imported {len(data_dict)} datasets.")
    print(f"Available datasets: {list(data_dict.keys())}")
    
    return data_dict

# Run when script is executed directly
if __name__ == "__main__":
    data_dict = load_data()