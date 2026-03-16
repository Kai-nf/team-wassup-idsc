# FILE: data_loader.py

import pandas as pd
import numpy as np
import wfdb

def load_raw_dataset(metadata_csv_path, data_dir):
    metadata = pd.read_csv(metadata_csv_path)
    signals = []
    
    for _, row in metadata.iterrows():
        record = row['patient_id']
        record_path = f"{data_dir}/{record}"
        signal, _ = wfdb.rdsamp(record_path)
        signals.append(signal)
    
    # Stack signals into a numpy array
    signals_array = np.array([signal for signal in signals])
    
    return signals_array

if __name__ == "__main__":
    # Example usage
    metadata_path = 'metadata.csv'
    data_directory = 'files'
    signals = load_raw_dataset(metadata_path, data_directory)
    print(signals.shape)  # Should print the shape of the loaded signals array