import wfdb
import pandas as pd
import numpy as np

def load_raw_dataset(csv_path, data_dir):
    metadata = pd.read_csv(csv_path)
    # 363 records, 1200 samples (12s x 100Hz), 12 leads
    all_signals = np.zeros((len(metadata), 1200, 12))
    
    for i, pid in enumerate(metadata['patient_id']):
        # Path: files/188981/188981
        record = wfdb.rdrecord(f"{data_dir}/{pid}/{pid}")
        all_signals[i] = record.p_signal
        
    return all_signals, metadata['brugada'].values