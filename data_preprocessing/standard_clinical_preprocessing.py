import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def apply_clinical_filters(signal, fs=100):
    """
    Applies Bandpass (0.5-40Hz) and Notch (50Hz) filters.
    signal shape: (1200, 12) -> (samples, leads)
    """
    # 1. Bandpass Filter: 0.5–40 Hz
    nyquist = 0.5 * fs
    low, high = 0.5 / nyquist, 40 / nyquist
    b, a = butter(3, [low, high], btype='band')
    # apply along axis 0 (time axis)
    filtered_signal = filtfilt(b, a, signal, axis=0)
    
    # 2. Notch Filter: 50 Hz
    b_notch, a_notch = iirnotch(50, 30, fs)
    filtered_signal = filtfilt(b_notch, a_notch, filtered_signal, axis=0)
    
    return filtered_signal

def run_preprocessing_pipeline(all_signals, labels):
    """
    Handles Splitting, Filtering, and Scaling.
    """
    # 3. Holdout Split: 70% train / 30% test, stratified by class
    X_train, X_test, y_train, y_test = train_test_split(
        all_signals, 
        labels, 
        test_size=0.30, 
        random_state=42, 
        stratify=labels
    )

    # 4. Apply Filters to all splits
    # We apply filters individually to each 12-lead recording
    X_train_filt = np.array([apply_clinical_filters(record) for record in X_train])
    X_test_filt = np.array([apply_clinical_filters(record) for record in X_test])

    # 5. Feature Scaling: StandardScaler applied per-lead
    # StandardScaler expects (n_samples, n_features). 
    # We treat each (timestep x lead) as a feature or flatten.
    # Standard practice: Scale across all timesteps for each lead.
    
    scaler = StandardScaler()
    
    # Reshape to (N * 1200, 12) to fit scaler on lead-columns
    N_train, T, L = X_train_filt.shape
    N_test = X_test_filt.shape[0]
    
    X_train_reshaped = X_train_filt.reshape(-1, L)
    X_test_reshaped = X_test_filt.reshape(-1, L)

    # Fit ON TRAIN ONLY, transform on both
    X_train_scaled = scaler.fit_transform(X_train_reshaped)
    X_test_scaled = scaler.transform(X_test_reshaped)

    # Reshape back to original 3D format (N, 1200, 12)
    X_train_final = X_train_scaled.reshape(N_train, T, L)
    X_test_final = X_test_scaled.reshape(N_test, T, L)

    # 6. Save as dataset_v2_filtered.npy
    output_data = {
        'X_train': X_train_final,
        'X_test': X_test_final,
        'y_train': y_train,
        'y_test': y_test
    }
    np.save('data_preprocessing/dataset_v2_filtered.npy', output_data)
    print("Preprocessing complete. File saved as dataset_v2_filtered.npy")

    return X_train_final, X_test_final