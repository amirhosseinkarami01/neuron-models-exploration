"""Simple data loading utilities."""

import pandas as pd
import numpy as np
import os
from pathlib import Path


def load_training_data(data_path, file_indices, max_rows=None):
    """
    Load training data (input current and spike times).
    
    Args:
        data_path: Path to data directory
        file_indices: List of file indices to load
        max_rows: Maximum number of rows to load (for fast testing)
    
    Returns:
        List of dictionaries with 'current', 'spike_times', and 'index'
    """
    data = []
    
    for idx in file_indices:
        input_file = os.path.join(data_path, f'input_{idx}.csv')
        spikes_file = os.path.join(data_path, f'spikes_{idx}.csv')
        
        # Check if files exist
        if not os.path.exists(input_file) or not os.path.exists(spikes_file):
            print(f"Warning: Files for index {idx} not found")
            continue
        
        # Load input (assuming no header)
        if max_rows:
            input_df = pd.read_csv(input_file, header=None, nrows=max_rows)
        else:
            input_df = pd.read_csv(input_file, header=None)
        
        # Load spikes
        spikes_df = pd.read_csv(spikes_file, header=None)
        
        data.append({
            'index': idx,
            'current': input_df.iloc[1:, 1].values.astype(np.float32),
            'spike_times': spikes_df.iloc[1:, 0].values.astype(np.float32),
            'time': input_df.iloc[1:, 0].values.astype(np.float32) if input_df.shape[1] > 2 else None
        })
    
    return data


def load_test_data(data_path, file_indices, max_rows=None):
    """
    Load test data (input current only).
    
    Args:
        data_path: Path to data directory
        file_indices: List of file indices to load
        max_rows: Maximum number of rows to load
    
    Returns:
        List of dictionaries with 'current' and 'index'
    """
    data = []
    
    for idx in file_indices:
        input_file = os.path.join(data_path, f'input_{idx}.csv')
        
        if not os.path.exists(input_file):
            print(f"Warning: Test file for index {idx} not found")
            continue
        
        if max_rows:
            input_df = pd.read_csv(input_file, header=None, nrows=max_rows)
        else:
            input_df = pd.read_csv(input_file, header=None)
        
        data.append({
            'index': idx,
            'current': input_df.iloc[1:, 1].values.astype(np.float32),
            'time': input_df.iloc[1:, 0].values.astype(np.float32) if input_df.shape[1] > 2 else None
        })
    
    return data


def split_data(data, n_val):
    """
    Split data into training and validation sets.
    
    Args:
        data: List of data dictionaries
        n_val: Number of validation samples
    
    Returns:
        train_data, val_data
    """
    np.random.seed(42)  # For reproducibility
    indices = np.random.permutation(len(data))
    train_indices = indices[:-n_val]
    val_indices = indices[-n_val:]
    
    train_data = [data[i] for i in train_indices]
    val_data = [data[i] for i in val_indices]
    
    return train_data, val_data