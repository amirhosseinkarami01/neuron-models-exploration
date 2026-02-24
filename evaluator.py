"""Evaluation functions for spike prediction."""

import numpy as np
import pandas as pd


def find_matches(predicted_spikes, real_spikes, tolerance=2.0):
    """
    Find matches between predicted and real spikes.
    
    Args:
        predicted_spikes: Array of predicted spike times
        real_spikes: Array of real spike times
        tolerance: Matching tolerance in ms
    
    Returns:
        Number of matches
    """
    if len(predicted_spikes) == 0 or len(real_spikes) == 0:
        return 0
    
    matches = 0
    i, j = 0, 0
    
    while i < len(real_spikes) and j < len(predicted_spikes):
        diff = predicted_spikes[j] - real_spikes[i]
        
        if abs(diff) <= tolerance:
            matches += 1
            i += 1
            j += 1
        elif diff < 0:
            j += 1
        else:
            i += 1
    
    return matches


def compute_score(matches, real_count, pred_count):
    """
    Compute the scoring metric from the problem statement.
    
    Score = (matches - 0.1 * real_count) / (pred_count + real_count)
    """
    if pred_count + real_count == 0:
        return 0.0
    return (matches - 0.1 * real_count) / (pred_count + real_count)


def evaluate_prediction(predicted_times, real_times, tolerance=2.0):
    """
    Complete evaluation of a prediction.
    
    Returns:
        dict with matches, real_count, pred_count, score
    """
    matches = find_matches(predicted_times, real_times, tolerance)
    real_count = len(real_times)
    pred_count = len(predicted_times)
    score = compute_score(matches, real_count, pred_count)
    
    return {
        'matches': matches,
        'real_spikes': real_count,
        'pred_spikes': pred_count,
        'score': score
    }


def evaluate_model_on_data(model, data_point, dt=1.0, tolerance=2.0):
    """
    Evaluate a model on a single data point.
    
    Args:
        model: Neuron model with simulate() method
        data_point: Dict with 'current' and 'spike_times'
        dt: Time step
        tolerance: Matching tolerance
    
    Returns:
        dict with evaluation results
    """
    # Simulate
    spike_train = model.simulate(data_point['current'])
    predicted_times = np.where(spike_train == 1)[0] * dt
    
    # Evaluate
    result = evaluate_prediction(predicted_times, data_point['spike_times'], tolerance)
    result['file_index'] = data_point['index']
    result['model_name'] = model.name
    
    return result


def summarize_results(results_list):
    """
    Summarize evaluation results across multiple files.
    
    Args:
        results_list: List of result dicts from evaluate_model_on_data
    
    Returns:
        DataFrame with summary
    """
    df = pd.DataFrame(results_list)
    
    if len(df) == 0:
        return pd.DataFrame()
    
    summary = {
        'mean_score': df['score'].mean(),
        'std_score': df['score'].std(),
        'total_matches': df['matches'].sum(),
        'total_real': df['real_spikes'].sum(),
        'total_pred': df['pred_spikes'].sum(),
        'match_ratio': df['matches'].sum() / max(df['real_spikes'].sum(), 1)
    }
    
    # Add per-file details
    details = df[['file_index', 'score', 'matches', 'real_spikes', 'pred_spikes']]
    
    return summary, details