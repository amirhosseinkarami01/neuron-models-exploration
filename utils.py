"""Utility functions for visualization and saving results."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


def save_predictions(predicted_times, file_index, output_path):
    """
    Save predicted spike times to CSV.
    
    Args:
        predicted_times: Array of predicted spike times
        file_index: Index of the test file
        output_path: Directory to save to
    """
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, f'spikes_{file_index}.csv')
    pd.DataFrame({'spike_times': predicted_times}).to_csv(output_file, index=False)


def plot_comparison(real_times, pred_times, title="Spike Comparison", save_path=None):
    """
    Simple plot comparing real and predicted spikes.
    """
    fig, ax = plt.subplots(figsize=(10, 3))
    
    # Plot as raster
    ax.eventplot([real_times, pred_times], colors=['g', 'r'], 
                 lineoffsets=[0.5, -0.5], label=['Real', 'Predicted'])
    
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Spikes')
    ax.set_title(title)
    ax.set_ylim(-1, 2)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100)
    plt.show()


def plot_model_comparison(results_df, save_path=None):
    """
    Simple comparison plot for multiple models.
    """
    if results_df.empty:
        print("No results to plot")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Average scores
    ax = axes[0]
    avg_scores = results_df.groupby('model_name')['score'].mean().sort_values()
    colors = plt.cm.viridis(np.linspace(0, 1, len(avg_scores)))
    ax.barh(avg_scores.index, avg_scores.values, color=colors)
    ax.set_xlabel('Average Score')
    ax.set_title('Model Performance')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, v in enumerate(avg_scores.values):
        ax.text(v + 0.01, i, f'{v:.3f}', va='center')
    
    # Match ratio
    ax = axes[1]
    grouped = results_df.groupby('model_name').agg({
        'matches': 'sum',
        'real_spikes': 'sum'
    })
    match_ratio = grouped['matches'] / grouped['real_spikes']
    match_ratio = match_ratio.sort_values()
    
    ax.barh(match_ratio.index, match_ratio.values, color=colors)
    ax.set_xlabel('Match Ratio (Matches / Real Spikes)')
    ax.set_title('Detection Rate')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, v in enumerate(match_ratio.values):
        ax.text(v + 0.01, i, f'{v:.3f}', va='center')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100)
    plt.show()


def quick_summary(model_name, results):
    """
    Print a quick summary of results.
    """
    print("\n" + "=" * 50)
    print(f"RESULTS FOR {model_name}")
    print("=" * 50)
    
    for i, res in enumerate(results):
        print(f"File {res['file_index']}: Score={res['score']:.3f}, "
              f"Matches={res['matches']}, Real={res['real_spikes']}, "
              f"Pred={res['pred_spikes']}")
    
    avg_score = np.mean([r['score'] for r in results])
    total_matches = sum(r['matches'] for r in results)
    total_real = sum(r['real_spikes'] for r in results)
    
    print("-" * 50)
    print(f"Average Score: {avg_score:.4f}")
    print(f"Total Matches: {total_matches}/{total_real} ({total_matches/total_real:.2%})")
    print("=" * 50)