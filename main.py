#!/usr/bin/env python
"""
Main script for neural spike prediction.

This script runs experiments with different neuron models and evaluates their
performance on predicting spike times from input current.
"""

import os
import argparse
import numpy as np
import torch

# Import our modules
import config
from data_loader import load_training_data, load_test_data, split_data
from models import create_model, LIFNeuron, IzhikevichNeuron, AdExNeuron, SRMNeuron, RateBasedNeuron
from evaluator import evaluate_model_on_data, summarize_results
from optimizer import random_search_optimize, quick_grid_search
from utils import save_predictions, plot_comparison, plot_model_comparison, quick_summary


def run_experiment(model_name, train_data, val_data, optimize=True):
    """
    Run a single experiment with a specific model.
    
    Args:
        model_name: Name of the model to use
        train_data: Training data
        val_data: Validation data
        optimize: Whether to optimize parameters
    
    Returns:
        Model, validation results
    """
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {model_name}")
    print(f"{'='*60}")
    
    # Create model
    model = create_model(model_name, dt=config.DT)
    print(f"Initial parameters: {model.get_params()}")
    
    # Optimize if requested
    if optimize and len(train_data) > 0:
        print("\nOptimizing parameters...")
        # Use random search (faster)
        best_params = random_search_optimize(
            model_name, 
            train_data, 
            n_iterations=config.OPTIMIZATION_ITERATIONS,
            dt=config.DT
        )
        
        # Update model with best parameters
        for key, value in best_params.items():
            if hasattr(model, key):
                # If it's a Parameter, update properly
                if hasattr(getattr(model, key), 'data'):
                    getattr(model, key).data = torch.tensor(value, dtype=torch.float32)
                else:
                    setattr(model, key, value)
        print(f"Optimized parameters: {model.get_params()}")
    
    # Validate
    print("\nValidating...")
    val_results = []
    for data_point in val_data:
        result = evaluate_model_on_data(model, data_point, config.DT, config.MATCH_TOLERANCE)
        val_results.append(result)
        print(f"  File {result['file_index']}: Score={result['score']:.4f}, "
              f"Matches={result['matches']}, Pred={result['pred_spikes']}")
    
    # Summary
    summary, details = summarize_results(val_results)
    print(f"\nValidation Summary:")
    print(f"  Mean Score: {summary['mean_score']:.4f} ± {summary['std_score']:.4f}")
    print(f"  Total Matches: {summary['total_matches']}/{summary['total_real']} "
          f"({summary['match_ratio']:.2%})")
    
    return model, val_results


def predict_test(model, test_data, output_path):
    """
    Generate predictions for test data.
    
    Args:
        model: Trained neuron model
        test_data: List of test data points
        output_path: Directory to save predictions
    """
    print(f"\nGenerating predictions for {len(test_data)} test files...")
    
    for data_point in test_data:
        # Simulate
        spike_train = model.simulate(data_point['current'])
        predicted_times = np.where(spike_train == 1)[0] * config.DT
        
        # Save
        save_predictions(predicted_times, data_point['index'], output_path)
        print(f"  File {data_point['index']}: {len(predicted_times)} spikes")


def compare_all_models(train_data, val_data):
    """
    Compare all available models without optimization.
    Quick comparison to see which models perform best.
    """
    print("\n" + "="*60)
    print("COMPARING ALL MODELS (no optimization)")
    print("="*60)
    
    all_results = []
    
    for model_name in config.AVAILABLE_MODELS:
        print(f"\nTesting {model_name}...")
        model = create_model(model_name, dt=config.DT)
        
        for data_point in val_data[:2]:  # Use only first 2 validation files for speed
            result = evaluate_model_on_data(model, data_point, config.DT, config.MATCH_TOLERANCE)
            all_results.append(result)
            print(f"  File {result['file_index']}: Score={result['score']:.4f}")
    
    # Create DataFrame for analysis
    import pandas as pd
    results_df = pd.DataFrame(all_results)
    
    # Print summary
    print("\n" + "-"*40)
    print("Summary by model:")
    summary = results_df.groupby('model_name')['score'].agg(['mean', 'std', 'count'])
    print(summary.round(4))
    
    # Find best model
    best_model = summary['mean'].idxmax()
    best_score = summary['mean'].max()
    print(f"\n🏆 Best model: {best_model} with score {best_score:.4f}")
    
    return results_df


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Neural Spike Prediction')
    parser.add_argument('--model', type=str, default=config.DEFAULT_MODEL,
                        choices=config.AVAILABLE_MODELS + ['all'],
                        help='Model to use')
    parser.add_argument('--optimize', action='store_true', default=True,
                        help='Optimize parameters')
    parser.add_argument('--quick', action='store_true', default=False,
                        help='Quick mode (use fewer files)')
    parser.add_argument('--compare', action='store_true', default=False,
                        help='Compare all models')
    
    args = parser.parse_args()
    
    print("="*60)
    print("NEURAL SPIKE PREDICTION")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Optimize: {args.optimize}")
    print(f"Quick mode: {args.quick}")
    print("="*60)
    
    # Adjust parameters for quick mode
    if args.quick:
        n_train = 8
        n_val = 2
        max_rows = 500
    else:
        n_train = config.N_TRAIN_FILES - config.N_VAL_FILES
        n_val = config.N_VAL_FILES
        max_rows = config.MAX_ROWS
    
    # Load data
    print("\nLoading training data...")
    all_data = load_training_data(
        config.TRAIN_DATA_PATH, 
        range(1, n_train + n_val + 1),
        max_rows=max_rows
    )
    
    if len(all_data) == 0:
        print("Error: No training data found!")
        return
    
    # Split data
    train_data, val_data = split_data(all_data, n_val)
    print(f"Loaded {len(all_data)} files: {len(train_data)} train, {len(val_data)} validation")
    
    # Load test data
    print("\nLoading test data...")
    test_data = load_test_data(
        config.TEST_DATA_PATH,
        range(1, 11),  # Assuming 10 test files
        max_rows=max_rows
    )
    print(f"Loaded {len(test_data)} test files")
    
    # Compare all models if requested
    if args.compare:
        results_df = compare_all_models(train_data, val_data)
        plot_model_comparison(results_df, save_path='model_comparison.png')
        return
    
    # Run experiment with specific model
    if args.model == 'all':
        # Run all models sequentially
        all_val_results = []
        
        for model_name in config.AVAILABLE_MODELS:
            model, val_results = run_experiment(
                model_name, 
                train_data, 
                val_data, 
                optimize=args.optimize
            )
            all_val_results.extend(val_results)
            
            # Predict test for this model
            predict_test(model, test_data, f"{config.OUTPUT_PATH}/{model_name}")
        
        # Plot comparison
        if all_val_results:
            import pandas as pd
            results_df = pd.DataFrame(all_val_results)
            plot_model_comparison(results_df, save_path='all_models_comparison.png')
    
    else:
        # Run single model
        model, val_results = run_experiment(
            args.model,
            train_data,
            val_data,
            optimize=args.optimize
        )
        
        # Predict test
        predict_test(model, test_data, config.OUTPUT_PATH)
        
        # Quick summary
        quick_summary(args.model, val_results)
        
        # Plot first validation file
        if val_results and len(val_data) > 0:
            # Find the real spike times for the first validation file
            first_file_idx = val_data[0]['index']
            real_times = val_data[0]['spike_times']
            
            # Get predictions for this file
            spike_train = model.simulate(val_data[0]['current'])
            pred_times = np.where(spike_train == 1)[0] * config.DT
            
            plot_comparison(
                real_times, 
                pred_times,
                title=f"{args.model} - File {first_file_idx}",
                save_path=f"{args.model}_comparison.png"
            )
    
    print(f"\nDone! Predictions saved to {config.OUTPUT_PATH}/")


if __name__ == "__main__":
    main()