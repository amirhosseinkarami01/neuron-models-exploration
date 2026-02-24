"""Fast parameter optimization for neuron models."""

import numpy as np
import random
from models import create_model
from evaluator import evaluate_model_on_data
import torch


def random_search_optimize(model_name, train_data, n_iterations=30, dt=1.0):
    """
    Fast random search for model parameters.
    
    Args:
        model_name: Name of the model to optimize
        train_data: List of training data points
        n_iterations: Number of random parameter sets to try
        dt: Time step
    
    Returns:
        Best parameters found
    """
    print(f"\nOptimizing {model_name} with random search ({n_iterations} iterations)...")
    
    # Define parameter ranges for each model type
    param_ranges = {
        'LIF': {
            'tau_m': (5.0, 25.0),
            'v_th': (-65.0, -45.0),
            'v_reset': (-80.0, -65.0),
            'r_m': (50.0, 200.0)
        },
        'Izhikevich': {
            'a': (0.01, 0.1),
            'b': (0.1, 0.3),
            'c': (-70.0, -50.0),
            'd': (2.0, 10.0)
        },
        'AdEx': {
            'C': (100.0, 300.0),
            'gL': (5.0, 20.0),
            'EL': (-75.0, -65.0),
            'VT': (-55.0, -45.0),
            'a': (1.0, 5.0),
            'b': (40.0, 100.0)
        },
        'SRM': {
            'tau_m': (5.0, 20.0),
            'tau_s': (2.0, 10.0),
            'v_th': (-60.0, -50.0),
            'eta': (-10.0, -2.0)
        },
        'RateBased': {
            'tau': (10.0, 50.0),
            'threshold': (20.0, 50.0)
        }
    }
    
    # Get base model to know parameter names
    base_model = create_model(model_name, dt)
    current_params = base_model.get_params()
    
    # Determine which parameter set to use
    if 'Izhikevich' in model_name:
        ranges = param_ranges['Izhikevich']
    elif model_name in param_ranges:
        ranges = param_ranges[model_name]
    else:
        print(f"No predefined ranges for {model_name}, using defaults")
        return current_params
    
    # Use a subset of training data for faster optimization
    eval_data = train_data[:min(3, len(train_data))]
    
    best_score = -float('inf')
    best_params = current_params.copy()
    
    for i in range(n_iterations):
        # Generate random parameters
        test_params = {}
        for param, (low, high) in ranges.items():
            if param in current_params:
                # Sample around current value
                current_val = current_params.get(param, (low + high) / 2)
                # Random within range
                test_params[param] = random.uniform(low, high)
        
        # Create model with these parameters
        test_model = create_model(model_name, dt)
        
        # Special handling for Izhikevich (needs to preserve type)
        if 'Izhikevich' in model_name:
            # Keep the neuron type
            pass
        
        # Set parameters
        for key, value in test_params.items():
            if hasattr(test_model, key):
                # setattr(test_model, key, value)
                # If it's a Parameter, update properly
                if hasattr(test_model, key) and hasattr(getattr(test_model, key), 'data'):
                    getattr(test_model, key).data = torch.tensor(value, dtype=torch.float32)
                else:
                    setattr(test_model, key, value)
        
        # Evaluate
        total_score = 0
        for data_point in eval_data:
            result = evaluate_model_on_data(test_model, data_point, dt)
            total_score += result['score']
        
        avg_score = total_score / len(eval_data)
        
        if avg_score > best_score:
            best_score = avg_score
            best_params = test_params
            print(f"  Iter {i+1}: New best score = {avg_score:.4f}")
    
    print(f"Best parameters: {best_params}")
    print(f"Best score: {best_score:.4f}")
    
    return best_params


def quick_grid_search(model_name, train_data, dt=1.0):
    """
    Quick grid search for a few key parameters.
    Even faster than random search for simple models.
    """
    print(f"\nQuick grid search for {model_name}...")
    
    if model_name == 'LIF':
        # Just try a few combinations
        tau_values = [8, 10, 12, 15]
        vth_values = [-58, -55, -52]
        
        eval_data = train_data[:2]
        best_score = -float('inf')
        best_params = {}
        
        for tau in tau_values:
            for vth in vth_values:
                params = {'tau_m': tau, 'v_th': vth}
                model = create_model(model_name, dt)
                model.tau_m.data.fill_(tau)
                model.v_th.data.fill_(vth)
                
                total_score = 0
                for data in eval_data:
                    result = evaluate_model_on_data(model, data, dt)
                    total_score += result['score']
                
                avg_score = total_score / len(eval_data)
                
                if avg_score > best_score:
                    best_score = avg_score
                    best_params = params
                    print(f"  tau={tau}, vth={vth}: score={avg_score:.4f} (new best)")
        
        return best_params
    
    else:
        # Fall back to random search
        return random_search_optimize(model_name, train_data, n_iterations=15, dt=dt)