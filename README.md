# Neuron-Models-Exploration

# Neuron Models PyTorch

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-ee4c2c.svg)](https://pytorch.org)

A simple educational playground implementing classic computational neuron models in PyTorch.

## 🧠 What is this?

This repository contains PyTorch implementations of fundamental neuron models from computational neuroscience. The focus is on **understanding how different models work**, not on achieving perfect predictions.

## 📦 Models Included

| Model | Description |
|-------|-------------|
| **LIF** | Leaky Integrate-and-Fire - the classic spiking neuron |
| **Izhikevich** | Efficient model with multiple firing patterns (RS, FS, IB, CH) |
| **AdEx** | Adaptive Exponential - biophysically-inspired with adaptation |
| **SRM** | Spike Response Model - kernel-based phenomenological model |
| **RateBased** | Simple rate model with Poisson spike generation |

## 🚀 Quick Start

### Install

```bash
git clone https://github.com/yourusername/neuron-models-pytorch.git
cd neuron-models-pytorch
pip install numpy pandas torch matplotlib
```

### Run a model

```python
from models import IzhikevichNeuron
import numpy as np

# Create a regular spiking Izhikevich neuron
neuron = IzhikevichNeuron(dt=1.0, neuron_type='rs')

# Input current (pA)
current = np.ones(1000) * 10.0  # 10 pA step

# Simulate
spikes = neuron.simulate(current)
spike_times = np.where(spikes == 1)[0]
print(f"Spikes at: {spike_times} ms")
```

## 📁 Project Structure

```
neuron-models-pytorch/
├── models.py          # All neuron model implementations
├── data_loader.py     # Load input/spike data from CSV
├── evaluator.py       # Simple spike matching utilities
├── optimizer.py       # Basic parameter search
├── utils.py           # Plotting helpers
├── config.py          # Settings
└── main.py            # Run experiments
```

## 💻 Usage

### Try different models

```bash
# Run with default LIF
python main.py

# Try Izhikevich (regular spiking)
python main.py --model Izhikevich_RS

# Try Adaptive Exponential
python main.py --model AdEx

# Compare all models quickly
python main.py --compare
```

### Create your own model

1. Open `models.py`
2. Create a class inheriting from `BaseNeuron`
3. Implement the `simulate()` method

```python
class MyNeuron(BaseNeuron):
    def __init__(self, dt=1.0):
        super().__init__(dt)
        self.name = "MyModel"
    
    def simulate(self, I_input):
        # Your dynamics here
        spikes = np.zeros(len(I_input))
        return spikes
```

## 📊 Data Format

Place your data in:
```
data/train/input_1.csv
data/train/spikes_1.csv
data/test/input_1.csv
```

CSV files (no headers):
- **Input**: `time(ms), current(pA), voltage(mV)` or just current
- **Spikes**: `spike_time(ms)`

## 🎯 Why PyTorch?

Using PyTorch makes it easy to:
- Extend models with learnable parameters
- Leverage automatic differentiation
- Scale to batches of data
- Connect with deep learning pipelines

But the implementations are simple and readable.

## 📚 References

- [Izhikevich, 2003](https://www.izhikevich.org/publications/spikes.htm)
- [Gerstner & Kistler, 2002](https://neuronal-dynamics.epfl.ch/)
- [Dayan & Abbott, 2001](http://www.gatsby.ucl.ac.uk/~dayan/book/)

## 📄 License

MIT - use freely, learn freely.

---

**Built for learning computational neuroscience through code.**
```

This version:
- **Focuses on the models**, not the scoring
- **Simple and direct** language
- **Clear code examples**
- **Honest about purpose** (educational)
- **Easy to scan** with minimal sections
