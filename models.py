"""All neuron models in one file."""

import numpy as np
import torch
import torch.nn as nn


class BaseNeuron(nn.Module):
    """Base class for all neuron models."""
    
    def __init__(self, dt=1.0):
        super().__init__()
        self.dt = dt
        self.name = "Base"
    
    def forward(self, I_input):
        """Alias for simulate."""
        return self.simulate(I_input)
    
    def simulate(self, I_input):
        """Simulate neuron response. To be overridden."""
        raise NotImplementedError
    
    def get_params(self):
        """Return current parameters as dict."""
        return {}
    
    def set_params(self, params_dict):
        """Set parameters from dict."""
        for key, value in params_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)


class LIFNeuron(BaseNeuron):
    """
    Leaky Integrate-and-Fire neuron.
    
    Parameters:
        dt: time step (ms)
        tau_m: membrane time constant (ms)
        v_rest: resting potential (mV)
        v_th: threshold potential (mV)
        v_reset: reset potential after spike (mV)
        r_m: membrane resistance (MΩ)
        refractory: refractory period (ms)
    """
    
    def __init__(self, dt=1.0, tau_m=10.0, v_rest=-70.0, v_th=-55.0,
                 v_reset=-75.0, r_m=100.0, refractory=2.0):
        super().__init__(dt)
        self.name = "LIF"
        self.tau_m = tau_m
        self.v_rest = v_rest
        self.v_th = v_th
        self.v_reset = v_reset
        self.r_m = r_m
        self.refractory = refractory
        
        # For PyTorch compatibility
        self.tau_m = nn.Parameter(torch.tensor(tau_m, dtype=torch.float32))
        self.v_th = nn.Parameter(torch.tensor(v_th, dtype=torch.float32))
        self.v_reset = nn.Parameter(torch.tensor(v_reset, dtype=torch.float32))
        self.r_m = nn.Parameter(torch.tensor(r_m, dtype=torch.float32))
        self.v_rest = nn.Parameter(torch.tensor(v_rest, dtype=torch.float32))
        self.refractory = nn.Parameter(torch.tensor(refractory, dtype=torch.float32))
    
    def simulate(self, I_input):
        """Simulate LIF neuron."""
        # Convert to numpy if it's a tensor
        if torch.is_tensor(I_input):
            I_input = I_input.detach().cpu().numpy()
        
        n_steps = len(I_input)
        v = np.ones(n_steps) * self.v_rest.detach().cpu().numpy()
        spikes = np.zeros(n_steps)
        
        refractory_remaining = 0
        I_nA = I_input / 1000.0  # Convert pA to nA
        
        v_th_val = self.v_th.detach().cpu().numpy()
        v_reset_val = self.v_reset.detach().cpu().numpy()
        tau_m_val = self.tau_m.detach().cpu().numpy()
        r_m_val = self.r_m.detach().cpu().numpy()
        v_rest_val = self.v_rest.detach().cpu().numpy()
        refractory_val = self.refractory.detach().cpu().numpy()

        
        for t in range(1, n_steps):
            if refractory_remaining > 0:
                v[t] = v_reset_val
                refractory_remaining -= 1
            else:
                dv = (v_rest_val - v[t-1] + r_m_val * I_nA[t-1]) / tau_m_val * self.dt
                v[t] = v[t-1] + dv
                
                if v[t] >= v_th_val:
                    spikes[t] = 1
                    v[t] = v_reset_val
                    refractory_remaining = refractory_val / self.dt
        
        return spikes
    
    def get_params(self):
        return {
            'tau_m': self.tau_m.item(),
            'v_th': self.v_th.item(),
            'v_reset': self.v_reset.item(),
            'r_m': self.r_m.item(),
            'refractory': self.refractory
        }


class IzhikevichNeuron(BaseNeuron):
    """
    Izhikevich neuron model - can produce various firing patterns.
    
    Neuron types:
        'rs': Regular spiking
        'ib': Intrinsically bursting
        'ch': Chattering
        'fs': Fast spiking
    """
    
    def __init__(self, dt=1.0, neuron_type='rs'):
        super().__init__(dt)
        self.name = f"Izhikevich_{neuron_type}"
        self.neuron_type = neuron_type
        
        # Parameters for different neuron types (standard values)
        if neuron_type == 'rs':  # Regular spiking
            self.a, self.b, self.c, self.d = 0.02, 0.2, -65.0, 8.0
        elif neuron_type == 'ib':  # Intrinsically bursting
            self.a, self.b, self.c, self.d = 0.02, 0.2, -55.0, 4.0
        elif neuron_type == 'ch':  # Chattering
            self.a, self.b, self.c, self.d = 0.02, 0.2, -50.0, 2.0
        elif neuron_type == 'fs':  # Fast spiking
            self.a, self.b, self.c, self.d = 0.1, 0.2, -65.0, 2.0
        else:
            self.a, self.b, self.c, self.d = 0.02, 0.2, -65.0, 8.0
        
        # Make them PyTorch parameters
        self.a = nn.Parameter(torch.tensor(self.a, dtype=torch.float32))
        self.b = nn.Parameter(torch.tensor(self.b, dtype=torch.float32))
        self.c = nn.Parameter(torch.tensor(self.c, dtype=torch.float32))
        self.d = nn.Parameter(torch.tensor(self.d, dtype=torch.float32))
        
        self.v_th = 30.0  # Spike cutoff
    
    def simulate(self, I_input):
        """Simulate Izhikevich neuron."""
        if torch.is_tensor(I_input):
            I_input = I_input.detach().cpu().numpy()
        
        n_steps = len(I_input)
        v = np.ones(n_steps) * -65.0
        u = np.ones(n_steps) * self.b.detach().cpu().numpy() * -65.0
        spikes = np.zeros(n_steps)
        
        # Scale current appropriately for Izhikevich model
        I_scaled = I_input / 1000.0 * 100
        
        a_val = self.a.detach().cpu().numpy()
        b_val = self.b.detach().cpu().numpy()
        c_val = self.c.detach().cpu().numpy()
        d_val = self.d.detach().cpu().numpy()
        
        for t in range(1, n_steps):
            # Izhikevich equations
            dv = (0.04 * v[t-1]**2 + 5 * v[t-1] + 140 - u[t-1] + I_scaled[t-1]) * self.dt
            du = (a_val * (b_val * v[t-1] - u[t-1])) * self.dt
            
            v[t] = v[t-1] + dv
            u[t] = u[t-1] + du
            
            # Spike detection
            if v[t] >= self.v_th:
                spikes[t] = 1
                v[t] = c_val
                u[t] = u[t] + d_val
        
        return spikes
    
    def get_params(self):
        return {
            'a': self.a.item(),
            'b': self.b.item(),
            'c': self.c.item(),
            'd': self.d.item(),
            'type': self.neuron_type
        }


class AdExNeuron(BaseNeuron):
    """
    Adaptive Exponential Integrate-and-Fire model.
    More biophysically realistic.
    """
    
    def __init__(self, dt=1.0):
        super().__init__(dt)
        self.name = "AdEx"
        
        # Parameters
        self.C = nn.Parameter(torch.tensor(200.0))   # Capacitance (pF)
        self.gL = nn.Parameter(torch.tensor(10.0))   # Leak conductance (nS)
        self.EL = nn.Parameter(torch.tensor(-70.0))  # Leak reversal (mV)
        self.VT = nn.Parameter(torch.tensor(-50.0))  # Threshold (mV)
        self.deltaT = nn.Parameter(torch.tensor(2.0))  # Slope factor (mV)
        self.a = nn.Parameter(torch.tensor(2.0))     # Adaptation coupling (nS)
        self.b = nn.Parameter(torch.tensor(60.0))    # Spike-triggered adaptation (pA)
        self.tauw = nn.Parameter(torch.tensor(100.0))  # Adaptation time constant (ms)
        self.Vr = nn.Parameter(torch.tensor(-70.0))  # Reset voltage (mV)
        self.Vcut = 30.0  # Cutoff voltage (mV)
    
    def simulate(self, I_input):
        """Simulate AdEx neuron."""
        if torch.is_tensor(I_input):
            I_input = I_input.detach().cpu().numpy()
        
        n_steps = len(I_input)
        v = np.ones(n_steps) * self.EL.detach().cpu().numpy()
        w = np.zeros(n_steps)  # Adaptation current
        spikes = np.zeros(n_steps)
        
        # Get parameter values
        C_val = self.C.detach().cpu().numpy()
        gL_val = self.gL.detach().cpu().numpy()
        EL_val = self.EL.detach().cpu().numpy()
        VT_val = self.VT.detach().cpu().numpy()
        deltaT_val = self.deltaT.detach().cpu().numpy()
        a_val = self.a.detach().cpu().numpy()
        b_val = self.b.detach().cpu().numpy()
        tauw_val = self.tauw.detach().cpu().numpy()
        Vr_val = self.Vr.detach().cpu().numpy()
        
        for t in range(1, n_steps):
            # Membrane potential dynamics
            dv = (-gL_val * (v[t-1] - EL_val) + 
                  gL_val * deltaT_val * np.exp((v[t-1] - VT_val) / deltaT_val) - 
                  w[t-1] + I_input[t-1]) / C_val * self.dt
            
            # Adaptation dynamics
            dw = ((a_val * (v[t-1] - EL_val) - w[t-1]) / tauw_val) * self.dt
            
            v[t] = v[t-1] + dv
            w[t] = w[t-1] + dw
            
            # Spike detection
            if v[t] >= self.Vcut:
                spikes[t] = 1
                v[t] = Vr_val
                w[t] = w[t] + b_val
        
        return spikes
    
    def get_params(self):
        return {
            'C': self.C.item(),
            'gL': self.gL.item(),
            'EL': self.EL.item(),
            'VT': self.VT.item(),
            'deltaT': self.deltaT.item(),
            'a': self.a.item(),
            'b': self.b.item(),
            'tauw': self.tauw.item(),
            'Vr': self.Vr.item()
        }


class SRMNeuron(BaseNeuron):
    """
    Spike Response Model (SRM).
    """
    
    def __init__(self, dt=1.0):
        super().__init__(dt)
        self.name = "SRM"
        
        self.tau_m = nn.Parameter(torch.tensor(10.0))   # Membrane time constant
        self.tau_s = nn.Parameter(torch.tensor(5.0))    # Synaptic time constant
        self.v_th = nn.Parameter(torch.tensor(-55.0))   # Threshold
        self.v_reset = nn.Parameter(torch.tensor(-70.0))
        self.eta = nn.Parameter(torch.tensor(-5.0))     # Spike after-potential
        self.refractory = 3.0
    
    def simulate(self, I_input):
        """Simulate SRM neuron."""
        if torch.is_tensor(I_input):
            I_input = I_input.detach().cpu().numpy()
        
        n_steps = len(I_input)
        v = np.ones(n_steps) * self.v_reset.detach().cpu().numpy()
        spikes = np.zeros(n_steps)
        
        # Precompute filter (exponential kernel)
        t = np.arange(0, min(50, n_steps), self.dt)
        kernel = np.exp(-t/self.tau_m.detach().cpu().numpy()) - np.exp(-t/self.tau_s.detach().cpu().numpy())
        kernel = kernel / (np.sum(kernel) + 1e-10)  # Normalize
        
        # Filter input
        I_filtered = np.convolve(I_input, kernel, mode='same')
        
        v_th_val = self.v_th.detach().cpu().numpy()
        v_reset_val = self.v_reset.detach().cpu().numpy()
        eta_val = self.eta.detach().cpu().numpy()
        
        refractory_remaining = 0
        
        for t in range(1, n_steps):
            if refractory_remaining > 0:
                v[t] = v_reset_val + eta_val * np.exp(-refractory_remaining/10.0)
                refractory_remaining -= 1
            else:
                v[t] = v_reset_val + I_filtered[t] / 100.0
                
                if v[t] >= v_th_val:
                    spikes[t] = 1
                    v[t] = v_reset_val
                    refractory_remaining = self.refractory / self.dt
        
        return spikes
    
    def get_params(self):
        return {
            'tau_m': self.tau_m.item(),
            'tau_s': self.tau_s.item(),
            'v_th': self.v_th.item(),
            'v_reset': self.v_reset.item(),
            'eta': self.eta.item()
        }


class RateBasedNeuron(BaseNeuron):
    """
    Simple rate-based model with Poisson spike generation.
    """
    
    def __init__(self, dt=1.0):
        super().__init__(dt)
        self.name = "RateBased"
        
        self.tau = nn.Parameter(torch.tensor(20.0))      # Time constant
        self.threshold = nn.Parameter(torch.tensor(30.0)) # Firing threshold
        self.refractory = 2.0
        self.seed = 42  # For reproducibility
    
    def simulate(self, I_input):
        """Simulate rate-based neuron."""
        if torch.is_tensor(I_input):
            I_input = I_input.detach().cpu().numpy()
        
        n_steps = len(I_input)
        rate = np.zeros(n_steps)
        spikes = np.zeros(n_steps)
        
        tau_val = self.tau.detach().cpu().numpy()
        threshold_val = self.threshold.detach().cpu().numpy()
        
        # Low-pass filter the input
        alpha = self.dt / tau_val
        for t in range(1, n_steps):
            rate[t] = rate[t-1] + alpha * (I_input[t] - rate[t-1])
        
        # Convert to spikes using Poisson process
        np.random.seed(self.seed)
        firing_prob = np.clip(rate / threshold_val, 0, 1) * self.dt / 10
        
        refractory_remaining = 0
        for t in range(n_steps):
            if refractory_remaining > 0:
                refractory_remaining -= 1
            else:
                if np.random.random() < firing_prob[t]:
                    spikes[t] = 1
                    refractory_remaining = self.refractory / self.dt
        
        return spikes
    
    def get_params(self):
        return {
            'tau': self.tau.item(),
            'threshold': self.threshold.item()
        }


# Factory function to create models
def create_model(model_name, dt=1.0):
    """
    Create a neuron model by name.
    
    Args:
        model_name: Name of the model (e.g., 'LIF', 'Izhikevich_RS')
        dt: Time step
    
    Returns:
        Neuron model instance
    """
    if model_name == 'LIF':
        return LIFNeuron(dt=dt)
    elif model_name.startswith('Izhikevich'):
        if '_' in model_name:
            neuron_type = model_name.split('_')[1].lower()
        else:
            neuron_type = 'rs'
        return IzhikevichNeuron(dt=dt, neuron_type=neuron_type)
    elif model_name == 'AdEx':
        return AdExNeuron(dt=dt)
    elif model_name == 'SRM':
        return SRMNeuron(dt=dt)
    elif model_name == 'RateBased':
        return RateBasedNeuron(dt=dt)
    else:
        raise ValueError(f"Unknown model: {model_name}")