"""
Cognitive models for risky decision-making experiments.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy.optimize import minimize
from scipy.stats import bernoulli
from scipy.special import expit
import multiprocessing as mp

class ExpectedValueModel:
    """Expected Value model for risky decision-making."""
    
    def __init__(self, params: Optional[Dict[str, float]] = None):
        """
        Initialize the Expected Value model.
        
        Args:
            params: Optional model parameters (tau - temperature parameter)
        """
        self.params = params or {"tau": 1.0}
    
    def predict(self, p1: np.ndarray, v1: np.ndarray, p2: np.ndarray, v2: np.ndarray) -> np.ndarray:
        """
        Make predictions using Expected Value model with softmax.
        
        Args:
            p1: Probabilities for option 1 [batch_size, n_outcomes]
            v1: Values for option 1 [batch_size, n_outcomes]
            p2: Probabilities for option 2 [batch_size, n_outcomes]
            v2: Values for option 2 [batch_size, n_outcomes]
            
        Returns:
            Array of choice probabilities for option 2 (0-1 values)
        """
        # Calculate expected values
        ev1 = np.sum(p1 * v1, axis=1)
        ev2 = np.sum(p2 * v2, axis=1)
        
        # Apply softmax with temperature parameter
        tau = self.params["tau"]
        ev = np.stack([ev1, ev2], axis=1)
        exp_values = np.exp(tau * ev)
        choice_probs = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        
        # Return probability of choosing option 2
        return choice_probs[:, 1]
    
    def predict_choices(self, p1: np.ndarray, v1: np.ndarray, p2: np.ndarray, v2: np.ndarray) -> np.ndarray:
        """
        Make hard choice predictions (0 or 1) using Expected Value model.
        
        Args:
            p1: Probabilities for option 1 [batch_size, n_outcomes]
            v1: Values for option 1 [batch_size, n_outcomes]
            p2: Probabilities for option 2 [batch_size, n_outcomes]
            v2: Values for option 2 [batch_size, n_outcomes]
            
        Returns:
            Array of predictions (0 or 1)
        """
        choice_probs = self.predict(p1, v1, p2, v2)
        return (choice_probs > 0.5).astype(int)
    
    def _ev_likelihood(self, params, choices, p1, v1, p2, v2):
        """
        Calculate negative log likelihood for EV model (matching legacy implementation).
        """
        tau_raw = params[0]
        
        # Transform parameter using sigmoid function
        tau = expit(tau_raw) * 100  # Range: (0, 100)
        
        total_log_likelihood = 0
        
        # Calculate log likelihood for each trial
        for trial_choices, trial_p1, trial_v1, trial_p2, trial_v2 in zip(choices, p1, v1, p2, v2):
            # Transform the list to numpy array and normalize (matching legacy preprocessing)
            trial_p1 = np.array(trial_p1) / 100
            trial_v1 = np.array(trial_v1) / 1000
            trial_p2 = np.array(trial_p2) / 100
            trial_v2 = np.array(trial_v2) / 1000
            
            ev1 = np.sum(trial_p1 * trial_v1)
            ev2 = np.sum(trial_p2 * trial_v2)
            
            ev = np.array([ev1, ev2])
            p_choice = np.exp(tau * ev) / np.sum(np.exp(tau * ev))
            
            # Use bernoulli logpmf with p_choice[0] (probability of choosing option 1)
            trial_log_likelihood = np.sum(bernoulli.logpmf(trial_choices, p_choice[0]))
            total_log_likelihood += trial_log_likelihood
        
        # Return negative total log likelihood for minimization
        return -total_log_likelihood
    
    def _optimize_single(self, args):
        """Single optimization run (for multiprocessing)."""
        init_params, choices, p1, v1, p2, v2 = args
        bounds = [(-10, 10)]  # Parameter bounds for raw tau
        
        result = minimize(
            self._ev_likelihood,
            init_params,
            args=(choices, p1, v1, p2, v2),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000}
        )
        return result.fun, result.x
    
    def fit(self, p1: List, v1: List, p2: List, v2: List, choices: List, 
            n_repeat: int = 100, num_cores: Optional[int] = None) -> Dict[str, float]:
        """
        Fit model parameters to data using the legacy fitting procedure.
        
        Args:
            p1: List of probability arrays for option 1
            v1: List of value arrays for option 1
            p2: List of probability arrays for option 2
            v2: List of value arrays for option 2
            choices: List of choices (0 or 1)
            n_repeat: Number of random initializations (default: 100)
            num_cores: Number of CPU cores to use (default: cpu_count - 1)
            
        Returns:
            Dictionary of fitted parameters and negative log likelihood
        """
        if num_cores is None:
            num_cores = mp.cpu_count() - 1
        
        # Generate random initializations (matching legacy: 20 * rand - 10)
        args_list = [(20 * np.random.rand(1) - 10, choices, p1, v1, p2, v2) for _ in range(n_repeat)]
        
        # Run optimization with multiprocessing
        with mp.Pool(num_cores) as pool:
            results = pool.map(self._optimize_single, args_list)
        
        # Get the minimum negative log likelihood (filtering out NaN values)
        best_ll, best_params = min(results, key=lambda x: x[0] if not np.isnan(x[0]) else float('inf'))
        
        # Transform best parameters back to original scales
        tau = expit(best_params[0]) * 100
        
        # Update model parameters
        self.params = {"tau": tau}
        
        return {
            'best_params': {'tau': tau},
            'neg_log_likelihood': best_ll
        }

class ProspectTheoryModel:
    """Prospect Theory model for risky decision-making."""
    
    def __init__(self, params: Optional[Dict[str, float]] = None):
        """
        Initialize the Prospect Theory model.
        
        Args:
            params: Optional model parameters (alpha, lambda, gamma, beta, tau)
        """
        self.params = params or {
            "alpha": 0.88,  # risk attitude for gains
            "beta": 0.88,   # risk attitude for losses
            "lambda": 2.25,  # loss aversion
            "gamma": 0.65,   # probability weighting
            "tau": 1.0      # temperature parameter
        }
    
    def value_function(self, x: np.ndarray) -> np.ndarray:
        """Value function from Prospect Theory."""
        alpha = self.params["alpha"]
        beta = self.params["beta"]
        lambda_ = self.params["lambda"]
        
        return np.where(x >= 0, x**alpha, -lambda_ * (-x)**beta)
    
    def weight_function(self, p: np.ndarray) -> np.ndarray:
        """Probability weighting function from Prospect Theory."""
        gamma = self.params["gamma"]
        return p**gamma / (p**gamma + (1 - p)**gamma) ** (1/gamma)
    
    def predict(self, p1: np.ndarray, v1: np.ndarray, p2: np.ndarray, v2: np.ndarray) -> np.ndarray:
        """
        Make predictions using Prospect Theory model.
        
        Args:
            p1: Probabilities for option 1 [batch_size, n_outcomes]
            v1: Values for option 1 [batch_size, n_outcomes]
            p2: Probabilities for option 2 [batch_size, n_outcomes]
            v2: Values for option 2 [batch_size, n_outcomes]
            
        Returns:
            Array of predictions (0 or 1)
        """
        # Calculate weighted values
        u1 = self.value_function(v1)
        u2 = self.value_function(v2)
        
        w1 = self.weight_function(p1)
        w2 = self.weight_function(p2)
        
        # Calculate prospect values
        pv1 = np.sum(w1 * u1, axis=1)
        pv2 = np.sum(w2 * u2, axis=1)
        
        # Choose option with higher prospect value
        return (pv2 > pv1).astype(int)
    
    def _pt_likelihood(self, params, choices, p1, v1, p2, v2):
        """
        Calculate negative log likelihood for PT model (matching legacy implementation).
        """
        alpha_raw, beta_raw, gamma_raw, lambda_raw, tau_raw = params
        
        # Transform parameters using sigmoid function (matching legacy)
        alpha = expit(alpha_raw)  # Range: (0, 1)
        beta = expit(beta_raw)    # Range: (0, 1)
        gamma = expit(gamma_raw)  # Range: (0, 1)
        lambda_ = expit(lambda_raw) * 10  # Range: (0, 10)
        tau = expit(tau_raw) * 100  # Range: (0, 100)
        
        # Value function
        def value(x):
            return np.where(x >= 0, x**alpha, -lambda_ * (-x)**beta)
        
        # Probability weighting function
        def weight(p):
            return p**gamma / (p**gamma + (1-p)**gamma)**(1/gamma)
        
        total_log_likelihood = 0
        
        # Calculate log likelihood for each trial
        for trial_choices, trial_p1, trial_v1, trial_p2, trial_v2 in zip(choices, p1, v1, p2, v2):
            # Transform the list to numpy array and normalize (matching legacy preprocessing)
            trial_p1 = np.array(trial_p1) / 100
            trial_v1 = np.array(trial_v1) / 1000
            trial_p2 = np.array(trial_p2) / 100
            trial_v2 = np.array(trial_v2) / 1000
            
            weighted_v1 = np.sum(weight(trial_p1) * value(trial_v1))
            weighted_v2 = np.sum(weight(trial_p2) * value(trial_v2))
            
            weighted_v = np.array([weighted_v1, weighted_v2])
            p_choice = np.exp(tau * weighted_v) / np.sum(np.exp(tau * weighted_v))
            
            # Use bernoulli logpmf with p_choice[0] (probability of choosing option 1)
            trial_log_likelihood = np.sum(bernoulli.logpmf(trial_choices, p_choice[0]))
            total_log_likelihood += trial_log_likelihood
        
        # Return negative total log likelihood for minimization
        return -total_log_likelihood
    
    def _optimize_single(self, args):
        """Single optimization run (for multiprocessing)."""
        init_params, choices, p1, v1, p2, v2 = args
        bounds = [(-10, 10)] * 5  # Parameter bounds for all 5 raw parameters
        
        result = minimize(
            self._pt_likelihood,
            init_params,
            args=(choices, p1, v1, p2, v2),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000}
        )
        return result.fun, result.x
    
    def fit(self, p1: List, v1: List, p2: List, v2: List, choices: List,
            n_repeat: int = 100, num_cores: Optional[int] = None) -> Dict[str, float]:
        """
        Fit model parameters to data using the legacy fitting procedure.
        
        Args:
            p1: List of probability arrays for option 1
            v1: List of value arrays for option 1
            p2: List of probability arrays for option 2
            v2: List of value arrays for option 2
            choices: List of choices (0 or 1)
            n_repeat: Number of random initializations (default: 100)
            num_cores: Number of CPU cores to use (default: cpu_count - 1)
            
        Returns:
            Dictionary of fitted parameters and negative log likelihood
        """
        if num_cores is None:
            num_cores = mp.cpu_count() - 1
        
        # Generate random initializations (matching legacy: 20 * rand - 10)
        args_list = [(20 * np.random.rand(5) - 10, choices, p1, v1, p2, v2) for _ in range(n_repeat)]
        
        # Run optimization with multiprocessing
        with mp.Pool(num_cores) as pool:
            results = pool.map(self._optimize_single, args_list)
        
        # Get the minimum negative log likelihood (filtering out NaN values)
        best_ll, best_params = min(results, key=lambda x: x[0] if not np.isnan(x[0]) else float('inf'))
        
        # Transform best parameters back to original scales
        alpha = expit(best_params[0])
        beta = expit(best_params[1])
        gamma = expit(best_params[2])
        lambda_ = expit(best_params[3]) * 10
        tau = expit(best_params[4]) * 100
        
        # Update model parameters
        self.params = {
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
            "lambda": lambda_,
            "tau": tau
        }
        
        return {
            'best_params': {
                'alpha': alpha,
                'beta': beta,
                'gamma': gamma,
                'lambda': lambda_,
                'tau': tau
            },
            'neg_log_likelihood': best_ll
        } 