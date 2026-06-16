import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import os
import pickle

def plot_qq_random_time_change(model_dir='../models', data_dir='../processed_data', out_dir='../output', target_airport='ORD'):
    print(f"Generating Random Time Change Q-Q Plot for {target_airport}...")
    try:
        alpha = np.load(os.path.join(model_dir, 'alpha.npy'))
        mu = np.load(os.path.join(model_dir, 'mu.npy'))
        df = pd.read_csv(os.path.join(data_dir, 'events.csv'))
        with open(os.path.join(data_dir, 'airports.pkl'), 'rb') as f:
            airports = pickle.load(f)
    except FileNotFoundError:
        print("Required files not found.")
        return

    os.makedirs(out_dir, exist_ok=True)
    
    if target_airport not in airports:
        print(f"{target_airport} not in Top 50 airports.")
        return
        
    target_idx = airports.index(target_airport)
    beta = 0.01 # From train.py
    
    # We use a limited subset (e.g., first 5000 events) for computational speed
    df_sub = df.head(5000).copy()
    
    # Extract only events that happened at the target airport
    target_events = df_sub[df_sub['NODE'] == target_idx]['TIME_MINUTES'].values
    all_times = df_sub['TIME_MINUTES'].values
    all_nodes = df_sub['NODE'].values.astype(int)
    
    if len(target_events) < 10:
        print(f"Not enough events for {target_airport} in this subset to generate a reliable Q-Q plot.")
        return
        
    # Calculate the Integrated Intensity (Compensator) Lambda(t) at each target event time
    print(f"Calculating Compensator Integrals for {len(target_events)} events...")
    
    Lambda_t = np.zeros(len(target_events))
    
    for k, t_k in enumerate(target_events):
        # Base background integral
        L = mu[target_idx] * t_k
        
        # Self-Exciting integral from all past events in the network
        # Find all events before t_k
        idx_end = np.searchsorted(all_times, t_k)
        if idx_end > 0:
            past_t = all_times[:idx_end]
            past_u = all_nodes[:idx_end]
            
            # Sum over all i: alpha_{u_i, target} * (1 - e^{-beta * (t_k - t_i)})
            # Note: We only add contagion if alpha > 0 to save computation
            active_mask = alpha[past_u, target_idx] > 0
            if np.any(active_mask):
                valid_past_t = past_t[active_mask]
                valid_past_u = past_u[active_mask]
                
                L += np.sum(alpha[valid_past_u, target_idx] * (1.0 - np.exp(-beta * (t_k - valid_past_t))))
                
        Lambda_t[k] = L
        
    # The Random Time Change Theorem:
    # tau_k = Lambda(t_k) - Lambda(t_{k-1})
    tau = np.diff(Lambda_t)
    
    # Remove any numerical zeros or negative diffs just in case
    tau = tau[tau > 1e-5]
    
    if len(tau) == 0:
        print("Failed to calculate meaningful tau intervals.")
        return
        
    # Sort empirical quantiles
    tau_sorted = np.sort(tau)
    N = len(tau_sorted)
    
    # Theoretical Exp(1) quantiles
    p = (np.arange(1, N + 1) - 0.5) / N
    theoretical_quantiles = -np.log(1 - p)
    
    # Plotting
    plt.figure(figsize=(8, 8))
    plt.scatter(theoretical_quantiles, tau_sorted, color='blue', alpha=0.6, edgecolors='black')
    
    # Y = X reference line
    max_val = max(np.max(theoretical_quantiles), np.max(tau_sorted))
    plt.plot([0, max_val], [0, max_val], color='red', linestyle='--', linewidth=2, label='Perfect Fit ($y=x$)')
    
    plt.title(f'Continuous-Time Q-Q Plot ({target_airport})\nRandom Time Change Theorem', fontsize=16, fontweight='bold')
    plt.xlabel('Theoretical Quantiles (Exponential Distribution)', fontsize=14)
    plt.ylabel('Empirical Quantiles (Transformed Inter-Event Times)', fontsize=14)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    out_path = os.path.join(out_dir, f'qq_plot_{target_airport}.png')
    plt.savefig(out_path, dpi=300)
    print(f"Q-Q Plot successfully generated and saved to {out_path}")

if __name__ == "__main__":
    plot_qq_random_time_change()
