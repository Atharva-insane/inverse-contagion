import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from model import NeuralGraphHawkesProcess
import pickle

def plot_insights(model_path='../models/nghp_model.pth', data_dir='../processed_data'):
    print("Loading data for deeper visual insights...")
    try:
        X = np.load(os.path.join(data_dir, 'X.npy'))
        adj = np.load(os.path.join(data_dir, 'adj.npy'))
        with open(os.path.join(data_dir, 'airports.pkl'), 'rb') as f:
            airports = pickle.load(f)
    except FileNotFoundError:
        print("Processed data not found.")
        return

    device = torch.device("cpu")
    num_nodes = X.shape[2]
    node_features = X.shape[3]
    seq_len = X.shape[1]
    
    model = NeuralGraphHawkesProcess(num_nodes=num_nodes, node_features=node_features, seq_len=seq_len)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
    except FileNotFoundError:
        print("Model not found.")
        return

    os.makedirs('../output', exist_ok=True)
    
    # --- INSIGHT 1: The Learned Exponential Decay Kernel ---
    beta = torch.nn.functional.softplus(model.beta).item()
    print(f"Learned Decay Parameter (Beta): {beta:.4f}")
    
    hours = np.linspace(0, 12, 100) # Plot over 12 hours
    decay_curve = np.exp(-beta * hours)
    
    plt.figure(figsize=(8, 5))
    plt.plot(hours, decay_curve, color='purple', linewidth=3, label=r'$\kappa(t) = e^{-\beta t}$')
    plt.fill_between(hours, decay_curve, color='purple', alpha=0.2)
    plt.title("The Learned Temporal Decay Kernel", fontsize=14)
    plt.xlabel("Hours Since Initial Delay", fontsize=12)
    plt.ylabel("Contagion Remaining (%)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12)
    plt.tight_layout()
    kernel_path = '../output/learned_kernel.png'
    plt.savefig(kernel_path, dpi=300)
    print(f"Kernel visualization saved to {kernel_path}")
    
    # --- INSIGHT 2: Stacked Area Chart (Background vs Contagion) ---
    # We will simulate a rolling cascade over 24 hours to show how background and contagion stack.
    # Take a real sequence from the dataset
    sample_idx = 100 # Arbitrary point in time
    sample_X = torch.FloatTensor(X[sample_idx:sample_idx+24]).to(device) # Shape: (24, SeqLen, N, F)
    adj_tensor = torch.FloatTensor(adj).to(device)
    
    target_apt = 'ATL'
    target_idx = airports.index(target_apt) if target_apt in airports else 0
    
    mu_list = []
    contagion_list = []
    
    with torch.no_grad():
        for t in range(24):
            x_t = sample_X[t].unsqueeze(0) # (1, SeqLen, N, F)
            _ = model(x_t, adj_tensor)
            mu, contagion, _ = model.extract_hawkes_components()
            
            mu_val = mu[0, target_idx, 0].item()
            contagion_val = contagion[0, target_idx, 0].item()
            
            mu_list.append(mu_val)
            contagion_list.append(contagion_val)
            
    mu_arr = np.array(mu_list)
    contagion_arr = np.array(contagion_list)
    total_arr = mu_arr + contagion_arr
    
    time_steps = np.arange(24)
    
    plt.figure(figsize=(10, 6))
    plt.stackplot(time_steps, mu_arr, contagion_arr, labels=[r'Background Delay ($\mu$)', r'Network Contagion ($\alpha \star y$)'],
                  colors=['#3498db', '#e74c3c'], alpha=0.8)
    
    plt.plot(time_steps, total_arr, color='black', linewidth=2, label=r'Total Intensity ($\lambda$)')
    
    plt.title(f"Hawkes Decomposition: {airports[target_idx]} over 24 Hours", fontsize=16)
    plt.xlabel("Hour of Day (Simulation)", fontsize=12)
    plt.ylabel("Expected Delay (Minutes)", fontsize=12)
    plt.legend(loc='upper left', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    decomp_path = f'../output/hawkes_decomposition_{airports[target_idx]}.png'
    plt.savefig(decomp_path, dpi=300)
    print(f"Decomposition visualization saved to {decomp_path}")

if __name__ == "__main__":
    plot_insights()
