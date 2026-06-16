import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from model import NeuralGraphHawkesProcess
import pickle

def plot_branching(model_path='../models/nghp_model.pth', data_dir='../processed_data'):
    print("Loading data to calculate Branching Ratios...")
    try:
        X = np.load(os.path.join(data_dir, 'X.npy'))
        adj = np.load(os.path.join(data_dir, 'adj.npy'))
        with open(os.path.join(data_dir, 'airports.pkl'), 'rb') as f:
            airports = pickle.load(f)
            
        # Get held-out validation set
        dataset_size = len(X)
        train_size = int(0.8 * dataset_size)
        X_val = X[train_size:]
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

    # Calculate the AVERAGE alpha matrix across a large sample of the validation set
    # to find the true underlying structural branching ratios (rather than one specific event)
    sample_size = min(500, len(X_val))
    indices = np.random.choice(len(X_val), sample_size, replace=False)
    sample_X = torch.FloatTensor(X_val[indices]).to(device)
    adj_tensor = torch.FloatTensor(adj).to(device)
    
    with torch.no_grad():
        _ = model(sample_X, adj_tensor)
        _, _, alpha = model.extract_hawkes_components()
        alpha_mat = alpha.mean(dim=0).numpy() # (N, N) Average over the batch
        beta = torch.nn.functional.softplus(model.beta).numpy() # (N,)

    # Calculate Branching Ratio Matrix (Gamma)
    sum_decay = np.exp(-beta) / (1 - np.exp(-beta))
    Gamma = alpha_mat * sum_decay[:, np.newaxis]
    
    # Calculate the Total Outgoing Branching Ratio (Sum over all targets)
    out_branching = np.sum(Gamma, axis=0)
    
    # Sort and plot EVERY airport
    top_n = len(airports)
    top_indices = np.argsort(out_branching)[::-1]
    
    top_airports_labels = [airports[i] for i in top_indices]
    top_branching_values = out_branching[top_indices]
    
    # Create a sleek horizontal bar chart (Taller to fit all 50 airports)
    plt.figure(figsize=(12, 16))
    
    # Use a colormap to emphasize severity
    colors = plt.cm.Reds(np.linspace(0.8, 0.3, top_n))
    
    bars = plt.barh(np.arange(top_n), top_branching_values[::-1], color=colors[::-1], edgecolor='black')
    
    plt.yticks(np.arange(top_n), top_airports_labels[::-1], fontsize=10, fontweight='bold')
    plt.xlabel('Total Expected Offspring (Cascading Delays Triggered per 1 Initial Delay)', fontsize=14)
    plt.title('Aviation Systemic Risk Index\n(Network Branching Ratio by Source Airport)', fontsize=18, fontweight='bold')
    
    # Add data labels
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.05, bar.get_y() + bar.get_height()/2, 
                 f'{width:.2f}', va='center', ha='left', fontsize=9)
                 
    # Adjust x-axis limit to fit text
    max_val = np.max(top_branching_values)
    plt.xlim(0, max_val * 1.15)
    
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    os.makedirs('../output', exist_ok=True)
    out_path = '../output/branching_ratio_barchart.png'
    plt.savefig(out_path, dpi=300)
    print(f"Branching Ratio Heatmap saved to {out_path}")

if __name__ == "__main__":
    plot_branching()
