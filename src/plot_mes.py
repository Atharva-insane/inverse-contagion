import torch
import numpy as np
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
from model import NeuralGraphHawkesProcess

def plot_mes(model_path='../models/nghp_model.pth', data_dir='../processed_data'):
    print("Loading data for MES Visualization...")
    try:
        X = np.load(os.path.join(data_dir, 'X.npy'))
        adj = np.load(os.path.join(data_dir, 'adj.npy'))
        with open(os.path.join(data_dir, 'airports.pkl'), 'rb') as f:
            airports = pickle.load(f)
    except FileNotFoundError:
        print("Processed data not found.")
        return

    num_nodes = X.shape[2]
    node_features = X.shape[3]
    seq_len = X.shape[1]
    
    model = NeuralGraphHawkesProcess(num_nodes=num_nodes, node_features=node_features, seq_len=seq_len)
    try:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
    except FileNotFoundError:
        print("Model not found.")
        return
        
    device = torch.device("cpu")
    adj_tensor = torch.FloatTensor(adj).to(device)
    
    # Create a dummy tensor to extract attention
    seed_cascade = np.zeros((1, seq_len, num_nodes, node_features))
    seed_tensor = torch.FloatTensor(seed_cascade).to(device)
    
    with torch.no_grad():
        _ = model(seed_tensor, adj_tensor)
        _, _, alpha = model.extract_hawkes_components()
        
    last_step_alpha = alpha[-1].numpy() # (N, N)
    
    # Calculate MES
    uniform_threshold = 1.0 / num_nodes
    mes_scores = []
    for i in range(num_nodes):
        incoming_weights = last_step_alpha[i, :]
        incoming_weights_no_self = np.copy(incoming_weights)
        incoming_weights_no_self[i] = 0.0
        score = np.sum(incoming_weights_no_self > uniform_threshold)
        mes_scores.append(score)
        
    mes_scores = np.array(mes_scores)
    
    # Sort and get top 15
    sorted_indices = np.argsort(mes_scores)[::-1][:15]
    top_airports = [airports[i] for i in sorted_indices]
    top_scores = mes_scores[sorted_indices]
    
    # Plotting
    plt.figure(figsize=(10, 6))
    
    # Create a gradient color map for the bars
    colors = plt.cm.Reds(np.linspace(0.8, 0.4, len(top_scores)))
    
    bars = plt.bar(top_airports, top_scores, color=colors, edgecolor='black')
    
    plt.title('Top 15 Most Vulnerable US Airports (Multi-Exposure Score)', fontsize=14, pad=15)
    plt.xlabel('Airport IATA Code', fontsize=12)
    plt.ylabel('MES (Number of Significant Exposing Edges)', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                 f'{int(height)}',
                 ha='center', va='bottom')
                 
    plt.tight_layout()
    
    os.makedirs('../output', exist_ok=True)
    out_path = '../output/mes_distribution.png'
    plt.savefig(out_path, dpi=300)
    print(f"\nMES Bar Chart saved successfully to {out_path}")

if __name__ == "__main__":
    plot_mes()
