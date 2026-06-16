import torch
import numpy as np
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
from model import NeuralGraphHawkesProcess

def visualize_contagion(model_path='../models/nghp_model.pth', data_dir='../processed_data'):
    print("Loading data for geographical visualization...")
    
    # Load airports metadata to get Lat/Lon
    try:
        airports_df = pd.read_csv('../airports.csv')
    except FileNotFoundError:
        print("airports.csv not found in the root directory.")
        return

    try:
        X = np.load(os.path.join(data_dir, 'X.npy'))
        adj = np.load(os.path.join(data_dir, 'adj.npy'))
        with open(os.path.join(data_dir, 'airports.pkl'), 'rb') as f:
            top_airports = pickle.load(f)
    except FileNotFoundError:
        print("Processed data not found. Run data_prep.py first.")
        return

    num_nodes = X.shape[2]
    node_features = X.shape[3]
    seq_len = X.shape[1]
    
    # Filter airport metadata to only our top airports, preserving the exact order
    airport_meta = airports_df[airports_df['IATA_CODE'].isin(top_airports)].set_index('IATA_CODE')
    airport_meta = airport_meta.reindex(top_airports)

    # Load Model
    model = NeuralGraphHawkesProcess(num_nodes=num_nodes, node_features=node_features, seq_len=seq_len)
    try:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
    except FileNotFoundError:
        print(f"Model {model_path} not found. Please train first.")
        return
        
    device = torch.device("cpu")
    adj_tensor = torch.FloatTensor(adj).to(device)
    
    # Find index for JFK (New York)
    target_airport = 'JFK'
    if target_airport in top_airports:
        target_idx = top_airports.index(target_airport)
    else:
        target_idx = 0 # Fallback
        target_airport = top_airports[0]
        
    # Create an artificial severe delay at JFK
    seed_cascade = np.zeros((1, seq_len, num_nodes, node_features))
    seed_cascade[0, -1, target_idx, 0] = 50.0 # 50 delay events at the source
    seed_tensor = torch.FloatTensor(seed_cascade).to(device)
    
    with torch.no_grad():
        total_intensity = model(seed_tensor, adj_tensor)
        mu, contagion, alpha = model.extract_hawkes_components()
        
    last_step_alpha = alpha[-1].numpy() # (N, N)
    att_from_source = last_step_alpha[:, target_idx] # How much target airport infected others
    
    # --- Plotting ---
    plt.figure(figsize=(12, 8))
    
    # Simple bounds for Continental US
    plt.xlim(-125, -65)
    plt.ylim(25, 50)
    plt.title(f"Hawkes Process Contagion Spread from {target_airport}", fontsize=16)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")

    # Plot all top airports
    plt.scatter(airport_meta['LONGITUDE'], airport_meta['LATITUDE'], c='blue', s=20, alpha=0.5, label='Airports')

    source_lon = airport_meta.loc[target_airport, 'LONGITUDE']
    source_lat = airport_meta.loc[target_airport, 'LATITUDE']
    
    # Plot the source of disruption
    plt.scatter(source_lon, source_lat, c='red', s=200, marker='*', label=f'Source ({target_airport})')

    # Plot the infection lines (edges) with high alpha weights
    for idx in range(1, num_nodes):
        weight = att_from_source[idx]
        if weight > 0.01: # Threshold to only show meaningful contagion paths
            target_lon = airport_meta.iloc[idx]['LONGITUDE']
            target_lat = airport_meta.iloc[idx]['LATITUDE']
            
            # Line thickness based on infectivity weight
            plt.plot([source_lon, target_lon], [source_lat, target_lat], 
                     c='red', alpha=weight, linewidth=weight * 50)
            
            # Label the highly infected airports
            plt.text(target_lon, target_lat, top_airports[idx], fontsize=9)

    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Save the figure
    os.makedirs('../output', exist_ok=True)
    out_path = f'../output/hawkes_contagion_map_{target_airport}.png'
    plt.savefig(out_path, dpi=300)
    print(f"\nVisualization saved successfully to {out_path}")
    print("This map visually proves how the Hawkes Process reverse-engineered the network spread!")

if __name__ == "__main__":
    visualize_contagion()
