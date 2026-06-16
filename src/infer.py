import torch
import numpy as np
import pickle
import os
from model import NeuralGraphHawkesProcess

def reverse_engineer_cascade(model_path='../models/nghp_model.pth', data_dir='../processed_data'):
    print("Loading data and model for NGHP inference...")
    try:
        X = np.load(os.path.join(data_dir, 'X.npy'))
        adj = np.load(os.path.join(data_dir, 'adj.npy'))
        with open(os.path.join(data_dir, 'airports.pkl'), 'rb') as f:
            airports = pickle.load(f)
            
        # Extract held-out validation set chronologically
        dataset_size = len(X)
        train_size = int(0.8 * dataset_size)
        X_val = X[train_size:]
    except FileNotFoundError:
        print("Required data/files not found.")
        return

    num_nodes = X.shape[2]
    node_features = X.shape[3]
    seq_len = X.shape[1]
    
    model = NeuralGraphHawkesProcess(num_nodes=num_nodes, node_features=node_features, seq_len=seq_len)
    
    try:
        model.load_state_dict(torch.load(model_path))
        model.eval()
    except FileNotFoundError:
        print(f"Model {model_path} not found. Please train first.")
        return
        
    print("\n--- Reverse Engineering Results (Hawkes Process) ---")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    adj_tensor = torch.FloatTensor(adj).to(device)
    
    # Replay a real historical cascade from the held-out validation set
    # Find the snapshot with the most severe delay event to trace its propagation
    max_delay_idx = np.unravel_index(np.argmax(X_val[:, -1, :, 0]), X_val[:, -1, :, 0].shape)
    cascade_idx = max_delay_idx[0]
    target_idx = max_delay_idx[1]
    target_airport = airports[target_idx]
    
    seed_cascade = X_val[cascade_idx:cascade_idx+1] # (1, seq_len, N, F)
    seed_tensor = torch.FloatTensor(seed_cascade).to(device)
    
    print(f"\n--- Replaying True Historical Cascade (Held-out Validation) ---")
    print(f"Conditioning on observed network state. Primary disruption identified at: {target_airport}")
    
    # 1. Simulate the cascade intensity
    with torch.no_grad():
        total_intensity = model(seed_tensor, adj_tensor)
        mu, contagion, alpha = model.extract_hawkes_components()
        
    predicted_delays = total_intensity.cpu().numpy()[0, :, 0]
    mu_np = mu.cpu().numpy()[0, :, 0]
    contagion_np = contagion.cpu().numpy()[0, :, 0]
    
    affected_airports = np.argsort(predicted_delays)[::-1]
    
    print(f"\nSource of historical disruption: {target_airport}")
    print("Top 5 Airports infected (Total Expected Delay Events = Background + Contagion):")
    for idx in affected_airports[:5]:
        if idx != target_idx and predicted_delays[idx] > 0:
             print(f" - {airports[idx]}: Total: {predicted_delays[idx]:.2f} events (Background: {mu_np[idx]:.2f}, Contagion: {contagion_np[idx]:.2f})")

    # 2. Extract Hawkes Infectivity
    last_step_alpha = alpha[-1].cpu().numpy() # (N, N)
    
    print(f"\n--- Hawkes Infectivity Matrix Analysis ---")
    print(f"Which edges allowed the infection to spread from {target_airport}?")
    
    att_from_source = last_step_alpha[:, target_idx]
    att_sorted_indices = np.argsort(att_from_source)[::-1]
    
    for idx in att_sorted_indices[:5]:
        if idx != target_idx and att_from_source[idx] > 0:
            print(f" - Edge {target_airport} -> {airports[idx]} | Infectivity alpha: {att_from_source[idx]:.4f}")
            
    print("\nNotice how the model explicitly decouples the 'Background' delay from the 'Contagion' delay.")

    # 3. Calculate Multi-Exposure Score (MES)
    # MES measures how many different upstream airports pose a significant contagion risk to a specific airport.
    # We define "significant" as having an infectivity weight greater than the uniform average (1/N).
    print(f"\n--- Multi-Exposure Score (MES) Analysis ---")
    uniform_threshold = 1.0 / num_nodes
    
    # Calculate MES for all airports
    mes_scores = []
    for i in range(num_nodes):
        # How many incoming edges to airport i have an alpha > threshold? (Excluding self-loops)
        incoming_weights = last_step_alpha[i, :]
        # set self-loop to 0 for this calculation
        incoming_weights_no_self = np.copy(incoming_weights)
        incoming_weights_no_self[i] = 0.0
        
        score = np.sum(incoming_weights_no_self > uniform_threshold)
        mes_scores.append(score)
        
    mes_scores = np.array(mes_scores)
    most_exposed_indices = np.argsort(mes_scores)[::-1]
    
    print("Top 5 Most Vulnerable Airports (Highest Multi-Exposure to network cascading):")
    for idx in most_exposed_indices[:5]:
        print(f" - {airports[idx]}: Multi-Exposure Score = {mes_scores[idx]} (Highly sensitive to {mes_scores[idx]} distinct incoming flight paths)")

if __name__ == "__main__":
    reverse_engineer_cascade()
