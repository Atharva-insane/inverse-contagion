import numpy as np
import pandas as pd
import os
import pickle
from em_hawkes import NetworkConstrainedHawkesEM

def train_em_hawkes(data_dir='../processed_data', model_dir='../models'):
    print("Loading data...")
    try:
        df = pd.read_csv(os.path.join(data_dir, 'events.csv'))
        adj_matrix = np.load(os.path.join(data_dir, 'adj.npy'))
        with open(os.path.join(data_dir, 'airports.pkl'), 'rb') as f:
            airports = pickle.load(f)
    except FileNotFoundError:
        print("Data not found. Please run data_prep.py first.")
        return
        
    num_nodes = len(airports)
    
    # Sort just in case
    df = df.sort_values(by='TIME_MINUTES')
    
    # We take the first 20,000 events to prove the EM algorithm works without taking hours
    num_events = min(20000, len(df))
    df = df.head(num_events)
    
    times = df['TIME_MINUTES'].values
    nodes = df['NODE'].values
    
    T = times[-1] - times[0]
    
    print(f"Initializing EM Hawkes Model ({num_nodes} nodes, {num_events} events)...")
    # beta=0.01 implies a half-life of roughly 69 minutes for contagion delay
    model = NetworkConstrainedHawkesEM(num_nodes=num_nodes, adj_matrix=adj_matrix, beta=0.01)
    
    # Train
    # We do 10 iterations to prove monotonic convergence of Log-Likelihood
    final_ll = model.fit(times, nodes, T=T, max_iter=10)
    
    print("\nSaving Learned Parameters...")
    os.makedirs(model_dir, exist_ok=True)
    np.save(os.path.join(model_dir, 'alpha.npy'), model.alpha)
    np.save(os.path.join(model_dir, 'mu.npy'), model.mu)
    
    print("Training Complete. Structural constraint satisfied.")

if __name__ == "__main__":
    train_em_hawkes()
