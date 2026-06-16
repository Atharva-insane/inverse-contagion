import numpy as np
import pandas as pd
import os
import pickle
import networkx as nx

def run_morphological_analysis(model_dir='../models', data_dir='../processed_data', out_dir='../output'):
    print("Loading EM Estimated Parameters and Metadata...")
    try:
        alpha = np.load(os.path.join(model_dir, 'alpha.npy'))
        with open(os.path.join(data_dir, 'airports.pkl'), 'rb') as f:
            airports = pickle.load(f)
    except FileNotFoundError:
        print("Model parameters not found. Please run train.py first.")
        return
        
    num_nodes = len(airports)
    
    print("Computing Morphological Metrics...")
    
    # 1. Asymmetric Strengths
    # In-Strength (Vulnerability): Sum of alpha coming INTO node j from all i
    # alpha[i, j] is effect of i on j. So sum over axis 0.
    in_strength = np.sum(alpha, axis=0)
    
    # Out-Strength (Contagiousness): Sum of alpha going OUT of node i to all j
    out_strength = np.sum(alpha, axis=1)
    
    # 2. Eigenvector Centrality
    # Create directed graph
    G = nx.DiGraph()
    for i in range(num_nodes):
        G.add_node(airports[i])
        for j in range(num_nodes):
            if alpha[i, j] > 1e-5: # threshold to remove numerical zeros
                G.add_edge(airports[i], airports[j], weight=alpha[i, j])
                
    try:
        eigen_centrality = nx.eigenvector_centrality_numpy(G, weight='weight')
    except Exception as e:
        print(f"Eigenvector centrality failed to converge: {e}")
        eigen_centrality = {apt: 0.0 for apt in airports}
        
    # Compile Results
    results = []
    for i, apt in enumerate(airports):
        results.append({
            'Airport': apt,
            'In_Strength_Vulnerability': in_strength[i],
            'Out_Strength_Contagiousness': out_strength[i],
            'Eigenvector_Centrality': eigen_centrality[apt],
            'Net_Contagion': out_strength[i] - in_strength[i] # >0 means source, <0 means sink
        })
        
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values(by='Eigenvector_Centrality', ascending=False)
    
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'morphology_results.csv')
    df_results.to_csv(out_path, index=False)
    
    print(f"Morphological Analysis Complete! Saved to {out_path}")
    print("\nTop 5 Most Central Hubs:")
    print(df_results[['Airport', 'Eigenvector_Centrality', 'Out_Strength_Contagiousness']].head(5).to_string(index=False))

if __name__ == "__main__":
    run_morphological_analysis()
