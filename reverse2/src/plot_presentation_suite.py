import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import os
import pickle

def generate_visual_suite(model_dir='../models', data_dir='../processed_data', out_dir='../output'):
    print("Loading Data for Visual Suite...")
    try:
        alpha = np.load(os.path.join(model_dir, 'alpha.npy'))
        mu = np.load(os.path.join(model_dir, 'mu.npy'))
        df = pd.read_csv(os.path.join(data_dir, 'events.csv'))
        with open(os.path.join(data_dir, 'airports.pkl'), 'rb') as f:
            airports = pickle.load(f)
        morph_df = pd.read_csv(os.path.join(out_dir, 'morphology_results.csv'))
    except FileNotFoundError:
        print("Required files not found.")
        return

    os.makedirs(out_dir, exist_ok=True)
    num_nodes = len(airports)
    
    # 1. Network Topology Graph
    print("Generating Contagion Topology Graph...")
    plt.figure(figsize=(14, 12))
    G = nx.DiGraph()
    for i in range(num_nodes):
        G.add_node(airports[i])
        
    # Add top 50 edges by alpha
    flat_indices = np.argsort(alpha.flatten())[::-1][:75] # Top 75 edges
    for idx in flat_indices:
        i, j = np.unravel_index(idx, alpha.shape)
        if alpha[i, j] > 0.001:
            G.add_edge(airports[i], airports[j], weight=alpha[i, j]*50)
            
    # Node sizes from Eigenvector Centrality
    centrality_map = dict(zip(morph_df['Airport'], morph_df['Eigenvector_Centrality']))
    node_sizes = [centrality_map.get(node, 0) * 10000 + 500 for node in G.nodes()]
    
    pos = nx.spring_layout(G, seed=42, k=0.5)
    edges = G.edges()
    weights = [G[u][v]['weight'] for u,v in edges]
    
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='lightcoral', alpha=0.8, edgecolors='black')
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
    nx.draw_networkx_edges(G, pos, edgelist=edges, width=weights, edge_color='gray', 
                           arrowsize=20, alpha=0.6, connectionstyle='arc3,rad=0.1')
                           
    plt.title('Top 75 Structural Contagion Pathways', fontsize=20, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'network_topology.png'), dpi=300)
    plt.close()
    
    # 2. Adjacency-Constrained Alpha Heatmap
    print("Generating Alpha Heatmap...")
    plt.figure(figsize=(10, 8))
    im = plt.imshow(alpha, cmap='Reds', aspect='auto')
    plt.colorbar(im, label='Infectivity Weight ($\u03B1_{ij}$)')
    
    # Show only top 20 airports on axes for clarity
    top_20 = airports[:20]
    plt.xticks(np.arange(20), top_20, rotation=90)
    plt.yticks(np.arange(20), top_20)
    plt.xlim(-0.5, 19.5)
    plt.ylim(19.5, -0.5)
    
    plt.title(r'Strictly Constrained Infectivity Matrix ($\alpha_{ij}$)', fontsize=16, fontweight='bold')
    plt.xlabel('Target Airport (Infected)')
    plt.ylabel('Source Airport (Delayed)')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'alpha_heatmap.png'), dpi=300)
    plt.close()
    
    # 3. Hawkes Intensity Decomposition (Time Series)
    print("Generating Hawkes Intensity Decomposition...")
    # Target ORD (index of ORD)
    if 'ORD' in airports:
        target_idx = airports.index('ORD')
        # Filter first 5000 events
        df_sub = df.head(5000)
        start_t = df_sub['TIME_MINUTES'].min()
        
        # We will plot 12 hours (720 mins) starting from minute 2000
        plot_start = start_t + 2000
        plot_end = plot_start + 720
        
        time_grid = np.linspace(plot_start, plot_end, 500)
        mu_val = mu[target_idx]
        
        # Calculate contagion intensity
        contagion_intensity = np.zeros_like(time_grid)
        beta = 0.01
        
        past_events = df_sub[df_sub['TIME_MINUTES'] < plot_end]
        for _, row in past_events.iterrows():
            t_k = row['TIME_MINUTES']
            u_k = int(row['NODE'])
            a = alpha[u_k, target_idx]
            if a > 0:
                # Add to grid points where t > t_k
                mask = time_grid > t_k
                contagion_intensity[mask] += a * beta * np.exp(-beta * (time_grid[mask] - t_k))
                
        plt.figure(figsize=(12, 6))
        plt.plot(time_grid, np.full_like(time_grid, mu_val), label=r'Background Rate ($\mu$)', color='blue', linestyle='--')
        plt.plot(time_grid, mu_val + contagion_intensity, label=r'Total Intensity ($\lambda(t)$)', color='red')
        plt.fill_between(time_grid, mu_val, mu_val + contagion_intensity, color='red', alpha=0.3, label='Network Contagion Cascades')
        
        plt.title('ORD Intensity Decomposition (Continuous Time)', fontsize=16, fontweight='bold')
        plt.xlabel('Time (Minutes)', fontsize=14)
        plt.ylabel(r'Expected Events per Minute ($\lambda(t)$)', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'hawkes_decomposition_ORD.png'), dpi=300)
        plt.close()

    # 4. MAE Model Comparison Bar Chart
    print("Generating MAE Comparison Chart...")
    plt.figure(figsize=(8, 6))
    labels = ['Naive Baseline', 'Statistical EM Hawkes']
    values = [1.6376, 1.3909] # Hardcoded from our previous run
    colors = ['gray', 'crimson']
    
    bars = plt.bar(labels, values, color=colors, edgecolor='black', width=0.6)
    plt.ylabel('Mean Absolute Error (Events/Hour)', fontsize=14)
    plt.title('Predictive Superiority on Held-Out Data', fontsize=16, fontweight='bold')
    plt.ylim(0, 2.0)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.05, f"{yval:.4f}", ha='center', va='bottom', fontsize=14, fontweight='bold')
        
    # Add an arrow showing improvement
    plt.annotate('-15.06%', xy=(1, 1.45), xytext=(0, 1.45),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=2, headwidth=8),
                 fontsize=14, fontweight='bold', ha='center', va='center')
                 
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'mae_comparison.png'), dpi=300)
    plt.close()
    
    print("All visuals generated successfully!")

if __name__ == "__main__":
    generate_visual_suite()
