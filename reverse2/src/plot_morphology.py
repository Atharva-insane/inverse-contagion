import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

def plot_morphology(out_dir='../output'):
    print("Loading Morphology Results...")
    try:
        df = pd.read_csv(os.path.join(out_dir, 'morphology_results.csv'))
    except FileNotFoundError:
        print("Results not found. Run morphology.py first.")
        return
        
    print("Generating Systemic Risk Scatter Plot...")
    plt.figure(figsize=(12, 10))
    
    # X-axis: In-Strength (Vulnerability)
    # Y-axis: Out-Strength (Contagiousness)
    # Bubble Size: Eigenvector Centrality
    
    x = df['In_Strength_Vulnerability']
    y = df['Out_Strength_Contagiousness']
    sizes = df['Eigenvector_Centrality'] * 2000 # Scale for visibility
    labels = df['Airport']
    
    # Plot formatting
    scatter = plt.scatter(x, y, s=sizes, c=y-x, cmap='coolwarm', alpha=0.7, edgecolors='black', linewidth=1.5)
    
    # Add labels
    for i, label in enumerate(labels):
        if sizes[i] > 100 or y[i] > np.median(y): # Only label significant airports
            plt.text(x[i], y[i] + (sizes[i]/100000), label, fontsize=10, ha='center', va='bottom', fontweight='bold')
            
    # Add quadrants
    plt.axvline(x=np.median(x), color='gray', linestyle='--', alpha=0.5)
    plt.axhline(y=np.median(y), color='gray', linestyle='--', alpha=0.5)
    
    # Quadrant text
    plt.text(np.max(x)*0.8, np.max(y)*0.9, 'CONTAGION SOURCES\n(High Output, Low Input)', 
             fontsize=12, alpha=0.5, ha='center', va='center', color='red', weight='bold')
    plt.text(np.max(x)*0.8, np.min(y)*1.1, 'CONTAGION SINKS\n(Absorb Delays)', 
             fontsize=12, alpha=0.5, ha='center', va='center', color='blue', weight='bold')
             
    plt.title('Aviation Network Morphology:\nVulnerability vs. Contagiousness', fontsize=18, fontweight='bold')
    plt.xlabel('Vulnerability (Total Incoming Infectivity Weight)', fontsize=14)
    plt.ylabel('Contagiousness (Total Outgoing Infectivity Weight)', fontsize=14)
    
    # Colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Net Contagion (Out - In)', fontsize=12)
    
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    
    out_path = os.path.join(out_dir, 'morphology_scatter.png')
    plt.savefig(out_path, dpi=300)
    print(f"Scatter plot saved to: {out_path}")

if __name__ == "__main__":
    plot_morphology()
