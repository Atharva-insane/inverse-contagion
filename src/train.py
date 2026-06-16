import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from model import NeuralGraphHawkesProcess
from torch.utils.data import TensorDataset, DataLoader

def train_model(data_dir='../processed_data', epochs=20, batch_size=16, lr=0.001):
    print("Loading processed data for Neural Graph Hawkes Process...")
    try:
        X = np.load(os.path.join(data_dir, 'X.npy'))
        Y = np.load(os.path.join(data_dir, 'Y.npy'))
        adj = np.load(os.path.join(data_dir, 'adj.npy'))
    except FileNotFoundError:
        print("Data not found. Please run data_prep.py first.")
        return
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    X_tensor = torch.FloatTensor(X).to(device)
    Y_tensor = torch.FloatTensor(Y).to(device)
    adj_tensor = torch.FloatTensor(adj).to(device)
    
    dataset = TensorDataset(X_tensor, Y_tensor)
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    num_nodes = X.shape[2]
    node_features = X.shape[3]
    seq_len = X.shape[1]
    
    model = NeuralGraphHawkesProcess(num_nodes=num_nodes, node_features=node_features, seq_len=seq_len).to(device)
    
    # Optimize the Point-Process Negative Log-Likelihood (NLL) for discrete event counts
    criterion = nn.PoissonNLLLoss(log_input=False, full=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    print("Starting training of NGHP...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_X, batch_Y in train_loader:
            optimizer.zero_grad()
            intensity = model(batch_X, adj_tensor)
            loss = criterion(intensity, batch_Y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_Y in val_loader:
                intensity = model(batch_X, adj_tensor)
                loss = criterion(intensity, batch_Y)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss (Poisson NLL): {train_loss:.4f} | Val NLL Loss: {val_loss:.4f}")
        
    os.makedirs('../models', exist_ok=True)
    torch.save(model.state_dict(), '../models/nghp_model.pth')
    print("Model saved to ../models/nghp_model.pth")

if __name__ == "__main__":
    train_model(epochs=2)
