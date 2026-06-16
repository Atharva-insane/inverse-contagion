import pandas as pd
import numpy as np
import os
import pickle

def load_and_preprocess_data(csv_path, top_k_airports=50, seq_len=24):
    """
    Loads flight data, creates airport network, and extracts delay cascades with covariates.
    """
    print(f"Loading data from {csv_path}...", flush=True)
    try:
        df = pd.read_csv(csv_path, usecols=['MONTH', 'DAY', 'DAY_OF_WEEK', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT', 
                                            'SCHEDULED_DEPARTURE', 'DEPARTURE_DELAY'])
    except ValueError as e:
        print(f"Warning: Column mismatch. Error: {e}")
        df = pd.read_csv(csv_path)

    df = df.dropna(subset=['ORIGIN_AIRPORT', 'DESTINATION_AIRPORT', 'DEPARTURE_DELAY'])
    df['DEPARTURE_DELAY'] = df['DEPARTURE_DELAY'].apply(lambda x: max(0, x))
    
    airport_counts = df['ORIGIN_AIRPORT'].value_counts()
    top_airports = airport_counts.head(top_k_airports).index.tolist()
    df = df[df['ORIGIN_AIRPORT'].isin(top_airports) & df['DESTINATION_AIRPORT'].isin(top_airports)]
    airport_to_idx = {apt: i for i, apt in enumerate(top_airports)}
    
    print("Constructing adjacency matrix...", flush=True)
    adj_matrix = np.zeros((top_k_airports, top_k_airports))
    flight_counts = df.groupby(['ORIGIN_AIRPORT', 'DESTINATION_AIRPORT']).size().reset_index(name='count')
    for _, row in flight_counts.iterrows():
        o_idx = airport_to_idx.get(row['ORIGIN_AIRPORT'])
        d_idx = airport_to_idx.get(row['DESTINATION_AIRPORT'])
        if o_idx is not None and d_idx is not None:
            adj_matrix[o_idx, d_idx] += row['count']
            
    # Binary adjacency for GAT masking
    adj_mask = (adj_matrix > 0).astype(float)
    np.fill_diagonal(adj_mask, 1.0) # Self-loops are essential for GAT
        
    print("Extracting temporal cascades...", flush=True)
    df['HOUR'] = (pd.to_numeric(df['SCHEDULED_DEPARTURE'], errors='coerce') // 100).fillna(0).astype(int)
    df.loc[df['HOUR'] > 23, 'HOUR'] = 23
    
    # We group to get average delay, but we also want the time features
    grouped = df.groupby(['MONTH', 'DAY', 'HOUR', 'ORIGIN_AIRPORT']).agg({
        'DEPARTURE_DELAY': 'mean',
        'DAY_OF_WEEK': 'first'
    }).reset_index()
    
    months = df['MONTH'].unique()
    days = df['DAY'].unique()
    hours = np.arange(24)
    
    index = pd.MultiIndex.from_product([months, days, hours], names=['MONTH', 'DAY', 'HOUR'])
    time_df = pd.DataFrame(index=index).reset_index()
    merged = pd.merge(time_df, grouped, on=['MONTH', 'DAY', 'HOUR'], how='left')
    
    # Pivot Delay
    pivot_delay = merged.pivot_table(index=['MONTH', 'DAY', 'HOUR'], columns='ORIGIN_AIRPORT', values='DEPARTURE_DELAY', fill_value=0)
    pivot_delay = pivot_delay.reindex(columns=top_airports, fill_value=0)
    
    # Pivot Day of Week
    pivot_dow = merged.pivot_table(index=['MONTH', 'DAY', 'HOUR'], columns='ORIGIN_AIRPORT', values='DAY_OF_WEEK', fill_value=0)
    pivot_dow = pivot_dow.reindex(columns=top_airports, fill_value=0)
    
    # We can encode Hour and DOW as cyclical features globally
    num_steps = len(pivot_delay)
    hours_array = np.array([idx[2] for idx in pivot_delay.index]) # Get HOUR from MultiIndex
    dows_array = np.array([idx[1] % 7 for idx in pivot_delay.index]) # Approximation if day_of_week pivot has zeroes
    
    # Fix DOW using actual dates or an approximation. For simplicity, since the data is sequential, 
    # we just use the first column of pivot_dow where it's non-zero, or interpolate.
    # To be extremely robust, we use cyclical time encoding for the Hour:
    hour_sin = np.sin(2 * np.pi * hours_array / 24.0)
    hour_cos = np.cos(2 * np.pi * hours_array / 24.0)
    
    # Build node features: [Delay, Hour_Sin, Hour_Cos]
    # Shape: (T, N, F)
    F = 3
    node_features = np.zeros((num_steps, top_k_airports, F))
    node_features[:, :, 0] = pivot_delay.values
    for i in range(top_k_airports):
        node_features[:, i, 1] = hour_sin
        node_features[:, i, 2] = hour_cos
    
    X, Y = [], []
    for i in range(len(node_features) - seq_len):
        X.append(node_features[i:i+seq_len])
        Y.append(node_features[i+seq_len, :, 0:1]) # Only predict the delay
        
    X = np.array(X)
    Y = np.array(Y)
    
    print(f"Data shape: X: {X.shape}, Y: {Y.shape}")
    
    return X, Y, adj_mask, top_airports

if __name__ == "__main__":
    data_path = "../flights.csv"
    if os.path.exists(data_path):
        X, Y, adj, airports = load_and_preprocess_data(data_path)
        os.makedirs('../processed_data', exist_ok=True)
        np.save('../processed_data/X.npy', X)
        np.save('../processed_data/Y.npy', Y)
        np.save('../processed_data/adj.npy', adj)
        with open('../processed_data/airports.pkl', 'wb') as f:
            pickle.dump(airports, f)
        print("Data preprocessed and saved to 'processed_data/'")
    else:
        print(f"Dataset {data_path} not found. Please place flights.csv in the root directory.")
