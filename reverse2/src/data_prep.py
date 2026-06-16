import pandas as pd
import numpy as np
import os
import pickle

def prepare_continuous_data(csv_path='../../flights.csv', data_dir='../processed_data', top_k_airports=50):
    print(f"Loading data from {csv_path}...")
    
    usecols = ['YEAR', 'MONTH', 'DAY', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT', 
               'SCHEDULED_DEPARTURE', 'DEPARTURE_DELAY']
    
    try:
        df = pd.read_csv(csv_path, usecols=usecols)
    except FileNotFoundError:
        print(f"Error: Could not find {csv_path}")
        return
        
    print("Filtering missing data...")
    df = df.dropna(subset=['ORIGIN_AIRPORT', 'DESTINATION_AIRPORT', 'DEPARTURE_DELAY', 'SCHEDULED_DEPARTURE'])
    
    # 1. Identify Top K Airports
    airport_counts = df['ORIGIN_AIRPORT'].value_counts()
    top_airports = airport_counts.head(top_k_airports).index.tolist()
    airport_to_idx = {apt: i for i, apt in enumerate(top_airports)}
    
    # Save airport mapping
    os.makedirs(data_dir, exist_ok=True)
    with open(os.path.join(data_dir, 'airports.pkl'), 'wb') as f:
        pickle.dump(top_airports, f)
        
    print(f"Filtering dataset to the Top {top_k_airports} Mega-Hub airports...")
    df = df[df['ORIGIN_AIRPORT'].isin(top_airports) & df['DESTINATION_AIRPORT'].isin(top_airports)].copy()
    
    # 2. Build Structural Adjacency Matrix (Network Constraint)
    print("Constructing physical Adjacency Matrix...")
    adj_matrix = np.zeros((top_k_airports, top_k_airports))
    flight_counts = df.groupby(['ORIGIN_AIRPORT', 'DESTINATION_AIRPORT']).size().reset_index(name='count')
    for _, row in flight_counts.iterrows():
        o_idx = airport_to_idx.get(row['ORIGIN_AIRPORT'])
        d_idx = airport_to_idx.get(row['DESTINATION_AIRPORT'])
        if o_idx is not None and d_idx is not None:
            adj_matrix[o_idx, d_idx] = 1.0
            
    # Save the Adjacency Matrix for the EM constraint
    np.save(os.path.join(data_dir, 'adj.npy'), adj_matrix)
    
    # 3. Extract Continuous Time Marked Point Process Events
    print("Extracting Severe Contagion Events (>15 min delay)...")
    df = df[df['DEPARTURE_DELAY'] > 15].copy()
    
    print("Constructing Real-Time Continuous Timestamps...")
    df['SCHEDULED_DEPARTURE'] = df['SCHEDULED_DEPARTURE'].astype(int)
    df['HOUR'] = df['SCHEDULED_DEPARTURE'] // 100
    df['MINUTE'] = df['SCHEDULED_DEPARTURE'] % 100
    df['HOUR'] = df['HOUR'].apply(lambda x: 0 if x >= 24 else x)
    
    df['SCHEDULED_DATETIME'] = pd.to_datetime({
        'year': df['YEAR'],
        'month': df['MONTH'],
        'day': df['DAY'],
        'hour': df['HOUR'],
        'minute': df['MINUTE']
    }, errors='coerce')
    
    df = df.dropna(subset=['SCHEDULED_DATETIME'])
    df['ACTUAL_DEPARTURE_TIMESTAMP'] = df['SCHEDULED_DATETIME'] + pd.to_timedelta(df['DEPARTURE_DELAY'], unit='m')
    
    # Sort chronologically
    df = df.sort_values(by='ACTUAL_DEPARTURE_TIMESTAMP')
    
    # Map node strings to integers for fast EM access
    df['NODE'] = df['ORIGIN_AIRPORT'].map(airport_to_idx)
    
    # We want time to be a floating point scalar (e.g., total minutes since start of year)
    # The EM algorithm mathematically requires time to be a float scalar t_i
    start_time = df['ACTUAL_DEPARTURE_TIMESTAMP'].min()
    df['TIME_MINUTES'] = (df['ACTUAL_DEPARTURE_TIMESTAMP'] - start_time).dt.total_seconds() / 60.0
    
    event_log = df[['TIME_MINUTES', 'NODE', 'DEPARTURE_DELAY']]
    
    out_path = os.path.join(data_dir, 'events.csv')
    event_log.to_csv(out_path, index=False)
    
    print(f"Success! Continuous-Time Event Log saved to: {out_path}")
    print(f"Total Events: {len(event_log)}")
    print(f"Adjacency Sparsity: {adj_matrix.mean() * 100:.2f}% connections exist.")

if __name__ == "__main__":
    prepare_continuous_data()
