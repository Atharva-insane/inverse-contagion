import pandas as pd
import numpy as np
import os
import pickle

def extract_continuous_log(csv_path='../flights.csv', data_dir='../processed_data', top_k_airports=50):
    print(f"Loading data from {csv_path}...")
    
    # Load raw data
    usecols = ['YEAR', 'MONTH', 'DAY', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT', 
               'SCHEDULED_DEPARTURE', 'DEPARTURE_DELAY']
    
    try:
        df = pd.read_csv(csv_path, usecols=usecols)
    except FileNotFoundError:
        print(f"Error: Could not find {csv_path}")
        return
        
    print("Filtering for severe contagion events (>15 min delays)...")
    df = df.dropna(subset=['ORIGIN_AIRPORT', 'DESTINATION_AIRPORT', 'DEPARTURE_DELAY', 'SCHEDULED_DEPARTURE'])
    
    # A contagion event is defined as a delay > 15 mins
    df = df[df['DEPARTURE_DELAY'] > 15].copy()
    
    # Load the EXACT top 50 airports used by the model for consistency
    try:
        with open(os.path.join(data_dir, 'airports.pkl'), 'rb') as f:
            top_airports = pickle.load(f)
            # Ensure it is exactly top_k
            top_airports = top_airports[:top_k_airports]
    except FileNotFoundError:
        print("airports.pkl not found. Recalculating Top 50 airports...")
        airport_counts = df['ORIGIN_AIRPORT'].value_counts()
        top_airports = airport_counts.head(top_k_airports).index.tolist()
        
    print(f"Filtering dataset to the Top {len(top_airports)} Mega-Hub airports...")
    df = df[df['ORIGIN_AIRPORT'].isin(top_airports) & df['DESTINATION_AIRPORT'].isin(top_airports)]
    
    print("Constructing Real-Time Continuous Timestamps...")
    # SCHEDULED_DEPARTURE is in format HHMM (e.g., 5 = 00:05, 1230 = 12:30, 2400 = 00:00 next day)
    # We must cleanly parse this
    df['SCHEDULED_DEPARTURE'] = df['SCHEDULED_DEPARTURE'].astype(int)
    
    # Extract Hours and Minutes
    df['HOUR'] = df['SCHEDULED_DEPARTURE'] // 100
    df['MINUTE'] = df['SCHEDULED_DEPARTURE'] % 100
    
    # Handle the 24:00 edge case in aviation data (it means 00:00 of the NEXT day)
    # For simplicity, we cap it at 23 to avoid datetime out-of-bounds, or replace 24 with 0
    df['HOUR'] = df['HOUR'].apply(lambda x: 0 if x >= 24 else x)
    
    # Create the base scheduled datetime
    df['SCHEDULED_DATETIME'] = pd.to_datetime({
        'year': df['YEAR'],
        'month': df['MONTH'],
        'day': df['DAY'],
        'hour': df['HOUR'],
        'minute': df['MINUTE']
    }, errors='coerce')
    
    # Drop any that failed to parse
    df = df.dropna(subset=['SCHEDULED_DATETIME'])
    
    # Add the exact delay minutes to get the precise real-world occurrence of the event
    df['ACTUAL_DEPARTURE_TIMESTAMP'] = df['SCHEDULED_DATETIME'] + pd.to_timedelta(df['DEPARTURE_DELAY'], unit='m')
    
    # Sort chronologically to represent the true Continuous-Time Event Log
    df = df.sort_values(by='ACTUAL_DEPARTURE_TIMESTAMP')
    
    print("Finalizing Continuous-Time Event Log...")
    event_log = df[['ACTUAL_DEPARTURE_TIMESTAMP', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT', 'DEPARTURE_DELAY']]
    
    # Save the artifact
    out_path = os.path.join(data_dir, 'continuous_event_log.csv')
    os.makedirs(data_dir, exist_ok=True)
    event_log.to_csv(out_path, index=False)
    
    print(f"Success! Continuous-Time Event Log saved to: {out_path}")
    print(f"Total Events Logged: {len(event_log)}")
    print("\nSample of the Event Log:")
    print(event_log.head(10).to_string(index=False))

if __name__ == "__main__":
    extract_continuous_log()
