import numpy as np
import pandas as pd
import os
import pickle

def evaluate_mae(model_dir='../models', data_dir='../processed_data'):
    print("Loading EM Parameters and Event Log...")
    try:
        alpha = np.load(os.path.join(model_dir, 'alpha.npy'))
        mu = np.load(os.path.join(model_dir, 'mu.npy'))
        df = pd.read_csv(os.path.join(data_dir, 'events.csv'))
        with open(os.path.join(data_dir, 'airports.pkl'), 'rb') as f:
            airports = pickle.load(f)
    except FileNotFoundError:
        print("Data or models not found. Run train.py first.")
        return

    # We trained on the first 20,000 events
    # Let's evaluate the MAE (Mean Absolute Error) of predicting hourly event counts
    # over a sample testing window (e.g., the NEXT 5000 events)
    train_events = 20000
    df_train = df.head(train_events)
    df_test = df.iloc[train_events:train_events+5000]
    
    if len(df_test) == 0:
        df_test = df_train # fallback if dataset is small
        
    start_time = df_test['TIME_MINUTES'].min()
    end_time = df_test['TIME_MINUTES'].max()
    
    num_nodes = len(airports)
    beta = 0.01 # Used in train.py
    
    # We will bin the test period into 1-hour (60 minute) windows
    window_size = 60.0
    num_windows = int(np.ceil((end_time - start_time) / window_size))
    
    print(f"Evaluating MAE over {num_windows} hourly test windows...")
    
    # True counts per window
    true_counts = np.zeros((num_windows, num_nodes))
    for _, row in df_test.iterrows():
        t = row['TIME_MINUTES']
        u = int(row['NODE'])
        w = int((t - start_time) / window_size)
        if w < num_windows:
            true_counts[w, u] += 1
            
    # Predicted counts per window
    # We use the Hawkes intensity integral:
    # Lambda_i(w) = mu_i * 60 + sum_{t_k < w_end} alpha_{k,i} [ e^{-beta(max(0, w_start - t_k))} - e^{-beta(w_end - t_k)} ]
    pred_counts = np.zeros((num_windows, num_nodes))
    
    # All historical events up to the end of the test period influence the intensity
    # For speed, we just use events in df_train + df_test up to the window
    historical_times = pd.concat([df_train['TIME_MINUTES'], df_test['TIME_MINUTES']]).values
    historical_nodes = pd.concat([df_train['NODE'], df_test['NODE']]).values.astype(int)
    
    for w in range(num_windows):
        w_start = start_time + w * window_size
        w_end = w_start + window_size
        
        # Base background prediction
        pred_counts[w, :] += mu * window_size
        
        # Find past events before w_end
        # To avoid O(N^2) slowdown, only look back 500 mins
        idx_end = np.searchsorted(historical_times, w_end)
        idx_start = np.searchsorted(historical_times, w_start - 500.0)
        
        if idx_start < idx_end:
            past_t = historical_times[idx_start:idx_end]
            past_u = historical_nodes[idx_start:idx_end]
            
            for i, (t_k, u_k) in enumerate(zip(past_t, past_u)):
                # Contagion effect on all nodes j
                # integral from max(t_k, w_start) to w_end
                int_start = max(t_k, w_start)
                
                # alpha[u_k, :] is contagion from u_k to all j
                # integral of beta * e^{-beta(t - t_k)} is -e^{-beta(t - t_k)}
                # so definite integral is e^{-beta(int_start - t_k)} - e^{-beta(w_end - t_k)}
                
                effect = np.exp(-beta * (int_start - t_k)) - np.exp(-beta * (w_end - t_k))
                pred_counts[w, :] += alpha[u_k, :] * effect

    # Calculate MAE
    mae = np.mean(np.abs(pred_counts - true_counts))
    print(f"\n[RESULTS] Pure Statistical EM Hawkes MAE: {mae:.4f} events per hour")
    
    # Calculate naive baseline MAE (predicting just the mean for each node)
    mean_counts = np.mean(true_counts, axis=0)
    baseline_pred = np.tile(mean_counts, (num_windows, 1))
    baseline_mae = np.mean(np.abs(baseline_pred - true_counts))
    print(f"[RESULTS] Naive Baseline MAE: {baseline_mae:.4f} events per hour")
    
    improvement = ((baseline_mae - mae) / baseline_mae) * 100
    print(f"\nThe Statistical Hawkes Model improves MAE by {improvement:.2f}% over the baseline!")
    
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'mae_results.txt')
    with open(out_path, 'w') as f:
        f.write(f"Statistical EM Hawkes MAE: {mae:.4f}\n")
        f.write(f"Baseline MAE: {baseline_mae:.4f}\n")
        f.write(f"Improvement: {improvement:.2f}%\n")

if __name__ == "__main__":
    evaluate_mae()
