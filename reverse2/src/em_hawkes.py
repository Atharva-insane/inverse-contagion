import numpy as np
import os
import pandas as pd
import pickle
import time

class NetworkConstrainedHawkesEM:
    def __init__(self, num_nodes, adj_matrix, beta=0.01):
        self.num_nodes = num_nodes
        self.adj_matrix = adj_matrix # (N, N) binary mask of allowed edges
        self.beta = beta
        
        # Initialize parameters
        self.mu = np.random.uniform(0.01, 0.1, num_nodes)
        
        # Initialize alpha, strictly enforcing the network constraint from day 1
        self.alpha = np.random.uniform(0.01, 0.1, (num_nodes, num_nodes))
        self.alpha = self.alpha * self.adj_matrix
        
    def fit(self, times, nodes, T, max_iter=20, tol=1e-4):
        """
        Expectation-Maximization (EM) algorithm for Multivariate Hawkes Process.
        times: 1D array of event times (sorted)
        nodes: 1D array of event node IDs (0 to N-1)
        T: Total observation time
        """
        M = len(times)
        print(f"Starting EM Algorithm on {M} events over {T:.2f} time units...")
        
        # Pre-allocate branching probability matrix P
        # P[j, i] represents probability that event i triggered event j
        # Because M can be large, we use a sparse approach by only looking back a finite window.
        # But for M=10000, we can use a dense upper triangular matrix or list of dicts.
        
        for iteration in range(max_iter):
            start_t = time.time()
            
            # E-STEP
            # Compute log-likelihood and branching probabilities
            log_lik = 0.0
            
            # P_diag represents probability event j is background
            P_diag = np.zeros(M)
            
            # expected offspring sum for M-step
            sum_P_ij = np.zeros((self.num_nodes, self.num_nodes))
            
            # To avoid O(M^2) exploding, we limit the lookback window to where e^-beta*dt is significant
            # e.g., beta = 0.01, half-life is ~69 mins. Look back 500 mins.
            lookback_mins = 500.0
            
            for j in range(M):
                t_j = times[j]
                u_j = nodes[j]
                
                # Background intensity
                lam_j = self.mu[u_j]
                
                # Find valid past events
                # We can optimize this by finding the index where times > t_j - lookback_mins
                i_start = np.searchsorted(times, t_j - lookback_mins)
                
                if i_start < j:
                    past_t = times[i_start:j]
                    past_u = nodes[i_start:j]
                    
                    dt = t_j - past_t
                    # Kernel values
                    kappa = self.alpha[past_u, u_j] * self.beta * np.exp(-self.beta * dt)
                    lam_j += np.sum(kappa)
                    
                    # Store probabilities
                    # P[i->j] = kappa / lam_j
                    p_ij = kappa / lam_j
                    
                    # Accumulate for M-step
                    # np.add.at is used to accumulate grouped by origin node
                    np.add.at(sum_P_ij[:, u_j], past_u, p_ij)
                    
                P_diag[j] = self.mu[u_j] / lam_j
                log_lik += np.log(lam_j)
            
            # Subtract the integral of the intensity
            integral_mu = np.sum(self.mu) * T
            
            # Integral of trigger functions
            # sum_{i=1}^M sum_{v=1}^N alpha_{u_i, v} (1 - e^{-beta(T - t_i)})
            # For each node u, how many events occurred?
            integral_alpha = 0.0
            sum_integral_term = np.zeros(self.num_nodes)
            for i in range(M):
                u_i = nodes[i]
                term = 1.0 - np.exp(-self.beta * (T - times[i]))
                sum_integral_term[u_i] += term
                integral_alpha += np.sum(self.alpha[u_i, :]) * term
                
            log_lik -= (integral_mu + integral_alpha)
            
            # M-STEP
            # Update mu
            for u in range(self.num_nodes):
                mask = (nodes == u)
                if np.sum(mask) > 0:
                    self.mu[u] = np.sum(P_diag[mask]) / T
            
            # Update alpha
            for u in range(self.num_nodes):
                for v in range(self.num_nodes):
                    if self.adj_matrix[u, v] > 0: # NETWORK CONSTRAINT
                        if sum_integral_term[u] > 0:
                            self.alpha[u, v] = sum_P_ij[u, v] / sum_integral_term[u]
                    else:
                        self.alpha[u, v] = 0.0 # Strictly enforce zeroes
            
            duration = time.time() - start_t
            print(f"Iter {iteration+1:02d} | Log-Likelihood: {log_lik:.2f} | Time: {duration:.1f}s")
            
        print("EM Algorithm Converged.")
        return log_lik

if __name__ == "__main__":
    pass
