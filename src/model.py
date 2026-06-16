import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphAttentionInfectivity(nn.Module):
    """
    Computes the dynamic infectivity matrix alpha_{ij} using Graph Attention.
    """
    def __init__(self, in_features, dropout=0.2, alpha=0.2):
        super(GraphAttentionInfectivity, self).__init__()
        self.dropout = dropout
        self.alpha = alpha

        self.W = nn.Parameter(torch.empty(size=(in_features, in_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        
        self.a = nn.Parameter(torch.empty(size=(2*in_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        B, N = h.size(0), h.size(1)
        Wh = torch.matmul(h, self.W)
        
        a_input = self._prepare_attentional_mechanism_input(Wh)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(-1))

        zero_vec = -9e15 * torch.ones_like(e)
        if adj.dim() == 2:
            adj = adj.unsqueeze(0).expand(B, N, N)
            
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=-1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        return attention

    def _prepare_attentional_mechanism_input(self, Wh):
        B, N, D = Wh.size()
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=1)
        Wh_repeated_alternating = Wh.repeat(1, N, 1)
        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=-1)
        return all_combinations_matrix.view(B, N, N, 2 * D)

class NeuralGraphHawkesProcess(nn.Module):
    """
    Discrete-Time Neural Graph Hawkes Process for Cascade Modeling.
    Intensity: lambda(t) = mu(t) + alpha * sum_{k}(decay(k) * y(t-k))
    """
    def __init__(self, num_nodes, node_features, seq_len):
        super(NeuralGraphHawkesProcess, self).__init__()
        self.num_nodes = num_nodes
        self.seq_len = seq_len
        
        # Exogenous Background Rate mu(t)
        # We use node features (like time of day) to predict baseline delay
        self.mu_net = nn.Sequential(
            nn.Linear(node_features, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Softplus() # Intensity must be positive
        )
        
        # Infectivity Matrix alpha_{ij}
        self.infectivity_net = GraphAttentionInfectivity(in_features=node_features)
        
        # Time Decay Parameter beta
        # We learn a base beta for the exponential decay
        self.beta = nn.Parameter(torch.tensor([0.1]))

    def forward(self, x, adj):
        """
        x shape: (Batch, SeqLen, Nodes, Features)
        Features[..., 0] is the historical delay magnitude.
        Features[..., 1:] are covariates (time of day etc).
        """
        batch_size = x.size(0)
        
        # We want to predict the intensity at time t (next step).
        # We use the covariates at the LAST time step as a proxy for current time
        # (or assume covariates don't change drastically hour-to-hour).
        last_x = x[:, -1, :, :] # (B, N, F)
        
        # 1. Background Intensity mu(t)
        mu = self.mu_net(last_x) # (B, N, 1)
        
        # 2. Infectivity Matrix alpha_{ij}
        alpha = self.infectivity_net(last_x, adj) # (B, N, N)
        self.saved_alpha = alpha # For explainability
        
        # 3. Excitation / Contagion
        # sum_{k=1 to W} e^(-beta * k) * y(t-k)
        # x[:, :, :, 0] are the historical delays. Shape: (B, SeqLen, N)
        delays = x[:, :, :, 0]
        
        # Create decay vector: e^(-beta * k) where k goes from SeqLen down to 1
        # so delays[:, -1, :] corresponds to k=1 (1 hour ago)
        k_values = torch.arange(self.seq_len, 0, -1, device=x.device, dtype=torch.float32)
        decay = torch.exp(-F.softplus(self.beta) * k_values) # (SeqLen)
        
        # Apply decay to historical delays
        # delays: (B, SeqLen, N)
        # decay: (SeqLen) -> (1, SeqLen, 1)
        decay = decay.unsqueeze(0).unsqueeze(-1)
        decayed_history = delays * decay # (B, SeqLen, N)
        
        # Sum over time
        aggregated_history = decayed_history.sum(dim=1) # (B, N)
        
        # Multiply by infectivity matrix
        # aggregated_history shape (B, N). We want to multiply by alpha (B, N, N)
        # For node i, contagion = sum_j alpha_{ij} * aggregated_history_j
        contagion = torch.bmm(alpha, aggregated_history.unsqueeze(-1)) # (B, N, 1)
        self.saved_contagion = contagion
        self.saved_mu = mu
        
        # 4. Total Intensity
        lambda_t = mu + contagion # (B, N, 1)
        
        return lambda_t

    def extract_hawkes_components(self):
        """
        Returns the decoupled background and contagion components from the last forward pass.
        """
        return self.saved_mu, self.saved_contagion, self.saved_alpha
