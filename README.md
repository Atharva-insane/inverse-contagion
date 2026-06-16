# Inverse Contagion: Generative Modeling of Systemic Risk in Aviation Networks

<p align="center">
  <em>A Deep Learning Framework for Reverse-Engineering Spatial-Temporal Cascades via Neural Graph Hawkes Processes (NGHP).</em>
</p>

---

## Ⅰ. Abstract

Understanding the propagation of systemic failure in highly interconnected transportation networks remains a critical challenge in operations research. Traditional methodologies rely on forward-simulation paradigms or linear, moment-based statistical estimators (e.g., Bacry et al., 2012) which fail to capture the non-stationary, non-linear dynamics of modern aviation. 

This research introduces an **Inverse Contagion** framework. Rather than injecting artificial perturbations, we fit a **Neural Graph Hawkes Process (NGHP)** directly to historical observational data of the US aviation network. By integrating Multi-Head Graph Attention Networks (GAT) with the mathematical rigor of self-exciting point processes, this model successfully decouples exogenous environmental delays from endogenous network contagion, providing an interpretable, highly accurate Digital Twin of systemic risk.

---

## Ⅱ. Theoretical Formulation: The NGHP Architecture

The core innovation of this framework is the integration of Graph Representation Learning into the continuous-time formulation of the Multivariate Hawkes Process.

Let $\lambda_i(t)$ denote the conditional intensity of a delay event at node (airport) $i$ at time $t$. The NGHP formulates this intensity as:

$$ \lambda_i(t) = \mu_i(X_t) + \sum_{j \in \mathcal{V}} \alpha_{ij}(H_t) \int_{-\infty}^{t} \kappa(t - s) \, dN_j(s) $$

### 1. Non-Stationary Exogenous Intensity: $\mu_i(X_t)$
Unlike traditional models that assume a static background rate, our framework utilizes a multi-layer perceptron to map cyclical temporal covariates $X_t$ (sine/cosine transformations of the diurnal cycle) to a dynamic baseline intensity. This isolates purely local, spontaneous failures (e.g., localized weather events).

### 2. Dynamic Infectivity Matrix via GAT: $\alpha_{ij}(H_t)$
Standard moment-based Hawkes estimators rely on static cross-correlations, yielding a fixed influence matrix. In contrast, our model implements a **Spatial-Temporal Graph Attention Layer** over the hidden state representations $H_t$. The infectivity weight $\alpha_{ij}$ is computed dynamically per epoch, allowing the model to recognize that a specific directed edge (flight path) may be highly infectious during peak congestion but benign during off-peak hours.

### 3. Exponential Decay Kernel: $\kappa(\Delta t)$
The model learns a global decay parameter $\beta$ to parameterize the memory of the network, quantifying the half-life of a localized delay before it diffuses or is absorbed by network slack.

---

## Ⅲ. Quantitative Evaluation & Discoveries

The NGHP framework was trained and evaluated on the US DOT Flight Delays dataset, restricted to the top 50 highest-volume nodes. 

### Predictive Efficacy
The generative Digital Twin successfully captured the underlying transition dynamics of the system, achieving state-of-the-art predictive accuracy on holdout temporal sequences:
* **Mean Absolute Error (MAE)**: 7.77 minutes
* **Root Mean Squared Error (RMSE)**: 17.27 minutes

### Topological Vulnerability Assessment
By explicitly extracting the learned Attention Weights ($\alpha$), we calculate a **Multi-Exposure Score (MES)** for the topology. The MES quantifies the in-degree of significant contagion pathways for a given node. Our findings mathematically confirm that the highest systemic risk is concentrated in mega-hubs heavily reliant on tight turnaround scheduling:
1. **ATL (Atlanta)**: MES = 47
2. **DEN (Denver)**: MES = 45
3. **DFW (Dallas-Fort Worth)**: MES = 44

---

## Ⅳ. Repository Structure & Execution

```text
reverse/
├── flights.csv              # Raw Observational Data (US DOT)
├── airports.csv             # Geospatial Metadata
├── requirements.txt         # Environment Dependencies
├── models/                  # Serialized NGHP PyTorch Weights
├── output/                  # Geospatial and Topological Visualizations
├── processed_data/          # Spatio-Temporal Tensors and Adjacency Masks
└── src/
    ├── data_prep.py         # Extracts sequential cascades and cyclical covariates
    ├── model.py             # Defines the NGHP and Graph Attention topology
    ├── train.py             # Optimizes the Hawkes intensity objective function
    ├── evaluate.py          # Computes quantitative error metrics (MAE/RMSE)
    ├── infer.py             # Decouples exogenous vs. endogenous intensity factors
    └── visualize.py         # Renders geospatial plots of the learned \alpha matrix
```

### Reproducibility Guide
1. **Environment Setup**: Ensure CUDA availability for optimal tensor operations. Execute `pip install -r requirements.txt`.
2. **Feature Extraction**: Execute `src/data_prep.py` to compile the spatio-temporal features.
3. **Model Fitting**: Execute `src/train.py` to fit the NGHP to the empirical cascade distributions.
4. **Analysis Generation**: Execute `src/evaluate.py`, `src/infer.py`, and `src/visualize.py` to replicate the vulnerability findings and generate geospatial topology maps.

---
*Authored for the advancement of systemic risk modeling in complex transportation topologies.*
