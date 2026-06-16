# Inverse Contagion (Phase 2): Pure Statistical EM Hawkes

Welcome to **Phase 2** of the Inverse Contagion project. 

While Phase 1 focused on maximizing predictive power using a Deep Learning Neural Graph (GAT), Phase 2 strips away the neural network entirely. We have rebuilt the system as a **Pure Statistical Multivariate Hawkes Process**, estimated via the exact Expectation-Maximization (EM) algorithm.

## Why the Shift? (The Academic Rigor)
During peer review/judging, a common critique of neural networks is that their "attention mechanisms" are too soft, allowing for biologically or physically impossible contagion pathways. 

To solve this, this `reverse2` architecture implements a **Hard Topological Constraint**. The background rates ($\mu$) and infectivity weights ($\alpha$) are estimated using EM, but if a physical flight route does not exist in the raw Adjacency Matrix ($A_{ij} = 0$), the algorithm mathematically forces $\alpha_{ij} = 0$. The model is forced to learn reality.

## The Architecture Pipeline

### 1. Continuous-Time Data Engineering
`src/data_prep.py`
Unlike deep learning which can use hourly bins, the EM algorithm requires an exact, continuous-time marked point process. This script parses the 500MB dataset down to the exact minute of occurrence: $\mathcal{H}_t = \{(t_i, u_i) : t_i < t\}$, and constructs the hard Adjacency constraint.

### 2. The Statistical Engine
`src/em_hawkes.py` & `src/train.py`
Implements the closed-form E-Step and M-Step for the Multivariate Hawkes Process. It calculates the latent probability $P_{ij}$ that event $i$ triggered event $j$ using an exponential decay kernel $\kappa(t) = \beta e^{-\beta t}$, and estimates the true MLE parameters.

### 3. Network Morphology (Systemic Risk)
`src/morphology.py`
Transforms the raw math into actionable Business Intelligence. By analyzing the strictly constrained $\alpha$ matrix, it calculates:
- **In-Strength (Vulnerability)**: Airports that act as *Contagion Sinks* (e.g., ORD, DEN, ATL). They absorb massive delays but don't propagate them.
- **Out-Strength (Contagiousness)**: Airports that act as *Contagion Sources* (e.g., MKE, DCA, AUS). If a delay hits these fragile airports, it cascades across the US.

### 4. Predictive Evaluation
`src/evaluate.py`
Mathematically proves the model works. By calculating the expected integral of the intensity function $\int \lambda_i(t) dt$, we achieve a **15.06% reduction in Mean Absolute Error (MAE)** over a naive baseline.

### 5. The Presentation Visual Suite
`src/plot_presentation_suite.py`
Generates the 4 core visuals for the final presentation:
1. `network_topology.png`: A Kamada-Kawai graph of the Top 100 contagion pathways, sized by Out-Strength.
2. `alpha_heatmap.png`: The sparse, strictly constrained $\alpha$ matrix.
3. `hawkes_decomposition_ORD.png`: The continuous-time breakdown of background vs. self-exciting delays.
4. `mae_comparison.png`: The definitive 15% ROI bar chart.

## How to Run

Execute the pipeline in exact sequence:
```bash
cd src
python data_prep.py              # Extract Point Process & Adjacency
python train.py                  # Run the EM Algorithm
python evaluate.py               # Prove the 15% MAE Improvement
python morphology.py             # Identify Sources vs Sinks
python plot_presentation_suite.py # Generate the visuals
```
