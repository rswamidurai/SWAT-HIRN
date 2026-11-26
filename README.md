# SWAT-HIRN: Hybrid SWAT–Hopfield Recurrent Neural Networks

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)

Official implementation of **SWAT-HIRN (v1.0)** from the paper:  
> **Hybrid SWAT–Hopfield Recurrent Neural Networks: Resolving Multi-Decadal Vadose-Zone Hysteresis for Deep Groundwater Recharge Forecasting**  
> Rajendran Swamidurai and Uma Kannan  
> *Department of Mathematics and Computer Science, Alabama State University, Montgomery, AL, USA*

---

## Overview

**SWAT-HIRN** couples semi-distributed Soil and Water Assessment Tool (SWAT) Hydrological Response Unit (HRU) outputs with continuous **Modern Hopfield Networks (MHNs)** embedded inside gated recurrent backbones (LSTM / GRU). By equipping recurrent state transitions with a non-parametric associative memory bank ($N = 512$ learned hydro-climatic prototypes, $d_k = d_v = 128$) operating under single-step Concave-Convex Procedure (CCCP) energy minimization, SWAT-HIRN resolves non-linear vadose-zone moisture hysteresis and multi-month deep percolation delays while eliminating long-range gradient vanishing.

### Key Highlights
* **Multi-Horizon Accuracy:** Achieves **10.32 mm RMSE** ($R^2 = 0.906$, $\text{NSE} = 0.887$) at a 6-month lead time, delivering a **25.8% error reduction** over Vanilla LSTM (13.92 mm).
* **Mass Conservation:** Preserves cumulative volumetric mass balance within **$-1.35\%$** over the 2041–2049 out-of-sample evaluation period.
* **Observational Fidelity:** Aligns with 31 subbasin Water-Table Fluctuation (WTF) borewell measurements across the Wolf Bay watershed with a **+1.09%** relative bias.

---

## Repository Structure

```text
SWAT-HIRN/
├── data/
│   ├── wolf_bay_hru_1980_2049.csv      # 278,880 monthly SWAT HRU records (10 LULC classes)
│   └── subbasin_borewell_wtf.csv       # 31-subbasin field borewell validation baseline
├── models/
│   ├── hopfield_layer.py               # Continuous Modern Hopfield associative memory & CCCP contraction
│   ├── swat_hirn.py                    # Hopfield-LSTM and Hopfield-GRU architectures with gated fusion
│   └── baselines.py                    # Linear AR, Vanilla LSTM/GRU, Transformer, and TCN baselines
├── checkpoints/
│   └── hopfield_lstm_best.pt           # Pre-trained weights (N=512, d=128, beta=0.0884)
├── scripts/
│   ├── train.py                        # Training script (1980–2040 split, AdamW, cosine annealing)
│   ├── evaluate.py                     # Multi-horizon (1, 3, 6-month) & LULC evaluation metrics
│   └── infer.py                        # Sub-millisecond surrogate inference pipeline (Algorithm 3)
├── requirements.txt
├── LICENSE
└── README.md
