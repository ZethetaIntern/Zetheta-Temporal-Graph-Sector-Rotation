# Institutional Quantitative Recommendations & Strategy Report

## 1. Executive Summary
This report outlines the deployment and backtest validation of a two-tier quantitative framework for the Indian NSE sectoral indices:
* **Project 1B (TGSRN):** Graph Attention Networks (GAT) combined with GRU cells for dynamic sector rotation and systemic contagion tracking.
* **Project 2 (Constrained Portfolio Optimization):** SLSQP Quadratic Programming optimizer incorporating strict long-only mandates, 50% single-sector caps, and machine learning alpha-tilt integration.

## 2. Key Performance Metrics
* **GAT-Integrated Portfolio Annualized Return:** 11.76%
* **GAT-Integrated Portfolio Volatility:** 16.81%
* **GAT-Integrated Portfolio Sharpe Ratio:** 0.70
* **Maximum Drawdown:** -38.97%

## 3. Systemic Risk Insights
* **Contagion Detection:** The dynamic graph adjacency matrix successfully captured market-wide structural shifts, notably identifying the inter-sector correlation spike to **0.81** during early 2020 market stress.
* **Drawdown Mitigation:** Introducing covariance minimization and maximum sector caps effectively controlled tail risk during macro shocks compared to unhedged single-sector rotation models.

## 4. Implementation Guidelines for Live Deployment
1. **Execution Frequency:** Maintain daily or weekly rebalancing based on the derived turnover profiles to balance alpha capture with transaction friction.
2. **Risk Limits:** Enforce strict single-sector concentration caps (maximum 50%) to prevent excessive cyclical exposure.
3. **Regime Monitoring:** Continuously track rolling inter-sector correlation matrices as a leading indicator for liquidity crunches and systemic decoupling.
