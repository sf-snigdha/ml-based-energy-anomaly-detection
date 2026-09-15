# HP-OFF and SHAP findings

The physically simulated HP-OFF evaluation contains 399 observations.

Frozen-model performance:
- ROC-AUC: 0.9597
- PR-AUC: 0.9113
- Detected at frozen threshold: 35 of 399
- Threshold-specific recall: 8.77%

SHAP is applied to the same anomaly-score function used by the frozen model.

Leading global HP-OFF SHAP contributions:
- supply temperature: 30.61%
- return temperature: 14.23%
- CHP electrical output: 8.29%
- HP electrical power: 7.28%
- unserved heat: 6.71%
- HP thermal output: 6.68%
- bus voltage: 5.37%
- heat demand: 4.73%
- maximum line loading: 4.11%
- grid import: 2.95%

Detected HP-OFF events generally show broader disturbance propagation across the coupled system.
Missed events can still show strong heating-network temperature deviations but weaker simultaneous
effects in unserved heat, CHP/P2H response, electrical demand, voltage or grid import.

SHAP explains model behavior and does not prove physical causality.
