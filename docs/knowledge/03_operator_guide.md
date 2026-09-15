# Operator interpretation guide

An anomaly score indicates statistical abnormality and should be interpreted together with SHAP
and physical system context.

Supply and return temperatures describe the heating-network state.
Unserved heat represents thermal demand that cannot be supplied.
Grid import represents net electrical import.
HP electrical power and HP thermal output characterize heat-pump operation.
CHP and P2H variables capture coupled asset behavior and possible compensating dispatch.

A root-cause diagnosis should combine anomaly score, feature attributions, time-series context,
equipment state and engineering checks.
