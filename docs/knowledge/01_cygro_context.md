# CyGro anomaly-detection context

The project uses 15-minute sector-coupled power and heat observations from April to October 2026.

The frozen model is a regime-aware Isolation Forest using 14 features:
heat_demand_mw, base_electric_load_mw, p_site_pv_sim_total_mw,
p_hp_el_sim_total_mw, q_hp_th_sim_total_mw, cop_model,
p_p2h_el_sim_total_mw, p_bhkw_el_sim_total_mw,
powerlahn_bus_vm_pu, grid_import_mw, max_line_loading_pct,
unserved_heat_mw, powerlahn_supply_temp_k, powerlahn_return_temp_k.

The project anomaly score is:
anomaly_score = -IsolationForest.score_samples(X)

The frozen anomaly threshold is 0.6605666093730446.
