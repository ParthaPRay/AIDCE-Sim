from simulator import SimulationConfig, compare_scenarios, simulate_ai_data_center, summarize_kpis
from forecasting import run_forecasting

cfg = SimulationConfig(days=4, freq_minutes=30, gpus=1024, grid_capacity_mw=3.0)
df = simulate_ai_data_center(cfg)
assert len(df) == 4 * 48
assert (df["facility_power_mw"] >= df["it_power_mw"]).all()
assert (df["grid_power_mw"] >= 0).all()
kpis = summarize_kpis(df, cfg)
assert kpis["Facility energy (MWh)"] > 0
forecast = run_forecasting(df["grid_power_mw"], cfg.freq_minutes, forecast_hours=12, holdout_hours=12)
assert len(forecast.future) == 24
summary, outputs = compare_scenarios(cfg, ["baseline", "integrated"])
assert len(summary) == 2
print("Smoke test passed")
print(summary[["Facility energy (MWh)", "Peak grid load (MW)", "Carbon emissions (tCO2e)"]])
