from __future__ import annotations

import json
from dataclasses import asdict

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from forecasting import run_forecasting
from simulator import (
    DEFAULT_SCENARIO_LABELS,
    SimulationConfig,
    compare_scenarios,
    monte_carlo_summary,
    simulate_ai_data_center,
    summarize_kpis,
)

st.set_page_config(
    page_title="AIDCE-Sim",
    page_icon="⚡",
    layout="wide",
)

st.title("AIDCE-Sim: AI Data-Centre Energy Pattern Simulator")
st.caption(
    "A transparent, data-free simulator for workload-driven electricity demand, "
    "facility overhead, grid interaction, sustainability metrics, and load forecasting."
)

st.markdown(
    """
    **Developer:** Partha Pratim Ray, Sikkim University, India, [parthapratimray1986@gmail.com](mailto:parthapratimray1986@gmail.com)  
    **Date:** August 2, 2026  
    """
)

with st.expander("What this simulator does—and does not do", expanded=False):
    st.markdown(
        """
- Generates **synthetic** preparation, training, fine-tuning, and inference traces from explicit assumptions.
- Converts utilization into GPU, CPU, storage, network, cooling, auxiliary, facility, and grid power.
- Estimates PUE, electricity cost, operational carbon, water use, ramps, capacity exceedance, renewable use, and battery behavior.
- Compares intervention scenarios and forecasts the simulated grid-load series.
- It is **not** a substitute for measured telemetry, electrical transient/EMT studies, or a utility interconnection study.
        """
    )


def number(label, value, min_value, max_value, step, help_text=None):
    return st.number_input(
        label,
        min_value=min_value,
        max_value=max_value,
        value=value,
        step=step,
        help=help_text,
    )


with st.sidebar:
    st.header("Simulation controls")
    with st.expander("Time and reproducibility", expanded=True):
        start_date = st.date_input("Start date", value=pd.Timestamp("2026-01-01"))
        days = st.slider("Simulation horizon (days)", 2, 90, 14)
        freq_minutes = st.selectbox("Time resolution", [5, 10, 15, 30, 60], index=2)
        seed = number("Random seed", 42, 0, 100000, 1)

    with st.expander("Compute hardware", expanded=True):
        gpus = number("Number of GPUs/accelerators", 8192, 1, 500000, 128)
        gpu_tdp_w = number("GPU rated power (W)", 700.0, 50.0, 2000.0, 10.0)
        gpu_idle_fraction = st.slider("GPU idle power fraction", 0.05, 0.50, 0.18, 0.01)
        cpu_ratio = st.slider("CPU/GPU power ratio", 0.02, 0.50, 0.18, 0.01)
        storage_ratio = st.slider("Storage/GPU power ratio", 0.01, 0.25, 0.06, 0.01)
        network_ratio = st.slider("Network/GPU power ratio", 0.01, 0.30, 0.08, 0.01)

    with st.expander("AI workload mixture", expanded=True):
        prep = st.slider("Preparation share", 0, 100, 5)
        train = st.slider("Training share", 0, 100, 30)
        fine = st.slider("Fine-tuning share", 0, 100, 10)
        infer = st.slider("Inference share", 0, 100, 55)
        training_duty = st.slider("Training duty cycle", 0.10, 1.00, 0.88, 0.01)
        noise = st.slider("Correlated workload variability", 0.00, 0.20, 0.055, 0.005)
        spike_probability = st.slider("Inference spike probability/interval", 0.000, 0.100, 0.012, 0.001)
        spike_magnitude = st.slider("Inference spike magnitude", 0.00, 0.80, 0.22, 0.01)

    with st.expander("Facility and environment", expanded=False):
        cooling_type = st.selectbox(
            "Cooling preset",
            ["Custom", "Air", "Hybrid", "Direct-to-chip liquid", "Immersion"],
        )
        preset_pue = {
            "Custom": 1.22,
            "Air": 1.50,
            "Hybrid": 1.35,
            "Direct-to-chip liquid": 1.20,
            "Immersion": 1.08,
        }[cooling_type]
        base_pue = st.slider("Base PUE", 1.02, 2.50, float(preset_pue), 0.01)
        temp_sensitivity = st.slider("PUE temperature sensitivity", 0.000, 0.200, 0.035, 0.005)
        low_load_penalty = st.slider("Low-load PUE penalty", 0.000, 0.300, 0.055, 0.005)
        ambient_mean = st.slider("Mean ambient temperature (°C)", -10.0, 50.0, 24.0, 0.5)
        ambient_amp = st.slider("Daily temperature amplitude (°C)", 0.0, 20.0, 7.0, 0.5)
        wue = number("WUE (L per IT kWh)", 0.50, 0.0, 20.0, 0.05)

    with st.expander("Grid, price, and emissions", expanded=False):
        grid_capacity = number("Grid connection capacity (MW)", 15.0, 0.1, 5000.0, 0.5)
        carbon_intensity = number("Grid carbon intensity (kgCO₂e/kWh)", 0.45, 0.0, 2.0, 0.01)
        offpeak_tariff = number("Off-peak tariff (currency/kWh)", 0.08, 0.0, 10.0, 0.01)
        peak_tariff = number("Peak tariff (currency/kWh)", 0.15, 0.0, 10.0, 0.01)
        peak_start = st.slider("Peak period starts", 0, 23, 17)
        peak_end = st.slider("Peak period ends", peak_start + 1, 24, 22)

    with st.expander("Flexibility and onsite resources", expanded=False):
        dr_fraction = st.slider("Flexible workload shifted from peak", 0.0, 0.80, 0.0, 0.05)
        solar_mw = number("Solar capacity (MW)", 0.0, 0.0, 5000.0, 0.5)
        wind_mw = number("Wind capacity (MW)", 0.0, 0.0, 5000.0, 0.5)
        battery_power = number("Battery power (MW)", 0.0, 0.0, 5000.0, 0.5)
        battery_energy = number("Battery energy (MWh)", 0.0, 0.0, 50000.0, 1.0)
        battery_eff = st.slider("Battery round-trip efficiency", 0.50, 1.00, 0.90, 0.01)

shares_sum = prep + train + fine + infer
if shares_sum == 0:
    st.error("At least one workload share must be greater than zero.")
    st.stop()

cfg = SimulationConfig(
    start_date=str(start_date),
    days=int(days),
    freq_minutes=int(freq_minutes),
    seed=int(seed),
    gpus=int(gpus),
    gpu_tdp_w=float(gpu_tdp_w),
    gpu_idle_fraction=float(gpu_idle_fraction),
    cpu_to_gpu_power_ratio=float(cpu_ratio),
    storage_to_gpu_power_ratio=float(storage_ratio),
    network_to_gpu_power_ratio=float(network_ratio),
    preparation_share=float(prep),
    training_share=float(train),
    finetuning_share=float(fine),
    inference_share=float(infer),
    training_duty_cycle=float(training_duty),
    workload_noise=float(noise),
    spike_probability=float(spike_probability),
    spike_magnitude=float(spike_magnitude),
    base_pue=float(base_pue),
    pue_temperature_sensitivity=float(temp_sensitivity),
    pue_low_load_penalty=float(low_load_penalty),
    ambient_mean_c=float(ambient_mean),
    ambient_daily_amplitude_c=float(ambient_amp),
    wue_l_per_it_kwh=float(wue),
    grid_capacity_mw=float(grid_capacity),
    grid_carbon_kg_per_kwh=float(carbon_intensity),
    tariff_offpeak_per_kwh=float(offpeak_tariff),
    tariff_peak_per_kwh=float(peak_tariff),
    peak_hour_start=int(peak_start),
    peak_hour_end=int(peak_end),
    solar_capacity_mw=float(solar_mw),
    wind_capacity_mw=float(wind_mw),
    demand_response_shift_fraction=float(dr_fraction),
    battery_power_mw=float(battery_power),
    battery_energy_mwh=float(battery_energy),
    battery_roundtrip_efficiency=float(battery_eff),
)


@st.cache_data(show_spinner=False)
def cached_simulate(config_dict):
    return simulate_ai_data_center(SimulationConfig(**config_dict))


with st.spinner("Running simulation..."):
    df = cached_simulate(asdict(cfg))
kpis = summarize_kpis(df, cfg)

kpi_cols = st.columns(5)
headline = [
    ("Facility energy", f"{kpis['Facility energy (MWh)']:,.1f} MWh"),
    ("Peak grid load", f"{kpis['Peak grid load (MW)']:,.2f} MW"),
    ("Weighted PUE", f"{kpis['Energy-weighted PUE']:.3f}"),
    ("Carbon", f"{kpis['Carbon emissions (tCO2e)']:,.1f} tCO₂e"),
    ("Water", f"{kpis['Water consumption (m3)']:,.1f} m³"),
]
for col, (label, value) in zip(kpi_cols, headline):
    col.metric(label, value)

if kpis["Capacity exceedance (hours)"] > 0:
    st.warning(
        f"The simulated grid load exceeds the configured connection capacity for "
        f"{kpis['Capacity exceedance (hours)']:.2f} hours."
    )

load_tab, breakdown_tab, forecast_tab, scenario_tab, uncertainty_tab, download_tab = st.tabs(
    [
        "Load patterns",
        "Energy & sustainability",
        "Forecast",
        "Scenario comparison",
        "Uncertainty",
        "Downloads",
    ]
)

with load_tab:
    chart_df = df.reset_index(names="time")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=chart_df["time"], y=chart_df["it_power_mw"], name="IT load", mode="lines"))
    fig.add_trace(go.Scatter(x=chart_df["time"], y=chart_df["facility_power_mw"], name="Facility load", mode="lines"))
    fig.add_trace(go.Scatter(x=chart_df["time"], y=chart_df["grid_power_mw"], name="Grid load", mode="lines"))
    fig.add_hline(y=cfg.grid_capacity_mw, line_dash="dash", annotation_text="Grid capacity")
    fig.update_layout(title="AI data-centre power demand", xaxis_title="Time", yaxis_title="Power (MW)", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    stage_cols = [
        "preparation_utilization",
        "training_utilization",
        "finetuning_utilization",
        "inference_utilization",
    ]
    stage_long = chart_df.melt(id_vars="time", value_vars=stage_cols, var_name="stage", value_name="utilization")
    stage_long["stage"] = stage_long["stage"].str.replace("_utilization", "", regex=False).str.replace("finetuning", "fine-tuning").str.title()
    fig2 = px.line(stage_long, x="time", y="utilization", color="stage", title="Synthetic workload-stage utilization")
    fig2.update_yaxes(range=[0, 1.05], title="Normalized utilization")
    st.plotly_chart(fig2, use_container_width=True)

    heat = df[["grid_power_mw"]].copy()
    heat["date"] = heat.index.date.astype(str)
    heat["time_of_day"] = heat.index.strftime("%H:%M")
    pivot = heat.pivot_table(index="date", columns="time_of_day", values="grid_power_mw", aggfunc="mean")
    fig3 = px.imshow(pivot, aspect="auto", labels={"x": "Time of day", "y": "Date", "color": "MW"}, title="Grid-load heat map")
    st.plotly_chart(fig3, use_container_width=True)

with breakdown_tab:
    energy_components = pd.DataFrame(
        {
            "Component": ["GPU", "CPU", "Storage", "Network", "Cooling", "Auxiliary"],
            "Energy (MWh)": [
                df["gpu_power_mw"].sum() * cfg.freq_minutes / 60,
                df["cpu_power_mw"].sum() * cfg.freq_minutes / 60,
                df["storage_power_mw"].sum() * cfg.freq_minutes / 60,
                df["network_power_mw"].sum() * cfg.freq_minutes / 60,
                df["cooling_power_mw"].sum() * cfg.freq_minutes / 60,
                df["auxiliary_power_mw"].sum() * cfg.freq_minutes / 60,
            ],
        }
    )
    c1, c2 = st.columns(2)
    c1.plotly_chart(px.pie(energy_components, names="Component", values="Energy (MWh)", title="Facility energy composition"), use_container_width=True)
    duration = np.sort(df["grid_power_mw"].to_numpy())[::-1]
    exceedance = np.arange(1, len(duration) + 1) / len(duration) * 100
    duration_df = pd.DataFrame({"Exceedance probability (%)": exceedance, "Grid power (MW)": duration})
    c2.plotly_chart(px.line(duration_df, x="Exceedance probability (%)", y="Grid power (MW)", title="Load-duration curve"), use_container_width=True)

    summary_table = pd.DataFrame({"Metric": list(kpis.keys()), "Value": list(kpis.values())})
    st.dataframe(summary_table, use_container_width=True, hide_index=True)

    cumulative = df[["cost", "carbon_kg", "water_l"]].cumsum().reset_index(names="time")
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=cumulative["time"], y=cumulative["carbon_kg"] / 1000, name="Carbon (tCO₂e)"))
    fig4.add_trace(go.Scatter(x=cumulative["time"], y=cumulative["water_l"] / 1000, name="Water (m³)", yaxis="y2"))
    fig4.update_layout(title="Cumulative environmental footprint", yaxis_title="Carbon (tCO₂e)", yaxis2=dict(title="Water (m³)", overlaying="y", side="right"), hovermode="x unified")
    st.plotly_chart(fig4, use_container_width=True)

with forecast_tab:
    c1, c2 = st.columns(2)
    forecast_hours = c1.slider("Forecast horizon (hours)", 6, 168, 48, 6)
    holdout_hours = c2.slider("Back-test window (hours)", 6, 168, 24, 6)
    try:
        with st.spinner("Training and comparing forecasting models..."):
            forecast_result = run_forecasting(
                df["grid_power_mw"],
                cfg.freq_minutes,
                forecast_hours=forecast_hours,
                holdout_hours=holdout_hours,
            )
        st.success(f"Best back-tested model: {forecast_result.best_model}")
        st.dataframe(forecast_result.metrics, use_container_width=True, hide_index=True)

        future = forecast_result.future.reset_index(names="time")
        recent = df["grid_power_mw"].tail(min(len(df), int(72 * 60 / cfg.freq_minutes))).reset_index(name="actual_mw")
        recent.columns = ["time", "actual_mw"]
        ffig = go.Figure()
        ffig.add_trace(go.Scatter(x=recent["time"], y=recent["actual_mw"], name="Recent simulated load"))
        ffig.add_trace(go.Scatter(x=future["time"], y=future["forecast_mw"], name="Forecast"))
        ffig.add_trace(go.Scatter(x=future["time"], y=future["upper_95_mw"], line=dict(width=0), showlegend=False, hoverinfo="skip"))
        ffig.add_trace(go.Scatter(x=future["time"], y=future["lower_95_mw"], fill="tonexty", line=dict(width=0), name="Approx. 95% interval", hoverinfo="skip"))
        ffig.update_layout(title="Grid-load forecast", xaxis_title="Time", yaxis_title="MW", hovermode="x unified")
        st.plotly_chart(ffig, use_container_width=True)
    except Exception as exc:
        st.error(f"Forecasting could not be completed: {exc}")
        forecast_result = None

with scenario_tab:
    selected = st.multiselect(
        "Scenarios",
        options=list(DEFAULT_SCENARIO_LABELS.keys()),
        default=["baseline", "efficient_cooling", "grid_aware", "renewable_bess", "integrated"],
        format_func=lambda x: DEFAULT_SCENARIO_LABELS[x],
    )
    if selected:
        with st.spinner("Running intervention scenarios..."):
            scenario_summary, _ = compare_scenarios(cfg, selected)
        st.dataframe(scenario_summary, use_container_width=True)
        metric_choice = st.selectbox(
            "Comparison metric",
            [
                "Facility energy (MWh)",
                "Peak grid load (MW)",
                "Carbon emissions (tCO2e)",
                "Water consumption (m3)",
                "Electricity cost",
                "Maximum ramp (MW/min)",
            ],
        )
        plot_df = scenario_summary[[metric_choice]].reset_index()
        st.plotly_chart(px.bar(plot_df, x="Scenario", y=metric_choice, title=f"Scenario comparison: {metric_choice}"), use_container_width=True)
        st.caption("Scenario presets are transparent what-if assumptions, not measured performance guarantees.")
    else:
        st.info("Select at least one scenario.")

with uncertainty_tab:
    runs = st.slider("Monte Carlo runs", 10, 200, 40, 10)
    run_mc = st.button("Run uncertainty analysis", type="primary")
    if run_mc:
        with st.spinner("Running parameter uncertainty analysis..."):
            mc = monte_carlo_summary(cfg, runs=runs)
        q = mc.drop(columns="run").quantile([0.05, 0.50, 0.95]).T
        q.columns = ["5th percentile", "Median", "95th percentile"]
        st.dataframe(q, use_container_width=True)
        st.plotly_chart(px.histogram(mc, x="peak_grid_mw", nbins=20, title="Uncertainty distribution of peak grid load"), use_container_width=True)
        st.session_state["mc_results"] = mc
    elif "mc_results" in st.session_state:
        mc = st.session_state["mc_results"]
        st.dataframe(mc.describe().T, use_container_width=True)

with download_tab:
    st.subheader("Export reproducible results")
    output_df = df.reset_index(drop=True)
    st.download_button(
        "Download full time series CSV",
        output_df.to_csv(index=False).encode("utf-8"),
        file_name="aidce_sim_timeseries.csv",
        mime="text/csv",
    )
    kpi_df = pd.DataFrame({"metric": list(kpis.keys()), "value": list(kpis.values())})
    st.download_button(
        "Download KPI summary CSV",
        kpi_df.to_csv(index=False).encode("utf-8"),
        file_name="aidce_sim_kpis.csv",
        mime="text/csv",
    )
    st.download_button(
        "Download configuration JSON",
        json.dumps(asdict(cfg), indent=2).encode("utf-8"),
        file_name="aidce_sim_config.json",
        mime="application/json",
    )
    if "forecast_result" in locals() and forecast_result is not None:
        st.download_button(
            "Download forecast CSV",
            forecast_result.future.reset_index(names="timestamp").to_csv(index=False).encode("utf-8"),
            file_name="aidce_sim_forecast.csv",
            mime="text/csv",
        )
    st.code("streamlit run app.py", language="bash")
