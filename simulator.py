"""Core simulation engine for AIDCE-Sim.

The simulator is intentionally data-free: it generates reproducible synthetic
AI workload traces from transparent assumptions, then converts them to IT,
facility, grid, carbon, water, and cost time series.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SimulationConfig:
    start_date: str = "2026-01-01"
    days: int = 14
    freq_minutes: int = 15
    seed: int = 42

    # Compute hardware
    gpus: int = 8192
    gpu_tdp_w: float = 700.0
    gpu_idle_fraction: float = 0.18
    cpu_to_gpu_power_ratio: float = 0.18
    storage_to_gpu_power_ratio: float = 0.06
    network_to_gpu_power_ratio: float = 0.08

    # Workload mixture (normalized internally)
    preparation_share: float = 0.05
    training_share: float = 0.30
    finetuning_share: float = 0.10
    inference_share: float = 0.55
    training_duty_cycle: float = 0.88
    workload_noise: float = 0.055
    spike_probability: float = 0.012
    spike_magnitude: float = 0.22

    # Facility and environment
    base_pue: float = 1.22
    pue_temperature_sensitivity: float = 0.035
    pue_low_load_penalty: float = 0.055
    cooling_share_of_overhead: float = 0.86
    ambient_mean_c: float = 24.0
    ambient_daily_amplitude_c: float = 7.0
    wue_l_per_it_kwh: float = 0.50

    # Grid, sustainability, and economics
    grid_capacity_mw: float = 15.0
    grid_carbon_kg_per_kwh: float = 0.45
    tariff_offpeak_per_kwh: float = 0.08
    tariff_peak_per_kwh: float = 0.15
    peak_hour_start: int = 17
    peak_hour_end: int = 22
    solar_capacity_mw: float = 0.0
    wind_capacity_mw: float = 0.0

    # Flexibility
    demand_response_shift_fraction: float = 0.0
    battery_power_mw: float = 0.0
    battery_energy_mwh: float = 0.0
    battery_roundtrip_efficiency: float = 0.90
    battery_initial_soc_fraction: float = 0.50
    battery_grid_target_mw: float = 0.0

    def to_dict(self) -> Dict[str, float | int | str]:
        return asdict(self)


DEFAULT_SCENARIO_LABELS = {
    "baseline": "Baseline",
    "efficient_cooling": "Efficient cooling",
    "power_cap": "15% GPU power cap",
    "grid_aware": "Grid-aware scheduling",
    "renewable_bess": "Renewables + BESS",
    "integrated": "Integrated package",
}


def _ar1_noise(n: int, rng: np.random.Generator, phi: float = 0.88) -> np.ndarray:
    eps = rng.normal(0.0, 1.0, n)
    out = np.zeros(n)
    for i in range(1, n):
        out[i] = phi * out[i - 1] + np.sqrt(max(1.0 - phi**2, 0.0)) * eps[i]
    std = out.std()
    return out / std if std > 0 else out


def _gaussian_peak(hour: np.ndarray, center: float, width: float) -> np.ndarray:
    return np.exp(-0.5 * ((hour - center) / width) ** 2)


def _normalize_shares(cfg: SimulationConfig) -> Tuple[float, float, float, float]:
    shares = np.array(
        [
            cfg.preparation_share,
            cfg.training_share,
            cfg.finetuning_share,
            cfg.inference_share,
        ],
        dtype=float,
    )
    if np.any(shares < 0) or shares.sum() <= 0:
        raise ValueError("Workload shares must be non-negative and sum to more than zero.")
    shares /= shares.sum()
    return tuple(float(x) for x in shares)


def _shift_flexible_profile(
    profile: np.ndarray,
    index: pd.DatetimeIndex,
    shift_fraction: float,
    peak_start: int,
    peak_end: int,
) -> np.ndarray:
    """Shift a fraction of flexible utilization from peak to off-peak periods.

    Energy is approximately conserved for each day, subject to a utilization cap
    of 1.0 and available off-peak headroom.
    """
    if shift_fraction <= 0:
        return profile.copy()

    shifted = profile.copy()
    day_keys = index.normalize()
    for day in pd.unique(day_keys):
        day_mask = day_keys == day
        hours = index[day_mask].hour
        peak = (hours >= peak_start) & (hours < peak_end)
        offpeak = ~peak
        day_vals = shifted[day_mask].copy()
        removed_vector = day_vals[peak] * min(shift_fraction, 0.95)
        removed = float(removed_vector.sum())
        day_vals[peak] -= removed_vector

        if removed > 0 and offpeak.any():
            headroom = np.clip(1.0 - day_vals[offpeak], 0.0, None)
            headroom_sum = float(headroom.sum())
            if headroom_sum > 0:
                addition = removed * headroom / headroom_sum
                day_vals[offpeak] = np.minimum(1.0, day_vals[offpeak] + addition)
        shifted[day_mask] = day_vals
    return shifted


def generate_workload_profiles(
    index: pd.DatetimeIndex, cfg: SimulationConfig
) -> pd.DataFrame:
    """Generate normalized (0-1) utilization traces for four AI lifecycle stages."""
    n = len(index)
    rng = np.random.default_rng(cfg.seed)
    points_per_day = int(round(24 * 60 / cfg.freq_minutes))
    t = np.arange(n, dtype=float)
    hour = index.hour.to_numpy() + index.minute.to_numpy() / 60.0
    dow = index.dayofweek.to_numpy()

    # Preparation / ETL: scheduled daytime blocks plus intermittent I/O bursts.
    business = ((hour >= 7) & (hour <= 21)).astype(float)
    prep_cycle = 0.18 + 0.42 * business + 0.12 * np.sin(2 * np.pi * t / max(points_per_day / 3, 1))
    prep_noise = 0.10 * _ar1_noise(n, rng, 0.80)
    prep = np.clip(prep_cycle + prep_noise, 0.04, 1.0)

    # Training: near-sustained high utilization with ramp, communication dips,
    # checkpoint spikes, and configurable job duty cycle.
    job_period = max(points_per_day * 2, 8)
    phase = (t % job_period) / job_period
    active = phase < cfg.training_duty_cycle
    ramp_fraction = min(0.05, cfg.training_duty_cycle / 4)
    ramp = np.clip(phase / max(ramp_fraction, 1e-6), 0.0, 1.0)
    ramp_down = np.clip((cfg.training_duty_cycle - phase) / max(ramp_fraction, 1e-6), 0.0, 1.0)
    envelope = active.astype(float) * np.minimum(ramp, ramp_down)
    comm_period = max(int(round(10 / cfg.freq_minutes)), 2)
    communication = 0.07 * (0.5 + 0.5 * np.sin(2 * np.pi * t / comm_period))
    checkpoint_period = max(int(round(6 * 60 / cfg.freq_minutes)), 2)
    checkpoint = 0.10 * ((t.astype(int) % checkpoint_period) == 0)
    training = envelope * (0.80 + 0.10 * _ar1_noise(n, rng, 0.68) - communication + checkpoint)
    training += (~active).astype(float) * 0.05
    training = np.clip(training, 0.03, 1.0)

    # Fine-tuning: intermittent experiments and validation bursts.
    finetune = np.full(n, 0.08, dtype=float)
    expected_bursts = max(4, int(cfg.days * 2.5))
    centers = rng.integers(0, max(n, 1), size=expected_bursts)
    widths = rng.integers(
        max(2, int(30 / cfg.freq_minutes)),
        max(3, int(4 * 60 / cfg.freq_minutes)),
        size=expected_bursts,
    )
    amplitudes = rng.uniform(0.35, 0.85, size=expected_bursts)
    for center, width, amplitude in zip(centers, widths, amplitudes):
        finetune += amplitude * np.exp(-0.5 * ((t - center) / max(width, 1)) ** 2)
    finetune += 0.07 * _ar1_noise(n, rng, 0.55)
    finetune = np.clip(finetune, 0.03, 1.0)

    # Inference: diurnal user demand, weekday effects, correlated noise, and spikes.
    morning = _gaussian_peak(hour, 10.5, 2.8)
    evening = _gaussian_peak(hour, 19.5, 3.2)
    overnight = _gaussian_peak(hour, 2.0, 3.0)
    weekday_factor = np.where(dow < 5, 1.0, 0.84)
    inference = (0.16 + 0.40 * morning + 0.52 * evening + 0.10 * overnight) * weekday_factor
    inference += 0.06 * _ar1_noise(n, rng, 0.92)
    spike_mask = rng.random(n) < cfg.spike_probability
    spike_impulse = spike_mask.astype(float) * rng.uniform(0.5, 1.0, n) * cfg.spike_magnitude
    # Short decay makes query surges last more than a single timestep.
    kernel_len = max(2, int(round(45 / cfg.freq_minutes)))
    kernel = np.exp(-np.arange(kernel_len) / max(kernel_len / 3, 1))
    spike_series = np.convolve(spike_impulse, kernel, mode="full")[:n]
    inference = np.clip(inference + spike_series, 0.04, 1.0)

    # Apply workload flexibility to the stages most amenable to temporal shifting.
    training = _shift_flexible_profile(
        training,
        index,
        cfg.demand_response_shift_fraction,
        cfg.peak_hour_start,
        cfg.peak_hour_end,
    )
    finetune = _shift_flexible_profile(
        finetune,
        index,
        cfg.demand_response_shift_fraction,
        cfg.peak_hour_start,
        cfg.peak_hour_end,
    )

    return pd.DataFrame(
        {
            "preparation_utilization": prep,
            "training_utilization": training,
            "finetuning_utilization": finetune,
            "inference_utilization": inference,
        },
        index=index,
    )


def _simulate_renewables(
    index: pd.DatetimeIndex, cfg: SimulationConfig, rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray]:
    n = len(index)
    hour = index.hour.to_numpy() + index.minute.to_numpy() / 60.0
    solar_shape = np.where(
        (hour >= 6) & (hour <= 18),
        np.sin(np.pi * (hour - 6) / 12),
        0.0,
    )
    cloud = np.clip(0.86 + 0.12 * _ar1_noise(n, rng, 0.96), 0.35, 1.05)
    solar = cfg.solar_capacity_mw * solar_shape * cloud

    wind_factor = np.clip(
        0.42
        + 0.13 * _ar1_noise(n, rng, 0.97)
        + 0.05 * np.sin(2 * np.pi * np.arange(n) / max(7 * 24 * 60 / cfg.freq_minutes, 1)),
        0.03,
        0.92,
    )
    wind = cfg.wind_capacity_mw * wind_factor
    return solar, wind


def _dispatch_battery(
    facility_mw: np.ndarray,
    renewable_mw: np.ndarray,
    tariff: np.ndarray,
    index: pd.DatetimeIndex,
    cfg: SimulationConfig,
) -> pd.DataFrame:
    n = len(index)
    dt_h = cfg.freq_minutes / 60.0
    power = max(cfg.battery_power_mw, 0.0)
    energy = max(cfg.battery_energy_mwh, 0.0)
    eta_rt = float(np.clip(cfg.battery_roundtrip_efficiency, 0.50, 1.0))
    eta_c = np.sqrt(eta_rt)
    eta_d = np.sqrt(eta_rt)
    target = (
        cfg.battery_grid_target_mw
        if cfg.battery_grid_target_mw > 0
        else 0.85 * max(cfg.grid_capacity_mw, 0.001)
    )

    renewable_to_load = np.minimum(renewable_mw, facility_mw)
    excess_renewable = np.maximum(renewable_mw - facility_mw, 0.0)
    net_load = np.maximum(facility_mw - renewable_to_load, 0.0)

    charge = np.zeros(n)
    discharge = np.zeros(n)
    soc = np.zeros(n)
    curtailed = np.zeros(n)
    grid = np.zeros(n)
    stored = energy * float(np.clip(cfg.battery_initial_soc_fraction, 0.0, 1.0))

    for i in range(n):
        local_hour = index[i].hour
        is_peak = cfg.peak_hour_start <= local_hour < cfg.peak_hour_end
        available_charge_mw = 0.0
        discharge_request_mw = 0.0

        if power > 0 and energy > 0:
            # Use excess onsite renewable first.
            available_charge_mw = min(power, excess_renewable[i])
            # Opportunistic off-peak grid charging below a conservative threshold.
            if not is_peak and net_load[i] < 0.62 * target:
                available_charge_mw = min(power, max(available_charge_mw, 0.62 * target - net_load[i]))
            # Peak shaving or tariff-responsive discharge.
            if net_load[i] > target:
                discharge_request_mw = min(power, net_load[i] - target)
            elif is_peak and stored > 0.12 * energy:
                discharge_request_mw = min(power, max(0.0, net_load[i] - 0.72 * target))

            max_charge_from_space = max((energy - stored) / max(eta_c * dt_h, 1e-9), 0.0)
            c = min(available_charge_mw, max_charge_from_space)
            max_discharge_from_soc = max(stored * eta_d / max(dt_h, 1e-9), 0.0)
            d = min(discharge_request_mw, max_discharge_from_soc)

            # Avoid simultaneous charge and discharge.
            if d > 0:
                c = 0.0
            stored += c * eta_c * dt_h
            stored -= d / max(eta_d, 1e-9) * dt_h
            stored = float(np.clip(stored, 0.0, energy))
            charge[i], discharge[i] = c, d

        excess_used_for_charge = min(charge[i], excess_renewable[i])
        grid_charge = max(charge[i] - excess_used_for_charge, 0.0)
        curtailed[i] = max(excess_renewable[i] - excess_used_for_charge, 0.0)
        grid[i] = max(net_load[i] + grid_charge - discharge[i], 0.0)
        soc[i] = stored

    return pd.DataFrame(
        {
            "renewable_to_load_mw": renewable_to_load,
            "renewable_curtailed_mw": curtailed,
            "battery_charge_mw": charge,
            "battery_discharge_mw": discharge,
            "battery_soc_mwh": soc,
            "grid_power_mw": grid,
        },
        index=index,
    )


def simulate_ai_data_center(cfg: SimulationConfig) -> pd.DataFrame:
    """Run one deterministic simulation for the supplied configuration."""
    if cfg.days <= 0:
        raise ValueError("days must be positive")
    if cfg.freq_minutes not in {1, 5, 10, 15, 30, 60}:
        raise ValueError("freq_minutes must be one of 1, 5, 10, 15, 30, 60")
    if cfg.gpus <= 0 or cfg.gpu_tdp_w <= 0:
        raise ValueError("GPU count and GPU TDP must be positive")
    if cfg.base_pue < 1.0:
        raise ValueError("PUE cannot be below 1.0")

    periods = int(round(cfg.days * 24 * 60 / cfg.freq_minutes))
    index = pd.date_range(
        start=pd.Timestamp(cfg.start_date),
        periods=periods,
        freq=f"{cfg.freq_minutes}min",
    )
    rng = np.random.default_rng(cfg.seed + 1009)
    profiles = generate_workload_profiles(index, cfg)
    p_share, t_share, f_share, i_share = _normalize_shares(cfg)

    composite = (
        p_share * profiles["preparation_utilization"].to_numpy()
        + t_share * profiles["training_utilization"].to_numpy()
        + f_share * profiles["finetuning_utilization"].to_numpy()
        + i_share * profiles["inference_utilization"].to_numpy()
    )
    composite += cfg.workload_noise * _ar1_noise(periods, rng, 0.78)
    composite = np.clip(composite, 0.02, 1.0)

    gpu_capacity_mw = cfg.gpus * cfg.gpu_tdp_w / 1_000_000.0
    gpu_power = gpu_capacity_mw * (
        cfg.gpu_idle_fraction + (1.0 - cfg.gpu_idle_fraction) * composite
    )

    cpu_power = (
        gpu_capacity_mw
        * cfg.cpu_to_gpu_power_ratio
        * (0.42 + 0.58 * composite)
    )
    storage_activity = np.clip(
        0.50 * profiles["preparation_utilization"].to_numpy()
        + 0.18 * profiles["training_utilization"].to_numpy()
        + 0.12 * profiles["finetuning_utilization"].to_numpy()
        + 0.20 * profiles["inference_utilization"].to_numpy(),
        0.0,
        1.0,
    )
    storage_power = (
        gpu_capacity_mw
        * cfg.storage_to_gpu_power_ratio
        * (0.52 + 0.48 * storage_activity)
    )
    network_activity = np.clip(
        0.58 * profiles["training_utilization"].to_numpy()
        + 0.20 * profiles["finetuning_utilization"].to_numpy()
        + 0.22 * profiles["inference_utilization"].to_numpy(),
        0.0,
        1.0,
    )
    network_power = (
        gpu_capacity_mw
        * cfg.network_to_gpu_power_ratio
        * (0.38 + 0.62 * network_activity)
    )
    it_power = gpu_power + cpu_power + storage_power + network_power

    hour = index.hour.to_numpy() + index.minute.to_numpy() / 60.0
    ambient = (
        cfg.ambient_mean_c
        + cfg.ambient_daily_amplitude_c * np.sin(2 * np.pi * (hour - 9.0) / 24.0)
        + 0.8 * _ar1_noise(periods, rng, 0.95)
    )
    it_load_fraction = it_power / max(float(np.nanmax(it_power)), 1e-9)
    dynamic_pue = (
        cfg.base_pue
        + cfg.pue_temperature_sensitivity * np.maximum(ambient - 22.0, 0.0) / 10.0
        + cfg.pue_low_load_penalty * (1.0 - it_load_fraction) ** 2
    )
    dynamic_pue = np.clip(dynamic_pue, 1.02, 2.50)
    facility_power = it_power * dynamic_pue
    overhead = facility_power - it_power
    cooling_power = overhead * float(np.clip(cfg.cooling_share_of_overhead, 0.0, 1.0))
    auxiliary_power = overhead - cooling_power

    solar, wind = _simulate_renewables(index, cfg, rng)
    renewable = solar + wind
    peak_mask = (index.hour >= cfg.peak_hour_start) & (index.hour < cfg.peak_hour_end)
    tariff = np.where(
        peak_mask,
        cfg.tariff_peak_per_kwh,
        cfg.tariff_offpeak_per_kwh,
    )
    dispatch = _dispatch_battery(facility_power, renewable, tariff, index, cfg)

    dt_h = cfg.freq_minutes / 60.0
    grid_power = dispatch["grid_power_mw"].to_numpy()
    interval_grid_kwh = grid_power * 1000.0 * dt_h
    interval_it_kwh = it_power * 1000.0 * dt_h
    interval_facility_kwh = facility_power * 1000.0 * dt_h

    result = profiles.copy()
    result["composite_utilization"] = composite
    result["ambient_temperature_c"] = ambient
    result["gpu_power_mw"] = gpu_power
    result["cpu_power_mw"] = cpu_power
    result["storage_power_mw"] = storage_power
    result["network_power_mw"] = network_power
    result["it_power_mw"] = it_power
    result["dynamic_pue"] = dynamic_pue
    result["cooling_power_mw"] = cooling_power
    result["auxiliary_power_mw"] = auxiliary_power
    result["facility_power_mw"] = facility_power
    result["solar_generation_mw"] = solar
    result["wind_generation_mw"] = wind
    result["renewable_generation_mw"] = renewable
    result = result.join(dispatch)
    result["tariff_per_kwh"] = tariff
    result["grid_capacity_mw"] = cfg.grid_capacity_mw
    result["grid_capacity_exceeded"] = result["grid_power_mw"] > cfg.grid_capacity_mw
    result["it_energy_kwh"] = interval_it_kwh
    result["facility_energy_kwh"] = interval_facility_kwh
    result["grid_energy_kwh"] = interval_grid_kwh
    result["cost"] = interval_grid_kwh * tariff
    result["carbon_kg"] = interval_grid_kwh * cfg.grid_carbon_kg_per_kwh
    # WUE is represented as site water per unit of IT energy.
    result["water_l"] = interval_it_kwh * cfg.wue_l_per_it_kwh
    result["timestamp"] = result.index
    return result


def summarize_kpis(df: pd.DataFrame, cfg: SimulationConfig) -> Dict[str, float]:
    dt_h = cfg.freq_minutes / 60.0
    facility_energy_mwh = float(df["facility_energy_kwh"].sum() / 1000.0)
    it_energy_mwh = float(df["it_energy_kwh"].sum() / 1000.0)
    grid_energy_mwh = float(df["grid_energy_kwh"].sum() / 1000.0)
    renewable_used_mwh = float(df["renewable_to_load_mw"].sum() * dt_h)
    peak_grid = float(df["grid_power_mw"].max())
    mean_grid = float(df["grid_power_mw"].mean())
    ramp = df["grid_power_mw"].diff().abs() / max(cfg.freq_minutes, 1)
    battery_throughput = float(
        (df["battery_charge_mw"].sum() + df["battery_discharge_mw"].sum()) * dt_h
    )
    cycles = (
        battery_throughput / (2.0 * cfg.battery_energy_mwh)
        if cfg.battery_energy_mwh > 0
        else 0.0
    )
    return {
        "IT energy (MWh)": it_energy_mwh,
        "Facility energy (MWh)": facility_energy_mwh,
        "Grid energy (MWh)": grid_energy_mwh,
        "Peak facility load (MW)": float(df["facility_power_mw"].max()),
        "Peak grid load (MW)": peak_grid,
        "Average grid load (MW)": mean_grid,
        "Grid load factor": mean_grid / peak_grid if peak_grid > 0 else 0.0,
        "Energy-weighted PUE": facility_energy_mwh / it_energy_mwh if it_energy_mwh > 0 else np.nan,
        "Renewable energy used (MWh)": renewable_used_mwh,
        "Renewable coverage (%)": 100.0 * renewable_used_mwh / facility_energy_mwh if facility_energy_mwh > 0 else 0.0,
        "Carbon emissions (tCO2e)": float(df["carbon_kg"].sum() / 1000.0),
        "Water consumption (m3)": float(df["water_l"].sum() / 1000.0),
        "Electricity cost": float(df["cost"].sum()),
        "Capacity exceedance (hours)": float(df["grid_capacity_exceeded"].sum() * dt_h),
        "Maximum ramp (MW/min)": float(ramp.max(skipna=True) if len(ramp) else 0.0),
        "95th percentile grid load (MW)": float(df["grid_power_mw"].quantile(0.95)),
        "Battery equivalent cycles": cycles,
    }


def scenario_configurations(cfg: SimulationConfig) -> Dict[str, SimulationConfig]:
    """Return transparent, publication-ready intervention scenarios."""
    approx_peak = max(cfg.grid_capacity_mw, 1.0)
    return {
        "baseline": cfg,
        "efficient_cooling": replace(
            cfg,
            base_pue=max(1.05, cfg.base_pue - 0.10),
            wue_l_per_it_kwh=max(0.0, cfg.wue_l_per_it_kwh * 0.75),
        ),
        "power_cap": replace(cfg, gpu_tdp_w=cfg.gpu_tdp_w * 0.85),
        "grid_aware": replace(
            cfg,
            demand_response_shift_fraction=max(cfg.demand_response_shift_fraction, 0.35),
        ),
        "renewable_bess": replace(
            cfg,
            solar_capacity_mw=max(cfg.solar_capacity_mw, 0.28 * approx_peak),
            wind_capacity_mw=max(cfg.wind_capacity_mw, 0.12 * approx_peak),
            battery_power_mw=max(cfg.battery_power_mw, 0.18 * approx_peak),
            battery_energy_mwh=max(cfg.battery_energy_mwh, 0.72 * approx_peak),
        ),
        "integrated": replace(
            cfg,
            base_pue=max(1.05, cfg.base_pue - 0.10),
            wue_l_per_it_kwh=max(0.0, cfg.wue_l_per_it_kwh * 0.75),
            gpu_tdp_w=cfg.gpu_tdp_w * 0.90,
            demand_response_shift_fraction=max(cfg.demand_response_shift_fraction, 0.35),
            solar_capacity_mw=max(cfg.solar_capacity_mw, 0.28 * approx_peak),
            wind_capacity_mw=max(cfg.wind_capacity_mw, 0.12 * approx_peak),
            battery_power_mw=max(cfg.battery_power_mw, 0.18 * approx_peak),
            battery_energy_mwh=max(cfg.battery_energy_mwh, 0.72 * approx_peak),
        ),
    }


def compare_scenarios(
    cfg: SimulationConfig,
    scenario_keys: Iterable[str] | None = None,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    configs = scenario_configurations(cfg)
    keys = list(scenario_keys) if scenario_keys is not None else list(configs)
    rows = []
    outputs: Dict[str, pd.DataFrame] = {}
    for key in keys:
        if key not in configs:
            continue
        sim = simulate_ai_data_center(configs[key])
        outputs[key] = sim
        kpis = summarize_kpis(sim, configs[key])
        kpis["Scenario"] = DEFAULT_SCENARIO_LABELS.get(key, key)
        rows.append(kpis)
    summary = pd.DataFrame(rows).set_index("Scenario") if rows else pd.DataFrame()
    return summary, outputs


def monte_carlo_summary(
    cfg: SimulationConfig,
    runs: int = 50,
    pue_sigma: float = 0.025,
    utilization_sigma: float = 0.06,
    carbon_sigma: float = 0.05,
) -> pd.DataFrame:
    """Run lightweight parameter uncertainty analysis and return KPI samples."""
    runs = int(np.clip(runs, 5, 500))
    rng = np.random.default_rng(cfg.seed + 777)
    rows = []
    for run in range(runs):
        perturbed = replace(
            cfg,
            seed=cfg.seed + run + 1,
            base_pue=max(1.02, rng.normal(cfg.base_pue, pue_sigma)),
            gpu_tdp_w=max(50.0, cfg.gpu_tdp_w * rng.normal(1.0, utilization_sigma)),
            grid_carbon_kg_per_kwh=max(
                0.0, rng.normal(cfg.grid_carbon_kg_per_kwh, carbon_sigma)
            ),
        )
        sim = simulate_ai_data_center(perturbed)
        k = summarize_kpis(sim, perturbed)
        rows.append(
            {
                "run": run + 1,
                "facility_energy_mwh": k["Facility energy (MWh)"],
                "peak_grid_mw": k["Peak grid load (MW)"],
                "carbon_tco2e": k["Carbon emissions (tCO2e)"],
                "cost": k["Electricity cost"],
                "pue": k["Energy-weighted PUE"],
            }
        )
    return pd.DataFrame(rows)
