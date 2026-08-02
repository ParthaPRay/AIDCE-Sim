# AIDCE-Sim

**AIDCE-Sim** is an interactive, data-free Python simulator for studying the energy-use patterns of AI data centres. It is designed to support a magazine-style technical article, teaching, scenario exploration, and the generation of synthetic benchmark traces when proprietary facility telemetry is unavailable.

## Main capabilities

- Synthetic workload traces for **model preparation, training, fine-tuning, and inference**.
- Component-level power estimation for GPU, CPU, storage, network, cooling, and auxiliary systems.
- Dynamic PUE affected by IT load and ambient temperature.
- Grid connection capacity, time-of-use tariffs, operational carbon intensity, and WUE-based water estimates.
- Onsite solar and wind generation.
- Rule-based battery dispatch and grid-aware temporal workload shifting.
- Interactive Plotly time series, stage traces, heat maps, load-duration curves, energy breakdowns, and scenario charts.
- Forecast comparison using seasonal naive, Holt-Winters, and lag-based gradient boosting.
- Monte Carlo parameter uncertainty.
- CSV and JSON downloads for full time series, KPIs, forecasts, and configuration.

## Important interpretation

The generated records are **synthetic scenarios**, not measurements. Results should be described as *simulation outputs under stated assumptions*. The tool is suitable for load-pattern studies and sustainability what-if analysis, but it does not replace:

- measured server and facility telemetry;
- hardware-specific benchmark calibration;
- utility interconnection studies;
- electromagnetic transient or harmonic analysis;
- causal claims about a real data centre.

## Installation

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

## Reproducibility

Each simulation is controlled by a fixed random seed. The dashboard can download the complete input configuration as JSON and the generated time series as CSV.

## Core equations

For normalized composite utilization \(u_t\), GPU power is represented as:

\[
P_{GPU,t}=N_{GPU}P_{rated}\left[f_{idle}+(1-f_{idle})u_t\right].
\]

IT power is the sum of GPU, CPU, storage, and network demand. Facility power is:

\[
P_{facility,t}=P_{IT,t}\times PUE_t.
\]

The dynamic PUE term combines a user-specified base PUE, a temperature penalty, and a low-load penalty. Operational grid emissions are:

\[
CO_{2,t}=E_{grid,t}\times CI_{grid}.
\]

Water is estimated using a user-specified WUE applied to IT energy:

\[
Water_t=E_{IT,t}\times WUE.
\]

## Suggested validation plan

1. Calibrate GPU idle and active power against `nvidia-smi` or accelerator telemetry.
2. Calibrate CPU, storage, and network ratios using rack PDU data.
3. Replace synthetic stage profiles with job scheduler logs when available.
4. Fit PUE-temperature coefficients using facility/BMS observations.
5. Replace static tariff and carbon intensity with time-varying regional series.
6. Report sensitivity and uncertainty, not only a single deterministic run.

## Project files

- `app.py` – Streamlit dashboard.
- `simulator.py` – workload, facility, renewable, battery, KPI, scenario, and Monte Carlo engine.
- `forecasting.py` – forecasting and back-testing.
- `example_config.json` – reproducible baseline configuration.
- `article_blueprint.md` – proposed magazine article structure and figure plan.
- `methodology_notes.md` – assumptions, equations, and limitations.

## License suggestion

For public release, add an OSI-approved license such as MIT or BSD-3-Clause after confirming institutional requirements.
