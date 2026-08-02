# AIDCE-Sim

**AIDCE-Sim** is an interactive, data-free Python simulator for studying the energy-use patterns of artificial intelligence (AI) data centres. It supports technical writing, teaching, scenario exploration, sustainability analysis, and the generation of synthetic benchmark traces when proprietary facility telemetry is unavailable.

> **Important:** AIDCE-Sim produces synthetic scenarios rather than measurements from a real data centre. Results should therefore be reported as **simulation outputs under stated assumptions**.

## Key capabilities

- Synthetic workload traces for:
  - model preparation;
  - model training;
  - model fine-tuning; and
  - model inference.
- Component-level power estimation for:
  - GPUs;
  - CPUs;
  - storage;
  - networking;
  - cooling; and
  - auxiliary systems.
- Dynamic Power Usage Effectiveness (PUE) influenced by IT load and ambient temperature.
- Grid-connection-capacity modelling.
- Time-of-use electricity-tariff modelling.
- Operational carbon-emission estimation.
- Water Usage Effectiveness (WUE)-based water-consumption estimation.
- On-site solar and wind generation.
- Rule-based battery dispatch.
- Grid-aware temporal workload shifting.
- Interactive Plotly visualizations, including:
  - time-series plots;
  - workload-stage traces;
  - heat maps;
  - load-duration curves;
  - energy-breakdown charts; and
  - scenario-comparison charts.
- Forecast comparison using:
  - seasonal-naive forecasting;
  - Holt-Winters exponential smoothing; and
  - lag-based gradient boosting.
- Monte Carlo parameter-uncertainty analysis.
- Downloadable CSV and JSON outputs for:
  - full time series;
  - key performance indicators;
  - forecasts;
  - forecast metrics; and
  - simulation configuration.

## Intended uses

AIDCE-Sim is intended for:

- AI data-centre energy-pattern studies;
- sustainability-oriented what-if analysis;
- grid-interaction scenario exploration;
- forecasting experiments;
- classroom demonstrations;
- technical articles and magazine-style publications;
- synthetic benchmark generation; and
- preliminary evaluation of workload-management strategies.

## Interpretation and limitations

The generated records are **synthetic scenarios**, not measurements. AIDCE-Sim is suitable for load-pattern analysis and controlled what-if experiments, but it does not replace:

- measured server, rack, or facility telemetry;
- hardware-specific benchmark calibration;
- building-management-system observations;
- utility interconnection studies;
- electromagnetic-transient studies;
- harmonic or power-quality analysis;
- protection-system studies; or
- causal claims about a real data centre.

When reporting results, use wording such as:

> “Under the stated simulation assumptions, the model produced…”

Avoid presenting synthetic values as observations from an operational facility.

## Project structure

```text
AIDCE-Sim/
├── app.py
├── simulator.py
├── forecasting.py
├── smoke_test.py
├── requirements.txt
├── example_config.json
├── README.md
├── article_blueprint.md
├── methodology_notes.md
├── sample_preview.png
├── sample_forecast.csv
├── sample_forecast_metrics.csv
├── sample_kpis.csv
├── sample_scenario_summary.csv
└── sample_timeseries.csv
```

### Main files

| File | Purpose |
|---|---|
| `app.py` | Streamlit dashboard and interactive user interface |
| `simulator.py` | Workload, facility, renewable-energy, battery, KPI, scenario, and Monte Carlo engine |
| `forecasting.py` | Forecasting, back-testing, and forecast evaluation |
| `smoke_test.py` | Basic functional test of simulation and forecasting modules |
| `requirements.txt` | Python dependencies |
| `example_config.json` | Reproducible baseline configuration |
| `article_blueprint.md` | Proposed technical-article structure and figure plan |
| `methodology_notes.md` | Assumptions, equations, modelling choices, and limitations |

## Installation

### Requirements

- Python 3.10 or later
- `pip`
- A modern web browser

### Windows

Open **Command Prompt** in the project directory and run:

```bat
py -m venv .venv
.venv\Scripts\activate

py -m pip install --upgrade pip setuptools wheel
py -m pip install -r requirements.txt

py smoke_test.py
py -m streamlit run app.py
```

Open the dashboard at:

```text
http://127.0.0.1:8501
```

### Linux or macOS

Open a terminal in the project directory and run:

```bash
python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt

python smoke_test.py
python -m streamlit run app.py
```

Open the dashboard at:

```text
http://127.0.0.1:8501
```

## Google Colab

A robust Colab notebook is included for running AIDCE-Sim in a temporary hosted environment.

The notebook:

- automatically locates the extracted project directory;
- handles Colab's preinstalled `blinker` conflict;
- stops older Streamlit and Cloudflared processes before restarting;
- avoids overwriting a running Cloudflared executable;
- checks the local Streamlit health endpoint;
- verifies the public tunnel URL before displaying it; and
- retries with HTTP/2 when necessary.

Typical Colab workflow:

1. Upload the AIDCE-Sim ZIP archive.
2. Extract the project.
3. Install the required libraries.
4. Run `smoke_test.py`.
5. Start Streamlit.
6. Start a Cloudflare Quick Tunnel.
7. Open the verified temporary dashboard URL.

> **Note:** Cloudflare Quick Tunnel links are temporary and are intended for testing and demonstration. They are not suitable for permanent hosting.

## Quick start

After installation, run:

```bash
python -m streamlit run app.py
```

Then:

1. configure the simulation duration and time resolution;
2. define accelerator and facility parameters;
3. choose workload-stage assumptions;
4. configure PUE, tariff, carbon, water, renewable, and battery settings;
5. run the simulation;
6. inspect the interactive visualizations;
7. compare baseline and intervention scenarios;
8. generate forecasts; and
9. download CSV or JSON outputs.

## Reproducibility

Each simulation is controlled by a fixed random seed. Using the same configuration and random seed produces the same synthetic time series.

The dashboard supports downloading:

- the complete input configuration as JSON;
- the generated time series as CSV;
- KPI summaries as CSV;
- scenario-comparison results as CSV; and
- forecast outputs and evaluation metrics as CSV.

For publication-quality reproducibility, report:

- software version;
- random seed;
- simulation duration;
- temporal resolution;
- hardware assumptions;
- workload-stage assumptions;
- PUE and cooling assumptions;
- grid carbon intensity;
- tariff assumptions;
- WUE;
- renewable-generation settings; and
- battery-dispatch settings.

## Core modelling equations

### GPU power

For normalized composite utilization $u_t$, GPU power is represented as:

```math
P_{\mathrm{GPU},t}
=
N_{\mathrm{GPU}} P_{\mathrm{rated}}
\left[
f_{\mathrm{idle}}
+
\left(1-f_{\mathrm{idle}}\right)u_t
\right].
```

where:

- $N_{\mathrm{GPU}}$ is the number of GPUs;
- $P_{\mathrm{rated}}$ is rated power per GPU;
- $f_{\mathrm{idle}}$ is the idle-power fraction; and
- $u_t$ is normalized utilization at time $t$.

### IT power

Total IT power is estimated as:

```math
P_{\mathrm{IT},t}
=
P_{\mathrm{GPU},t}
+
P_{\mathrm{CPU},t}
+
P_{\mathrm{storage},t}
+
P_{\mathrm{network},t}.
```

### Facility power

Facility power is calculated using dynamic PUE:

```math
P_{\mathrm{facility},t}
=
P_{\mathrm{IT},t}
\times
PUE_t.
```

The dynamic $PUE_t$ term combines:

- a user-defined base PUE;
- an ambient-temperature penalty; and
- a low-load penalty.

### Grid energy and emissions

Operational grid emissions are estimated as:

```math
CO_{2,t}
=
E_{\mathrm{grid},t}
\times
CI_{\mathrm{grid}},
```

where $CI_{\mathrm{grid}}$ is grid carbon intensity.

### Water use

Water use is estimated from IT energy and WUE:

```math
Water_t
=
E_{\mathrm{IT},t}
\times
WUE.
```

These equations are simplified high-level representations and should be calibrated before application to a real facility.

## Forecasting methods

AIDCE-Sim compares three forecasting approaches:

1. **Seasonal naive**  
   Uses the corresponding value from the previous seasonal cycle.

2. **Holt-Winters**  
   Models level, trend, and seasonality using exponential smoothing.

3. **Lag-based gradient boosting**  
   Uses lagged demand and time-derived predictors to model nonlinear load behaviour.

Forecast performance may be evaluated using metrics such as:

- Mean Absolute Error (MAE);
- Root Mean Squared Error (RMSE);
- Mean Absolute Percentage Error (MAPE), where appropriate; and
- coefficient of determination (\(R^2\)), where appropriate.

Forecasts generated from synthetic data should be interpreted as a comparison of modelling behaviour under controlled assumptions, not as evidence of real-world forecasting accuracy.

## Suggested validation plan

1. Calibrate GPU idle and active power against `nvidia-smi` or accelerator telemetry.
2. Calibrate CPU, storage, and network ratios using rack-level PDU observations.
3. Replace synthetic stage profiles with job-scheduler logs when available.
4. Fit PUE-temperature coefficients using facility or building-management-system data.
5. Replace static tariffs with time-varying regional tariff data.
6. Replace static carbon intensity with time-varying regional carbon-intensity data.
7. Calibrate WUE against cooling-system and water-meter observations.
8. Validate battery dispatch against actual efficiency, power, and energy constraints.
9. Report sensitivity and uncertainty rather than only a single deterministic run.
10. Distinguish clearly between calibrated, assumed, and synthetic parameters.

## Sample outputs

The repository includes example outputs generated from an illustrative configuration:

- `sample_timeseries.csv`
- `sample_kpis.csv`
- `sample_scenario_summary.csv`
- `sample_forecast.csv`
- `sample_forecast_metrics.csv`
- `sample_preview.png`

These files demonstrate the expected output format. They should not be treated as measurements from a real AI data centre.

## Future TestPyPI installation

After AIDCE-Sim is converted into a standard Python package, installation from TestPyPI may follow this pattern:

```bash
python -m pip install \
  --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ \
  aidce-sim
```

A future package structure may use:

```text
src/
└── aidce_sim/
    ├── __init__.py
    ├── app.py
    ├── simulator.py
    ├── forecasting.py
    └── cli.py
```

The package may define a console entry point in `pyproject.toml`:

```toml
[project.scripts]
aidce-sim = "aidce_sim.cli:main"
```

After installation, users would launch the application with:

```bash
aidce-sim
```

## Responsible reporting

When using AIDCE-Sim in a manuscript, report:

- all assumptions and parameter values;
- whether each parameter is measured, calibrated, literature-derived, or hypothetical;
- uncertainty ranges;
- sensitivity-analysis procedures;
- random seed;
- software version; and
- limitations of synthetic-data-based inference.

Avoid claims that imply validation against operational data unless such validation has actually been performed.

## Developer

**Partha Pratim Ray**  
Sikkim University, India  
Email: `parthapratimray1986@gmail.com`

## Citation

A formal software citation can be added after the first public release, archival deposit, or DOI assignment.

Suggested temporary citation format:

```text
Ray PP. AIDCE-Sim: An interactive data-free simulator for AI data-centre
energy-use patterns. Version 1.0. Sikkim University; 2026.
```

## License

No open-source license is declared in this draft.

Before public release, add an OSI-approved license such as:

- MIT License.

Confirm institutional intellectual-property and software-release requirements before selecting a license.

## Disclaimer

AIDCE-Sim is provided for research, education, and exploratory analysis. The software does not provide engineering certification, utility approval, operational guarantees, or professional advice for the design or operation of an actual data centre or electric-power system.
