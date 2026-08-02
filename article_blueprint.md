# Proposed IEEE Energy Sustainability Magazine Article

## Recommended title

**AIDCE-Sim: A Data-Free Interactive Framework for Simulating and Forecasting Sustainable AI Data-Centre Loads**

## Alternative titles

1. **From GPU Bursts to Grid Stress: Interactive Simulation of AI Data-Centre Energy Patterns**
2. **What If the Telemetry Is Missing? Synthetic Load Modelling for Sustainable AI Data Centres**
3. **AI Data Centres as Flexible Grid Loads: A Reproducible Simulation and Forecasting Toolkit**

## Central contribution

The article should not be framed as another general review. Its novelty is a **reproducible, transparent, data-free modelling workflow** that enables readers to explore how workload composition, hardware scale, PUE, temperature, tariffs, carbon intensity, renewable generation, batteries, and demand response shape AI data-centre energy patterns.

## Magazine-oriented structure

### 1. The measurement gap
Explain why public, high-resolution AI data-centre telemetry is scarce and why a transparent synthetic tool is useful for planning, education, sensitivity analysis, and hypothesis generation. Distinguish simulation from measurement.

### 2. Why AI loads look different
Describe preparation, training, fine-tuning, and inference. Emphasize sustained training plateaus, communication/checkpoint fluctuations, fine-tuning bursts, and diurnal/stochastic inference.

### 3. AIDCE-Sim architecture
Present a one-page workflow:

**User assumptions → stage workload generators → component power model → PUE/cooling model → renewable/BESS/grid model → KPIs → forecasting and scenario comparison**

### 4. Mathematical model
Keep equations compact and intuitive:

- stage-mixture utilization;
- GPU and non-GPU IT power;
- dynamic PUE and facility power;
- grid energy, cost, carbon, and WUE-based water;
- peak shifting and battery dispatch.

### 5. Four illustrative workload archetypes
Use the same installed capacity but vary workload mix:

| Archetype | Preparation | Training | Fine-tuning | Inference |
|---|---:|---:|---:|---:|
| Training-dominant | 5% | 75% | 10% | 10% |
| Inference-dominant | 5% | 10% | 5% | 80% |
| Balanced | 10% | 35% | 15% | 40% |
| Burst-stress | 5% | 20% | 15% | 60% with higher spike rate |

### 6. Sustainability interventions
Compare:

- baseline;
- lower-PUE cooling;
- GPU power cap;
- grid-aware scheduling;
- onsite renewables + BESS;
- integrated package.

### 7. Forecastability and grid relevance
Back-test seasonal naive, Holt-Winters, and gradient boosting. Discuss why forecasting errors and ramps matter for capacity planning, dispatch, reserves, and demand-response contracting.

### 8. What the simulator can—and cannot—tell us
Be explicit about calibration, uncertainty, and the absence of power-quality/EMT dynamics. Provide a pathway from synthetic traces to telemetry-calibrated digital twins.

### 9. Practical recommendations
Offer recommendations for data-centre operators, utilities, researchers, and policymakers.

## Suggested figures

1. **AIDCE-Sim architecture** – block diagram.
2. **Stage signatures** – normalized preparation/training/fine-tuning/inference traces.
3. **Facility load decomposition** – IT, cooling, auxiliary, and grid load over 48 hours.
4. **Heat map and load-duration curve** – temporal concentration and capacity implications.
5. **Forecast plot** – actual holdout, model comparison, and future interval.
6. **Scenario comparison** – peak, energy, carbon, water, and cost changes.
7. **Uncertainty plot** – Monte Carlo distribution of peak grid load or carbon.

## Suggested tables

1. User-controllable parameters, units, ranges, and interpretation.
2. Equations and output metrics.
3. Workload archetype definitions.
4. Scenario assumptions.
5. Results and percentage change from baseline.
6. Limitations and future extensions.

## Research questions

- RQ1: How does AI workload composition alter peak demand, ramps, load factor, and forecastability?
- RQ2: Which facility and scheduling interventions reduce energy and peak demand most effectively under different workload archetypes?
- RQ3: How do onsite renewables and batteries change grid-energy dependence, curtailment, and operational emissions?
- RQ4: How sensitive are conclusions to uncertain PUE, accelerator power, workload variability, and grid carbon intensity?

## Claims that are safe to make

- The tool generates reproducible synthetic traces under transparent assumptions.
- Different lifecycle stages produce qualitatively distinct load signatures.
- Scenario and uncertainty analysis reveal conditional trade-offs.
- Forecast models can be compared on held-out synthetic observations.

## Claims to avoid without measured validation

- Universal accuracy for real AI data centres.
- Hardware-agnostic prediction of actual energy consumption.
- Causal carbon or water savings for a named facility.
- Grid stability, harmonic, or ride-through compliance.
- Superiority of one forecasting model beyond the tested synthetic scenarios.
