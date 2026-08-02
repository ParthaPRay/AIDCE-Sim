# Methodology Notes for AIDCE-Sim

## 1. Modelling objective

AIDCE-Sim answers a narrow but useful question: **What power, energy, grid, carbon, water, and cost patterns could emerge from a configurable mixture of AI lifecycle workloads when measured data are unavailable?**

The tool uses a bottom-up, transparent synthetic model rather than pretending to infer a universal empirical profile.

## 2. Workload-stage generators

### Preparation
A scheduled, moderate-utilization trace represents data acquisition, cleaning, tokenization, shuffling, and ETL. It includes correlated I/O variability.

### Training
A high-utilization envelope includes ramp-up/ramp-down, communication-related modulation, checkpoint impulses, and configurable duty cycle. This produces sustained demand with rapid fluctuations.

### Fine-tuning
Randomly placed Gaussian bursts represent shorter experiments, validation, and hyperparameter trials. The average is lower than pre-training but more intermittent.

### Inference
A two-peak diurnal curve, weekday/weekend effect, correlated stochastic variation, and random short-lived surges represent user-driven request traffic.

The four shares are normalized internally, so they need not sum to exactly 100 in the UI.

## 3. IT and facility power

GPU demand depends on GPU count, rated power, idle fraction, and composite utilization. CPU, storage, and network power are scaled relative to installed GPU capacity and their stage-sensitive activity signals.

Dynamic PUE is bounded and includes:

- base PUE;
- temperature sensitivity above a reference temperature;
- a low-load efficiency penalty.

Facility overhead is split into cooling and auxiliary power.

## 4. Sustainability model

- Operational carbon uses grid electricity multiplied by user-entered average carbon intensity.
- Water uses WUE in litres per IT kWh.
- Solar follows a daylight sinusoid with correlated cloud variability.
- Wind follows a bounded correlated stochastic capacity factor.
- Renewable output is first consumed onsite; remaining energy can charge the battery, otherwise it is curtailed.

## 5. Flexibility model

Grid-aware scheduling shifts a user-selected fraction of training and fine-tuning utilization out of the configured peak period, subject to off-peak headroom. Energy is approximately conserved within each day.

The battery uses a transparent rule-based controller:

- charge from excess renewable generation;
- optionally charge during low-load off-peak periods;
- discharge above a grid target or during peak-price hours;
- enforce power, energy, efficiency, and state-of-charge limits.

## 6. Forecasting

The dashboard back-tests:

1. Seasonal naive forecast;
2. Additive Holt-Winters forecast;
3. Histogram gradient boosting with lag, rolling mean, time-of-day, day-of-week, and trend features.

The lowest holdout RMSE selects the future forecast. Approximate 95% intervals use holdout residual dispersion and widen with horizon. They are operational uncertainty bands, not formal probabilistic guarantees.

## 7. Scenario presets

- Efficient cooling: lower base PUE and WUE.
- GPU power cap: 15% lower rated GPU power.
- Grid-aware scheduling: 35% shift of flexible peak-period workload.
- Renewables + BESS: scaled onsite renewable and storage capacities.
- Integrated package: combines efficiency, a moderate power cap, scheduling, renewables, and storage.

These are illustrative interventions. Any article must state that results are conditional on scenario assumptions.

## 8. Recommended article experiments

Run at least four workload archetypes:

- training-dominant;
- inference-dominant;
- balanced multi-stage;
- burst-stress case.

For each archetype, compare baseline, efficient cooling, grid-aware scheduling, renewable+BESS, and integrated scenarios. Report energy, peak load, load factor, maximum ramp, capacity exceedance, carbon, water, cost, and forecast error.

## 9. Limitations

The current version does not model:

- accelerator-specific DVFS curves or thermal throttling;
- job completion time and quality-of-service constraints in detail;
- power electronics, harmonics, voltage/frequency ride-through, or EMT dynamics;
- spatial workload migration between regions;
- embodied carbon;
- water scarcity weighting;
- time-varying marginal emissions;
- real market bidding or unit commitment.

These omissions should be positioned as future extensions rather than hidden.
