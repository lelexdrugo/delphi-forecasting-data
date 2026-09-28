# Paired runtime dataset: which machine meets the deadline?

This archive holds the data behind the article *Which Machine Meets the Deadline? Runtime
Forecasting for Computation Placement in the Cloud–Edge Continuum* by Gabriele Scaffidi Militone,
Daniele Apiletti and Giovanni Malnati (Politecnico di Torino), submitted to *Forecasting* (MDPI).

Every job in the campaign ran on every machine of a four-machine fleet, so the runtime of each
alternative placement is known from measurement. The archive contains:

- the paired dataset;
- the outputs of the confirmatory analysis;
- the two specifications that were frozen before acquisition began;
- the scripts that turn the analysis outputs into the article's tables and figures.

With it, anyone can check the numbers in the article, rebuild its tables and figures, or score a new
forecaster on the same decision moments under the same protocol.

## Contents

```
data/
  dataset/     the paired dataset (704 executions) and its manifests
  analysis/    the 24 artifacts of the confirmatory analysis and their manifest
specs/
  controlled-replay-v4.yaml     acquisition specification, frozen before the first confirmatory job
  controlled-analysis-v4.yaml   analysis contract, frozen before the reduction
generators/
  tables/      make_results.py: the article's result tables and the macros its text uses
  figures/     render_all.py and one script per data figure
LICENSES/      CC BY 4.0 (data, specifications) and MIT (generators)
SHA256SUMS     checksums of every file in the archive
```

## The campaign in brief

**Fleet.** Four single-node Kubernetes clusters in three resource tiers. The operational cluster
names in the data map to tiers as follows:

| cluster | tier | machine | threads |
|---|---|---|---|
| `public-cloud` | large | Intel Xeon E-2136, 3.3 GHz, amd64 | 12 |
| `on-prem` | medium | Intel Xeon E5-1620 v2, 3.7 GHz, amd64 | 8 |
| `edge-1` | small-1 | Raspberry Pi 5, Cortex-A76, 2.4 GHz, arm64 | 4 |
| `edge-2` | small-2 | Raspberry Pi 4, Cortex-A72, 1.8 GHz, arm64 | 4 |

Capability does not follow size. The medium server lacks AVX2, and the Raspberry Pi 5 implements the
Arm integer dot-product instruction, which neither server has.

**Workloads.** Seven parametric families, each chosen for a mechanism that could reorder the
machines:

- `video-transcode`;
- `compress-encrypt`, which compresses, encrypts and hashes;
- `onnx-inference-fp32` and `onnx-inference-int8`, the same networks at full and quantized precision;
- `graph-kernel`;
- `duckdb-tpch`, analytical SQL;
- `stress-ng-cpu`, a synthetic CPU load repeated once per block as a drift monitor.

The 50 distinct workload points are parameter settings of these families. A further family,
`startup-calibration`, runs an empty container to measure start-up overhead.

**Design.** The campaign ran from 2026-09-22 to 2026-09-26. Its 70 forecast origins (decision
moments) fall in six chronological blocks:

- two training blocks of 12 origins each (`block-01`, `block-02`);
- a calibration block of 19 (`block-03`);
- three test blocks of 9 each (`block-04` to `block-06`), not read until the analysis ran.

At each origin one workload point ran on all four machines, with two repetitions per machine, and a
third for quantized inference and compress-encrypt. This gives 656 measured executions, plus 48
start-up calibration runs at 12 further origins, 704 in all.

- **Outcomes:** 703 executions completed; one was interrupted by the driver and is marked censored.
- **Background load:** each origin declares a background regime per machine: `none`,
  `r1-moderate`, `r2-heavy` (memory bandwidth) or `r3-cpu-pressure`. The CPU-pressure regime failed the
  analysis's contention-materialisation gate and was withdrawn as a contention regime; its rows stay
  in the data.
- **Deadlines:** each origin also declares a latency budget, its deadline, at 1.25, 1.75 or 3.00
  times a runtime estimate for the fastest machine, fixed before any outcome existed (tiers `tight`,
  `moderate`, `generous`).

## The dataset

`data/dataset/dataset.csv` has one row per execution: 704 rows and 127 columns. `split` tells the row's
role:

| split | origins | rows |
|---|---|---|
| `train` | 24 | 224 |
| `calibration` | 19 | 176 |
| `test` | 27 | 256 |
| `startup-calibration` | 12 | 48 |

### Key columns

| column | meaning |
|---|---|
| `origin_id`, `block_id`, `forecast_origin_utc` | the decision moment and its block |
| `execution_id`, `job_id`, `repetition` | one execution of one job on one machine |
| `point_id`, `family`, `workload_signature` | the workload point, its family and an immutable signature (image digest and parameters) |
| `candidate_cluster` | the machine the row was measured on (see the table above) |
| `execution_runtime_seconds` | **the label**: start to completion of the Kubernetes Job, in whole seconds |
| `outcome_status`, `completed`, `failure_or_censoring_reason` | how the execution ended |
| `latency_budget_seconds`, `latency_budget_tier` | the deadline declared at the origin |
| `job_deadline_seconds` | the 900 s job timeout, a censoring boundary and not the deadline |
| `completion_time_seconds` | time from the forecast origin to the execution's completion |
| `declared_state_regime` | the background regime the experimenter declared for this machine |
| `primary_size`, `primary_size_descriptor` | the family's main size parameter and its name |
| `params_json` | the full declared parameter vector |

### Feature groups

The analysis used four feature groups, listed column by column in `feature-manifest.json`. They are
available at the forecast origin and could be reproduced by a deployed forecaster.

| group | columns | contents |
|---|---|---|
| W, workload | `descriptor_*`, `log_descriptor_*`, `family`, `primary_size`, `requested_*` | what the job declares before it runs |
| H, history | `prior_*`, `runtime_*` | earlier executions that completed before the origin |
| C, target | `target_*`, `candidate_cluster`, `architecture` | machine descriptors from a revision-pinned registry |
| S, observed state | `observed_*`, minus six excluded columns | a node agent reading `/proc` at 1 Hz on each machine |

S uses CPU utilisation over 15 s and 60 s, disk and network throughput, memory working set, load
average and run-queue length. The analysis contract excludes six columns from it:

- the two temperature readings;
- `observed_cpu_current_ghz`;
- `observed_frequency_ratio_to_nominal`;
- `observed_throttle`;
- `observed_throttled_in_window`.

**Metadata, not features.** Some columns are kept for stratification and validity checks and never
enter a feature group:

- `declared_state_regime`, `role_hint` and `realized_background_population`, which are the
  experimenter's annotations;
- `allocated_*`, the scheduler's allocation totals;
- the `platform_*` fingerprint;
- the `validity_*` fields and the `work_*` fields.

**Missing values** mean the quantity does not apply or did not exist at the origin: a descriptor that a
family does not declare, or no earlier execution of that family on that machine. `work_self_reported_seconds` is empty for every row: the containers'
self-reported work interval was never recorded.

### Manifests

| file | contents |
|---|---|
| `dataset-manifest.json` | row count, dataset gate status and the SHA-256 of each dataset file as produced |
| `split-manifest.json` | origins and rows per split |
| `feature-manifest.json` | the four feature groups and the nested order W, W+H, W+H+C, W+H+C+S |
| `leakage-audit.json`, `dataset-gate.json` | results of the leakage audit and of the dataset gate |
| `provenance-manifest.json` | inputs of the dataset build and their checksums |

## The analysis outputs

`data/analysis/` holds what the confirmatory analysis produced. Its `analysis-manifest.json` records
the analysis code revision, the SHA-256 of the dataset and of both specifications, and the deviations
in force.

| file | contents |
|---|---|
| `point-metrics.json`, `fold-predictions.csv` | point accuracy per model on each split, and every prediction |
| `ranking-per-origin.csv`, `both-axes.json` | ordering and top-1 selection per origin |
| `model-selection.json` | the model selection inside the training blocks |
| `interaction-rank.json` | the registered comparison of the interaction model with the rank-one baseline |
| `scarcity-generalization.json`, `scarcity-predictions.csv` | history budgets and seen or unseen workload points |
| `uncertainty-metrics.json` | conformal prediction intervals |
| `decision-replay.csv`, `decision-replay-metrics.json` | the deadline-constrained placement rules, replayed |
| `statistics.json`, `held-out-membership.json` | paired comparisons and the membership of test rows |
| `contention-materialization.json`, `r3-retention.json` | the contention gate and the withdrawn regime |
| `censoring-accounting.json`, `leakage-checks.json`, `deployment-admissibility.json` | exclusions, leakage checks and the admissibility of every field |
| `clock-skew.json`, `state-coverage.json`, `label-decomposition.json`, `platform-fingerprint.json`, `platform-epoch.json` | validity checks on time alignment, sensing coverage, start-up overhead and platform stability |

## The specifications

The two files in `specs/` are byte-identical to the versions frozen for the campaign:

| file | SHA-256 |
|---|---|
| `controlled-replay-v4.yaml` | `48ccdf6d199180fb0e57ef2435d27aab40b46e9270e38803bcdf739c15b9d234` |
| `controlled-analysis-v4.yaml` | `e019b0e8fac7283780ee8bced18f50c2c1ecd49b618a535945f8ddc0b95397b0` |

The acquisition specification declares every origin, workload point, background regime, repetition
and budget. It references the workload container images by immutable digest; the images themselves
are not part of this archive. The analysis contract declares the splits, feature groups, models,
comparisons, gates and decision rules.

## Rebuilding the tables and figures

The generators need Python 3.10 or later with NumPy, pandas and Matplotlib (tested with Python
3.12.10, NumPy 2.4.6, pandas 3.0.3, Matplotlib 3.10.9). From the root of the archive:

```
pip install -r generators/requirements.txt
python generators/tables/make_results.py --analysis-dir data/analysis --dataset-dir data/dataset --output-dir out/tables
python generators/figures/render_all.py --analysis-dir data/analysis --dataset-dir data/dataset --output-dir out/figures
```

`make_results.py` writes the LaTeX result tables and a file of macros, one per number the article's
text quotes, with `MACROS.md` recording where each number comes from. Apart from their comment headers,
the tables it writes from this archive are identical to the ones in the article. The generators only
read the analysis outputs. Where they recompute a quantity, they check it against the value the
analysis emitted and stop if the two disagree.

## Pseudonymisation

Three columns identified the physical hosts: `platform_machine_id`, `platform_boot_id` and
`validity_agent_node`. They are replaced with labels derived from the cluster name, for example
`machine-public-cloud`, `boot-public-cloud` and `node-public-cloud`. Each machine kept one value of
each throughout the campaign, so the replacement preserves every equality the analysis tests. The
same labels replace the machine identifiers in `data/analysis/platform-fingerprint.json`. Local file
paths in `data/dataset/provenance-manifest.json` are shortened to file names.

No other value was changed. Three files therefore differ from the versions whose SHA-256 values the
manifests and the article record:

| file | SHA-256 of the original |
|---|---|
| `data/dataset/dataset.csv` | `9cef19fd331db98308ac97b8a00b73a056eb0cbde2c662b4a921bf432615c42c` |
| `data/dataset/provenance-manifest.json` | `85a1825d4db807ffacc1828246c08827972c1df357b1242a0a8df0c81934e9e4` |
| `data/analysis/platform-fingerprint.json` | `cfed04ca42032b5e74653ed27a14fdae0f81177fa28da6377fe4e38ba9ebff12` |

All other files are byte-identical to the originals. `SHA256SUMS` lists the checksum of every file
as archived here:

```
sha256sum -c SHA256SUMS
```

## Known limitations of the data

- Runtimes are recorded in whole seconds, so machines that finish within a second of each other tie.
- The work-scoped label subtracts a median start-up overhead per machine and regime. Its planned
  check against the containers' self-reported work interval could not be run.
- There is one machine per tier, so a tier effect is a single-machine effect.
- The contention gate's threshold was revised after the gate first ran. The article reports every
  result that depends on it as exploratory.
- Network-bound workloads are not included.

## How to cite

Please cite the article and this archive. The article reference will be added here once it is
published. The archive's DOI is shown on its Zenodo record; `CITATION.cff` gives the citation
metadata.

## Licence

The data and the specifications (`data/`, `specs/`) are released under the Creative Commons
Attribution 4.0 International licence (`LICENSE`, `LICENSES/CC-BY-4.0.txt`). The generators
(`generators/`) are released under the MIT licence (`LICENSES/MIT.txt`, `generators/LICENSE`).

## Contact

Gabriele Scaffidi Militone, Department of Control and Computer Engineering (DAUIN), Politecnico di
Torino, gabriele.scaffidi@polito.it
