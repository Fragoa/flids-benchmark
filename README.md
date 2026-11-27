# FLIDS benchmark

Code and reference tables for the federated intrusion-detection benchmark. The primary tuned evaluation holds out each of four NetFlow datasets, with five seeds and 15 methods. The revision reports 300 native evaluations and 1,120 allocation or participation variants. Each policy reuses the scores from its corresponding fitted model.

## Run the primary benchmark on Kaggle

1. Import [the fresh-run notebook](notebooks/current/FLIDS_Master_V3_Fresh.ipynb). Attach the project artifact `FLIDS_Experiment09_Results.zip`, or its extracted dataset with its reproducibility folder. Alternatively, set `EXP09_ZIP` to the exact mounted path of the canonical Experiment 09 ZIP.
2. Set `STAGES = [10, 11, 12]` in the configuration cell for the manuscript scope. The original notebook also offers stage 13. Leave `SMOKE_TEST = False` for research results. For a short software check, set it to True.
3. Run all cells. The notebook validates the input feature order and content digest. Kaggle writes checkpoints and the combined archive under `/kaggle/working`. The 11.25-hour budget allows time to export before a 12-hour session ends. A research run can take multiple sessions.
4. To continue in a new session, attach the latest `FLIDS_Master_V3_Fresh_Results.zip` and set `RESUME_ZIP` to its mounted path. The [historical recovery notebook](notebooks/current/FLIDS_Master_V3.ipynb) is specifically for the earlier V2 checkpoint.

## Experiment map

| Manuscript experiment | Code |
| --- | --- |
| Fixed-schedule experiments and synthesis (1–7) | [Expanded baseline notebook](notebooks/legacy/FLIDS_Master_Experiments01_to07_ExpandedBaselines_Kaggle.ipynb), [stage-specific notebooks](notebooks/legacy/) |
| Earlier Edge attack-family and integrity studies | [Edge notebooks and integrity engine](notebooks/edge/) |
| Historical Experiment 09 input preparation and retuning | [Retuning engine sources](experiments/retuning/), [Experiment 09 notebook](notebooks/legacy/FLIDS_Experiment09_Kaggle_AutoRun.ipynb) |
| Revised manuscript experiments 8, 9 and 10 | Historical stage IDs 10 (source tuning), 11 (allocation), and 12 (threshold policies). [Exact research engine](experiments/tuned/) |
| Reproduction of reported results | [Analysis scripts](analysis/) and [reference tables](reference/) |

The historical notebook filenames retain the experiment numbers used when they were created. Stage 13 is outside the paper's tuned comparison. The earlier Edge attack-family experiments have their own protocol.

## Reproduce the analysis locally

Use a Python environment with the packages in [requirements.txt](requirements.txt). These are dependency names, not a lockfile. Run these commands from the repository root. Supply your own copy of the completed fresh-run results archive.

```bash
python -m pip install -r requirements.txt
python analysis/export_tables.py /path/to/FLIDS_Master_V3_Fresh_Results.zip
python analysis/tail_calibration.py /path/to/FLIDS_Master_V3_Fresh_Results.zip
python analysis/plot_transfer.py
```

The first command after installation writes five tables in `outputs/`. The second reads those calibration results and the archived scores to render the ROC and target-calibration panels. The last uses the committed native transfer table. To regenerate the earlier figures from expanded Experiment 07 tables, run `python analysis/fixed_schedule_figures.py /path/to/FLIDS_Experiment07_Expanded outputs`.

Rates in the CSV files are fractions. The target-quantile and target-calibrated columns use target information retrospectively. The four holdouts are datasets; the five seeds repeat evaluations on each dataset. Source-derived thresholds do not guarantee the same false-positive rate after transfer.

## Inputs and provenance

Raw traffic, prepared input archives, checkpoints, and individual flow scores are supplied separately. [run_identity.json](reference/run_identity.json) records the training parameters and the SHA-256 digests of the research master source and canonical NF input. The three files in `experiments/tuned/` were checked against both the fresh-run notebook and the archived execution. Historical notebooks retain their source cells; saved outputs and execution metadata are cleared.

## Later analyses and code history

| Location | Purpose |
| --- | --- |
| [Original V2 modules and checkpoint identity](experiments/v2/) | Source of the 3,600-trial historical checkpoint, before the fresh-run revision. |
| [Notebook builders](tools/) and [execution checks](checks/) | Construct the recovery, fresh-run and reviewer notebooks and check checkpoint replay. The first two builders write into the working directory; run them in an empty build folder in recovery-then-fresh order. Recovery checks need the original Experiment 09 and Packages 10–13 ZIPs under the user's Downloads directory. |
| [Earlier NetFlow tree benchmark](notebooks/legacy/NF_CrossDomain_FedTrees_Kaggle_AutoRun.ipynb) | Historical cross-dataset experiment, distinct from the primary tuned protocol. |
| [Manuscript analyses](analysis/manuscript/) | Archived-score replay, target ROC and low-FPR tails, source-group checks, paired comparisons, alert scenarios and the tuned operating-point figure. Results are under `tables/`. |
| [Reviewer experiments](review/ablation/README.md) | Self-contained Kaggle notebook, matched no-ports and no-ports/TTL retraining, Edge row/signature replay, projection audit and completed audit tables. The full 600-pair retraining is a separate experiment. |

From the repository root, reproduce the manuscript analyses with:

```bash
python analysis/manuscript/analyze.py /path/to/FLIDS_Master_V3_Fresh_Results.zip
python analysis/manuscript/source_check.py /path/to/FLIDS_Master_V3_Fresh_Results.zip /path/to/FLIDS_Experiment01_Results.zip
python analysis/manuscript/operational.py /path/to/FLIDS_Master_V3_Fresh_Results.zip
python analysis/manuscript/plot_operating.py
```

The reviewer notebook embeds its engine files so it also works as a standalone Kaggle upload. The matching adjacent modules are provided for inspection and command-line use.
