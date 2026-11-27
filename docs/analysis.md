# Reproducing the paper tables and figures

Use a Python environment with the packages in [requirements.txt](../requirements.txt). These are dependency names, not a lockfile. Run these commands from the repository root. Supply your own copy of the completed fresh-run results archive.

```bash
python -m pip install -r requirements.txt
python analysis/export_tables.py /path/to/FLIDS_Master_V3_Fresh_Results.zip
python analysis/tail_calibration.py /path/to/FLIDS_Master_V3_Fresh_Results.zip
python analysis/plot_transfer.py
```

The first command after installation writes five tables in `outputs/`. The second reads those calibration results and the archived scores to render the ROC and target-calibration panels. The last uses the committed native transfer table. To regenerate the earlier figures from expanded Experiment 07 tables, run `python analysis/fixed_schedule_figures.py /path/to/FLIDS_Experiment07_Expanded outputs`.

Rates in the CSV files are fractions. The target-quantile and target-calibrated columns use target information retrospectively. The four holdouts are datasets; the five seeds repeat evaluations on each dataset. Source-derived thresholds do not guarantee the same false-positive rate after transfer.

For the revised manuscript, run the following from the repository root with:

```bash
python analysis/manuscript/analyze.py /path/to/FLIDS_Master_V3_Fresh_Results.zip
python analysis/manuscript/source_check.py /path/to/FLIDS_Master_V3_Fresh_Results.zip /path/to/FLIDS_Experiment01_Results.zip
python analysis/manuscript/operational.py /path/to/FLIDS_Master_V3_Fresh_Results.zip
python analysis/manuscript/plot_operating.py
```
