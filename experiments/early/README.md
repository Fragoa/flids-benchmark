# Running historical stages 01–09

The [stage index](../../README.md) links each research question to its engine. Use the original Kaggle entry points to preserve the archived configurations.

## Stages 01–07

Import the [master notebook](../../notebooks/legacy/FLIDS_Master_Experiments01_to07_ExpandedBaselines_Kaggle.ipynb). For the complete run attach the Edge-IIoTset DNN CSV and NF-UQ-NIDS-v2. Leave `RUN_STAGES = None` to run in protocol order. To select stages, use a list such as `['03', '04']`. Stage 04 replays Stage 03 predictions. Stage 07 reads all six original archives named `FLIDS_ExperimentNN_Results.zip` and produces `FLIDS_Experiment07_Statistical_Synthesis_Results.zip`.

The [dataset and split audit](FLIDS_DatasetSplit_Audit_Engine.py) and [extra baselines for stages 01 and 02](FLIDS_Master_Extra_Baselines.py) have separate entries in the [orchestrator](FLIDS_Master_Orchestrator.py).

## Stages 08–09

The [combined notebook](../../notebooks/legacy/FLIDS_Experiment08_09_Full_Kaggle_AutoRun.ipynb) prepares the NetFlow cache, runs Stage 08, retunes FedProx and checks the operating thresholds. It accepts the combined CSV or all four compatible source CSVs. The [standalone Stage 09 notebook](../../notebooks/legacy/FLIDS_Experiment09_Kaggle_AutoRun.ipynb) is available when the required earlier artifacts are already prepared. The combined Stage 09 output is the input to Stage 10.

## Local verification

From the repository root, verify the 14 engine and support modules against the embedded notebook sources, then run Stage 07 against the six actual result archives:

```bash
python checks/early_sources.py
python experiments/early/FLIDS_Experiment07_Statistical_Synthesis_Engine.py \
  --input-root /path/to/six-result-archives --output-root /path/to/results
```

For a short Stage 08–09 software test, keep the four modules together in a new directory:

```bash
workdir=$(mktemp -d)
cp experiments/retuning/*.py "$workdir/"
python "$workdir/engine09_main.py" --work "$workdir" --smoke
```
