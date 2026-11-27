# Running tuned stages 10–12

1. Import the [fresh-run notebook](../../notebooks/current/FLIDS_Master_V3_Fresh.ipynb) into Kaggle. Attach `FLIDS_Experiment09_Results.zip` or its extracted dataset containing the reproducibility folder. For a canonical Experiment 09 ZIP with a different filename, set `EXP09_ZIP` to its mounted path.
2. Set `STAGES = [10, 11, 12]` and leave `SMOKE_TEST = False` for research runs. These stages require the NF input only. The notebook also contains an optional Stage 13 outside the twelve-stage benchmark.
3. Run all cells. [Preflight](preflight.py) verifies the input; the [shared engine](master.py) dispatches each stage. Stage 10 selects native configurations, Stage 11 applies them to allocation and participation variants, and Stage 12 runs threshold analyses with saved scores. Completed bundles and analyses can be reused.
4. Download `FLIDS_Master_V3_Fresh_Results.zip` from Kaggle outputs. If the run pauses, attach that archive and the Stage 09 input in the next session and set `RESUME_ZIP` to the mounted results ZIP. The default budget reserves 45 minutes in a twelve-hour session for finishing and export.

The [run identity](../../reference/run_identity.json) records the historical input digest, master source digest and training settings. Use the [analysis guide](../../docs/analysis.md) to replay the paper results.
