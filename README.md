# FLIDS: federated intrusion detection across datasets

This repository contains the twelve research stages of the FLIDS benchmark. The early experiments test attack-family and dataset holdouts. The later experiments tune models using source data and test whether their thresholds still meet a false-alarm budget on a held-out dataset.

## Stages 01–12

Stage numbers below match the code and result archives. In the revised manuscript, experiments 8, 9 and 10 correspond to repository stages **10, 11 and 12**.

| Stage | Research question | Implementation | Main result |
| --- | --- | --- | --- |
| 01 | Can detectors recognize an attack family absent from training? | [Edge family holdouts](experiments/early/FLIDS_Experiment01_Engine_v1_1.py) | Family-level recall and predictions |
| 02 | Does a detector transfer to a held-out NetFlow dataset? | [NetFlow dataset holdouts](experiments/early/FLIDS_Experiment02_Engine.py) | Four dataset holdouts with fixed training schedules |
| 03 | How do source thresholds, calibration and feature choices affect transfer? | [Threshold and feature audit](experiments/early/FLIDS_Experiment03_Engine.py) | Saved scores and calibration diagnostics |
| 04 | What alert burden follows from the Stage 03 scores? | [Prevalence and alert-budget replay](experiments/early/FLIDS_Experiment04_Engine.py) | Projected precision and alerts under assumed prevalence |
| 05 | How sensitive is training to client composition and partial participation? | [Client heterogeneity](experiments/early/FLIDS_Experiment05_Engine.py) | Participation and allocation comparisons |
| 06 | Which allocation changes account for those differences? | [Controlled allocation experiment](experiments/early/FLIDS_Experiment06_Controlled_Heterogeneity_Engine.py) | Paired contrasts with a mixed control |
| 07 | How variable are the results across holdouts? | [Statistical synthesis](experiments/early/FLIDS_Experiment07_Statistical_Synthesis_Engine.py) | Synthesis of the six preceding archives |
| 08 | How do federated optimizers compare after source-only candidate selection? | [FedOpt tuning](experiments/retuning/engine08_original.py) | Selected models, scores and learning curves |
| 09 | Does retuning FedProx change the comparison? | [FedProx retuning](experiments/retuning/engine09_retune.py) and [threshold checks](experiments/retuning/engine09_main.py) | Paired before/after results and reproducibility inputs |
| 10 | How well do 15 tuned methods transfer under native allocation? | [Source-only tuning](experiments/tuned/master.py) | 300 native model evaluations across four datasets and five seeds |
| 11 | What changes when allocation or participation changes? | [Robustness scenarios](experiments/tuned/master.py) | 1,120 additional matched evaluations with native-selected configurations |
| 12 | Which source-derived thresholds meet the target budget? | [Threshold-policy evaluation](experiments/tuned/master.py) | Five policies evaluated on the same model scores |

## Choose a run

| Stages | Kaggle entry point | Inputs and settings |
| --- | --- | --- |
| 01–07 | [Fixed-schedule benchmark](notebooks/legacy/FLIDS_Master_Experiments01_to07_ExpandedBaselines_Kaggle.ipynb) | Attach the official Edge-IIoTset DNN CSV and NF-UQ-NIDS-v2 for a full run. Set `RUN_STAGES` to select a subset. See the [early-stage run guide](experiments/early/README.md) for dependencies. |
| 08–09 | [Combined retuning notebook](notebooks/legacy/FLIDS_Experiment08_09_Full_Kaggle_AutoRun.ipynb) | Attach the combined NF-UQ-NIDS-v2 CSV or the four compatible component CSVs. Stage 08 feeds Stage 09. |
| 10–12 | [Fresh-run notebook](notebooks/current/FLIDS_Master_V3_Fresh.ipynb) | Attach the Stage 09 results ZIP or its extracted dataset, then set `STAGES = [10, 11, 12]`. The [tuned-stage run guide](experiments/tuned/README.md) covers input checks and session recovery. |

## Results and supporting code

- [Reproduce the paper tables and figures](docs/analysis.md) from a completed results archive. [Reference tables](reference/) and [manuscript analyses](analysis/manuscript/) are included.
- [Reviewer follow-up experiments](review/ablation/README.md) cover matched feature ablations, projected feature overlap and Edge signature replay.
- [Historical V2 engine](experiments/v2/) and [recovery notebook](notebooks/current/FLIDS_Master_V3.ipynb) support the earlier checkpoint. [Notebook builders](tools/) and [execution checks](checks/) are grouped separately.
- [Edge development notebooks](notebooks/edge/) and [historical alternatives](notebooks/legacy/) retain their original filenames for traceability.

Research inputs and large score archives are supplied separately. The four held-out domains are datasets; seeds repeat evaluations within each dataset. Reported alert volumes depend on assumed attack prevalence.
