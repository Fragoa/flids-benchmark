# FLIDS reviewer experiments

Import **FLIDS_Review_Fixes.ipynb** into Kaggle. Attach the fresh-run results **as a ZIP** (FLIDS_Master_V3_Fresh_Results.zip). For the historical Edge audit, also attach FLIDS_Experiment01_Results.zip. The notebook embeds its Python modules. It runs offline once inputs and the Kaggle Python dependencies (NumPy, pandas, SciPy, scikit-learn, PyTorch, LightGBM) are available.

## What the code measures

1. **Edge signature replay.** Join Experiment 01's test split, full-feature signatures, target predictions and source-derived thresholds by holdout, seed and row ID. Reproduce archived TP/FP counts at nominal 1% FPR. For DDoS-UDP, compare the identities of detected attacks across methods within each seed. The included CSV covers all five archived core methods over five seeds. Every tested attack had a distinct archived signature within its fold. Four neural methods selected the same 464 of 5,000 attacks within each seed. This excludes repeated identical full signatures as the explanation for the shared 9.28% recall. It does not exclude near-duplicates, correlated captures or partially shared features.
2. **Projected feature overlap.** Count target rows whose raw numeric feature bytes match at least one record in another dataset, before fitting. For the four dataset holdouts combined, the counts are 0 (41 features), 11,041 (remove the two ports), and 11,046 (remove ports plus MIN_TTL and MAX_TTL). These are row counts and measure loss of feature distinctions. They do not establish traffic leakage or capture identity.
3. **Matched, retuned NetFlow ablation.** Compare 39 and 37 feature variants with the existing 41-feature native baseline, checking the digest of the retained 93,810-row cache, the archived training code and all five split-role digests. Keep the original rows after projection. Retune the original four candidates within each source-only outer training problem. Save per-pair target FPR, recall, AUROC, standardized partial AUC, target-informed tail recall, and each change relative to the full-feature model. Projected overlap should accompany interpretation; this experiment holds row membership fixed, not projected uniqueness.

## Execution and recovery

Set TRAIN=False to run audits alone. Research defaults to 11 hours per session and checkpoints each completed trial and scored model. After the budget stop, download FLIDS_Review_Ablation.zip from Kaggle outputs. Attach it alongside the original fresh results for the next session. Only one matching resume ZIP should be attached; the RESUME setting accepts an explicit path. Existing working state takes priority over a mounted checkpoint. Compare paired_results.csv across all 600 rows only when progress.json reports all_pairs_reached=true. An abrupt hard kill before export may require restoring the last saved archive.

The Edge replay and feature-overlap audit were run on the actual archives. A real no-ports logistic-regression fit for BoT-IoT, seed 0, completed its nested tuning and analysis; its checkpoint was restored into a clean directory and reproduced the paired result byte-for-byte. This single fit is a software check, not the full experiment.

## Standalone scripts

Use the notebook for Kaggle, or keep the Python files together and run:

    python edge_ties.py FLIDS_Experiment01_Results.zip edge_replay.csv
    python projection.py FLIDS_Master_V3_Fresh_Results.zip projection_audit.csv
    FLIDS_EXP02_ROOT=./engine_work python ablation.py FLIDS_Master_V3_Fresh_Results.zip ./review_ablation --hours 11

Historical stage 13 is outside this package.
