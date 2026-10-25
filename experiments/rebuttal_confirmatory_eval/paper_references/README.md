# Paper-era scripted references

The scripted agronomic reference numbers in the paper come from two earlier
implementations that the current `agrimanager/rollout/inference/ag_heuristic*`
code replaced under the same label. They are kept here, unchanged, so that
`run_e4_anchors.sh` can score exactly the policies the paper used. Do not use
them as the live baseline.

| File | What it is | Used for | sha256 |
|---|---|---|---|
| `extension_guideline_rollout_paper.py` | per-crop extension-guideline policy (19-crop N caps, legume rule) | WOFOST reference (T1.1, T1.2, T2.x, T3 WOFOST targets) | `e3f8d1b3130b34c10e69e308983d324558ced8485f5423e816e885d60c1f8769` |
| `random_rollout_paper_sim.py` | first implementation of the DSSAT and Cycles heuristics | DSSAT and Cycles references, via `paper_sim_anchor_rollout.py` | `a7136dbfe14f1f0078a09e4e1454b67049dd2d39d489c802a123c1e2c3450674` |

`run_e4_anchors.sh` copies both files into `$OUT_ROOT/_paper_policy/` and
writes their hashes to `PROVENANCE`. The DSSAT reference in the submitted paper
was set by hand; `paper_sim_anchor_rollout.py` reconstructs it and says how
close the reconstruction is.
