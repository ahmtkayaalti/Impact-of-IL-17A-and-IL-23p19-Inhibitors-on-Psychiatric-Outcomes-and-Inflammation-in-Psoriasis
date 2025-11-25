# Impact-of-IL-17A-and-IL-23p19-Inhibitors-on-Psychiatric-Outcomes-and-Inflammation-in-Psoriasis
# Data and code for IL-17/IL-23 biologics and psychodermatology study

This repository contains the analysis dataset and Python scripts used to
generate Figures 3–4 of the manuscript:

- `GÜNCEL PS EXCEL BİRLEŞİK.xlsx`: anonymized patient-level dataset
  used for the CRP, NLR, PASI, DLQI, HADS-D and HADS-A analyses.
- `figure_scripts.py`: Python script (pandas/numpy/matplotlib/seaborn)
  that reproduces the scatterplots and heatmaps shown in Figures 3–4.

To reproduce the figures:

```bash
python figure_scripts.py