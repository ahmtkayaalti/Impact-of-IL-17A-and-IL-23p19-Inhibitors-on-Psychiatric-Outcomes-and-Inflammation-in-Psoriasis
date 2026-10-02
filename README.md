# Impact of IL-17A and IL-23p19 Inhibitors on Psychiatric Outcomes and Inflammation in Psoriasis

Data, analysis code and figures for:

> Altunay IK, Demir A, Kayaaltı A, et al. Impact of IL-17A and IL-23p19 Inhibitors on Psychiatric Outcomes and Inflammation in Psoriasis. *JDDG: Journal der Deutschen Dermatologischen Gesellschaft*. 2026. https://doi.org/10.1111/ddg.70579

Multicenter prospective study, 235 biologic-naïve patients with moderate-to-severe plaque psoriasis, assessed at baseline and after 6 months of treatment with secukinumab, ixekizumab, risankizumab or guselkumab.

## Repository contents

| Folder | File | Description |
|---|---|---|
| `data/` | `psoriasis_data_anonymized.csv` | De-identified dataset (n = 235). Names and dates removed, row order randomised. |
| `analysis/` | `analysis.py` | Reproduces all values reported in Tables 1–5, Figure 4 and the Results text. Output: `results.txt`. |
| `analysis/` | `analysis_spss.sps` | SPSS syntax for the same analyses (IBM SPSS Statistics v27). |
| `figures/scripts/` | `figure1_…`, `figure2_…`, `figure3_…` | Scripts for Figures 1–3 (600 dpi TIFF and vector PDF, English and German). |
| `figures/output/` | `*.tif`, `*.pdf` | Final figure files as published. |

## How to reproduce

Requirements: Python ≥ 3.10 with `pandas`, `numpy`, `scipy`, `matplotlib`.

```bash
cd analysis && python analysis.py              # writes results.txt
cd ../figures/scripts && python figure2_boxplots.py   # etc.
```

For SPSS: open `analysis/analysis_spss.sps`, change the file path in the `FILE=` line, and run all.

## Codebook (`psoriasis_data_anonymized.csv`)

| Variable | Description | Coding |
|---|---|---|
| `id` | Study identifier (random) | 1–235 |
| `age` | Age, years | |
| `sex` | Sex | 1 = female, 2 = male |
| `marital` | Marital status | 1 = married, 2 = single, 3 = widowed |
| `education` | Education level | 1 = primary, 2 = middle school, 3 = high school, 4 = university |
| `comorbidity` | Comorbidity codes (comma-separated; 0 = none) | 1 = diabetes mellitus, 2 = hypertension, 10 = psoriatic arthritis, other codes = other conditions |
| `income` | Monthly household income (Turkish lira) | 1 = <10,000, 2 = 10,000–20,000, 3 = >20,000 |
| `drug` | Biologic agent | 1 = secukinumab, 2 = ixekizumab, 3 = risankizumab, 4 = guselkumab, empty = unknown (n = 1) |
| `PASI_0`, `PASI_6` | Psoriasis Area and Severity Index, baseline / month 6 | |
| `CRP_0`, `CRP_6` | C-reactive protein (mg/L), baseline / month 6 | |
| `NLR_0`, `NLR_6` | Neutrophil-to-lymphocyte ratio, baseline / month 6 | |
| `DLQI_0`, `DLQI_6` | Dermatology Life Quality Index (0–30) | |
| `HADSD_0`, `HADSD_6` | HADS depression subscale (0–21) | |
| `HADSA_0`, `HADSA_6` | HADS anxiety subscale (0–21) | |

Derived variables: PASI90 = (PASI_0 − PASI_6) / PASI_0 ≥ 0.90. Reductions are computed as baseline − month 6; the regression models use month 6 − baseline.

## Methods notes

- Medians and IQRs follow SPSS defaults (weighted average); Tables 3–4 use Tukey's hinges; Figures 2–3 use Excel's exclusive-median quartiles and whiskers to the last value within 1.5 × IQR.
- Effect sizes: r = |Z| / √N (Wilcoxon signed-rank), 95% CI from 5,000 bootstrap resamples.
- Figure 4 heatmaps show **unadjusted** pairwise Mann–Whitney U p-values. Bonferroni-adjusted Dunn's test results are reported in the text and in `results.txt`.

## Correction log (proof stage, October 2026)

Before publication, all reported values were re-checked against this dataset and re-run in SPSS. The following were corrected; study conclusions are unchanged:

- Table 2 median (IQR) values and three effect sizes; Tables 3–4 selected descriptive values, two p-values and effect sizes.
- Comorbidity (179 without / 56 with ≥1 comorbidity) and university education (24.3%).
- Table 5: HADS-A model summary (F(5,229) = 5.09, R² = 0.100, adjusted R² = 0.080).
- Spearman correlation ΔCRP–ΔHADS-A in the PASI90 subgroup: ρ = 0.261, p < 0.001 (previously copied from the wrong row of the output).
- Figure 4 legend: values are unadjusted; after Bonferroni correction only secukinumab vs. guselkumab remained significant for DLQI reduction (p = 0.017).
- Figure 1 (n = 235), Figure 2 legend labels, Figure 3 CRP unit (mg/L).

## Contact

Ahmet Kayaaltı, MD — University of Health Sciences, Şişli Hamidiye Etfal Training and Research Hospital, Department of Dermatology, Istanbul, Türkiye — ahmet199790@hotmail.com
