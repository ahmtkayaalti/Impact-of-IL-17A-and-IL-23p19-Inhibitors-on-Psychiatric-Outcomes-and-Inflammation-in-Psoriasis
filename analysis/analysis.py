"""
Reproducible analysis for:
"Impact of IL-17A and IL-23p19 Inhibitors on Psychiatric Outcomes and
Inflammation in Psoriasis" (JDDG, DOI 10.1111/ddg.70579)

Input : psoriasis_data_anonymized.csv (n = 235, no names or dates)
Output: results.txt (all values reported in Tables 1-5 and the Results text)

Usage : python analysis.py
Requirements: Python >= 3.10, pandas, numpy, scipy
"""

import numpy as np
import pandas as pd
from scipy import stats

DATA = "../data/psoriasis_data_anonymized.csv"
OUT = "results.txt"
N_BOOT = 5000
SEED = 2026

DRUGS = {1: "Secukinumab", 2: "Ixekizumab", 3: "Risankizumab", 4: "Guselkumab"}
VARS = ["PASI", "CRP", "NLR", "DLQI", "HADS-D", "HADS-A"]
COMORB = {1: "Diabetes mellitus", 2: "Hypertension", 3: "Coronary artery disease",
          4: "Arrhythmia", 5: "Nephrolithiasis", 6: "Hypothyroidism",
          7: "Allergic rhinitis", 8: "Migraine", 9: "Hyperlipidemia",
          10: "Psoriatic arthritis", 12: "Chronic kidney disease", 14: "PCOS"}

rng = np.random.default_rng(SEED)
lines = []


def out(s=""):
    print(s)
    lines.append(s)


def pct(v):
    """Percentiles with SPSS default (weighted average, definition 1)."""
    return np.percentile(np.asarray(v, float), [25, 50, 75], method="weibull")


def med_iqr(v, d=2):
    q1, m, q3 = pct(v)
    return f"{m:.{d}f} ({q1:.{d}f}-{q3:.{d}f})"


def wilcoxon_r(a, b):
    """Wilcoxon signed-rank test; effect size r = |z|/sqrt(N); bootstrap 95% CI."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    res = stats.wilcoxon(a, b, method="approx")
    n = len(a)
    r = abs(res.zstatistic) / np.sqrt(n)
    boots = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        try:
            z = stats.wilcoxon(a[idx], b[idx], method="approx").zstatistic
            boots.append(abs(z) / np.sqrt(n))
        except ValueError:
            continue
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return res.pvalue, r, lo, hi


def fmt_p(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def dunn_bonferroni(groups):
    """Dunn's test with Bonferroni correction. groups: dict name -> values."""
    names = list(groups)
    allv = np.concatenate([np.asarray(groups[k], float) for k in names])
    ranks = stats.rankdata(allv)
    N = len(allv)
    _, counts = np.unique(allv, return_counts=True)
    tie = np.sum(counts**3 - counts) / (12 * (N - 1))
    mean_rank, n, pos = {}, {}, 0
    for k in names:
        n[k] = len(groups[k])
        mean_rank[k] = ranks[pos:pos + n[k]].mean()
        pos += n[k]
    m = len(names) * (len(names) - 1) / 2
    res = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            se = np.sqrt((N * (N + 1) / 12 - tie) * (1 / n[a] + 1 / n[b]))
            z = (mean_rank[a] - mean_rank[b]) / se
            res[(a, b)] = min(1.0, 2 * stats.norm.sf(abs(z)) * m)
    return res


def std_ols(y, X):
    """OLS on z-standardised variables: standardised beta, 95% CI, p, F, R2."""
    y = np.asarray(y, float)
    Xv = X.values.astype(float)
    yz = (y - y.mean()) / y.std(ddof=1)
    Xz = (Xv - Xv.mean(0)) / Xv.std(0, ddof=1)
    A = np.column_stack([np.ones(len(yz)), Xz])
    beta, *_ = np.linalg.lstsq(A, yz, rcond=None)
    resid = yz - A @ beta
    n, k = A.shape
    df = n - k
    s2 = resid @ resid / df
    se = np.sqrt(np.diag(s2 * np.linalg.inv(A.T @ A)))
    t = beta / se
    p = 2 * stats.t.sf(np.abs(t), df)
    tcrit = stats.t.ppf(0.975, df)
    r2 = 1 - (resid @ resid) / (yz @ yz)
    adj = 1 - (1 - r2) * (n - 1) / df
    F = (r2 / (k - 1)) / ((1 - r2) / df)
    pF = stats.f.sf(F, k - 1, df)
    tab = pd.DataFrame({"beta": beta[1:], "ci_lo": beta[1:] - tcrit * se[1:],
                        "ci_hi": beta[1:] + tcrit * se[1:], "p": p[1:]},
                       index=X.columns)
    return tab, F, k - 1, df, pF, r2, adj


# ---------------------------------------------------------------- data
d = pd.read_csv(DATA)
d = d.rename(columns=lambda c: c.replace("HADSD", "HADS-D").replace("HADSA", "HADS-A"))
N = len(d)
for v in VARS:
    d[f"{v}_0"] = d[f"{v}_0"].astype(float)
    d[f"{v}_6"] = d[f"{v}_6"].astype(float)
d["drug"] = d["drug"].map(DRUGS)
d["PASI90"] = (d["PASI_0"] - d["PASI_6"]) / d["PASI_0"] >= 0.90

out(f"N = {N}")

# ---------------------------------------------------------------- Table 1
out("\n=== TABLE 1: Demographics ===")
out(f"Age: mean {d.age.mean():.1f} +/- {d.age.std():.1f}; median {med_iqr(d.age, 1)}")
for col, lab in [("sex", {1: "Female", 2: "Male"}),
                 ("marital", {1: "Married", 2: "Single", 3: "Widowed"}),
                 ("education", {1: "Primary", 2: "Middle", 3: "High school", 4: "University"}),
                 ("income", {1: "<10,000", 2: "10,000-20,000", 3: ">20,000"})]:
    vc = d[col].value_counts().sort_index()
    out(f"{col}: " + "; ".join(f"{lab.get(k, f'code {k}')} {c} ({c / N * 100:.1f}%)"
                              for k, c in vc.items()))
codes = d["comorbidity"].astype(str).str.replace(" ", "").str.split(",")
has = codes.apply(lambda l: l != ["0"])
out(f"Comorbidity: none {(~has).sum()} ({(~has).mean() * 100:.1f}%); "
    f"at least one {has.sum()} ({has.mean() * 100:.1f}%)")
for c, name in COMORB.items():
    k = codes.apply(lambda l: str(c) in l).sum()
    if k:
        out(f"  {name}: {k} ({k / N * 100:.1f}%)")
vc = d["drug"].value_counts(dropna=False)
out("Treatment: " + "; ".join(f"{k if isinstance(k, str) else 'Unknown'} {c} ({c / N * 100:.1f}%)"
                             for k, c in vc.items()))

# ---------------------------------------------------------------- Table 2
out("\n=== TABLE 2: Baseline vs month 6 (median (IQR), Wilcoxon) ===")
for v in VARS:
    p, r, lo, hi = wilcoxon_r(d[f"{v}_0"], d[f"{v}_6"])
    dec = 2 if v in ("CRP", "NLR") else 1
    out(f"{v:7s} {med_iqr(d[f'{v}_0'], dec):22s} {med_iqr(d[f'{v}_6'], dec):22s} "
        f"p {fmt_p(p):7s} r {r:.2f} (95% CI {lo:.2f}-{hi:.2f})")

# ---------------------------------------------------------------- HADS >= 8
out("\n=== HADS >= 8 (clinically significant symptoms) ===")
for v in ["HADS-D", "HADS-A"]:
    b, f = d[f"{v}_0"] >= 8, d[f"{v}_6"] >= 8
    remit = (b & ~f).sum() / b.sum()
    out(f"{v}: baseline {b.mean() * 100:.1f}%, month 6 {f.mean() * 100:.1f}%; "
        f"remission among symptomatic {remit * 100:.1f}%; NNT {1 / remit:.2f}")

# ---------------------------------------------------------------- Table 3
sub = d.dropna(subset=["drug"])
out(f"\n=== TABLE 3: CRP and NLR by biologic (n = {len(sub)}) ===")
for v in ["CRP", "NLR"]:
    for drug in DRUGS.values():
        g = sub[sub.drug == drug]
        p, r, lo, hi = wilcoxon_r(g[f"{v}_0"], g[f"{v}_6"])
        out(f"{v} {drug:13s} M0 {med_iqr(g[f'{v}_0'])}  M6 {med_iqr(g[f'{v}_6'])}  "
            f"change {med_iqr(g[f'{v}_6'] - g[f'{v}_0'])}  p {fmt_p(p)}  "
            f"r {r:.2f} ({lo:.2f}-{hi:.2f})")
    for lab, series in [("M0", f"{v}_0"), ("M6", f"{v}_6")]:
        kw = stats.kruskal(*[sub[sub.drug == k][series] for k in DRUGS.values()])
        out(f"  Kruskal-Wallis {v} {lab}: p {fmt_p(kw.pvalue)}")
    kw = stats.kruskal(*[(sub[sub.drug == k][f"{v}_6"] - sub[sub.drug == k][f"{v}_0"])
                         for k in DRUGS.values()])
    out(f"  Kruskal-Wallis {v} change: p {fmt_p(kw.pvalue)}")

# ---------------------------------------------------------------- Table 4
out("\n=== TABLE 4: DLQI and HADS by biologic ===")
for v in ["DLQI", "HADS-D", "HADS-A"]:
    for drug in ["Risankizumab", "Ixekizumab", "Secukinumab", "Guselkumab"]:
        g = sub[sub.drug == drug]
        p, r, lo, hi = wilcoxon_r(g[f"{v}_0"], g[f"{v}_6"])
        change = np.median(g[f"{v}_6"] - g[f"{v}_0"])
        out(f"{v:7s} {drug:13s} {med_iqr(g[f'{v}_0'], 1):20s} {med_iqr(g[f'{v}_6'], 1):18s} "
            f"median change {change:5.1f}  r {r:.2f} ({lo:.2f}-{hi:.2f})  p {fmt_p(p)}")

# ---------------------------------------------------------------- Figure 4 (Dunn)
out("\n=== Pairwise comparisons of change (Dunn, Bonferroni) ===")
for v in ["DLQI", "CRP", "NLR"]:
    groups = {k: (sub[sub.drug == k][f"{v}_0"] - sub[sub.drug == k][f"{v}_6"]).values
              for k in DRUGS.values()}
    kw = stats.kruskal(*groups.values())
    out(f"{v} reduction: Kruskal-Wallis p {fmt_p(kw.pvalue)}")
    for (a, b), p in dunn_bonferroni(groups).items():
        out(f"  {a} vs {b}: p {p:.3f}")

# ---------------------------------------------------------------- Spearman
out("\n=== Spearman: CRP/NLR reduction vs HADS improvement, by PASI90 ===")
out(f"PASI90 achieved: {d.PASI90.sum()} ({d.PASI90.mean() * 100:.1f}%)")
for marker in ["CRP", "NLR"]:
    x_all = d[f"{marker}_0"] - d[f"{marker}_6"]
    for grp, mask in [("non-PASI90", ~d.PASI90), ("PASI90", d.PASI90)]:
        for v in ["HADS-D", "HADS-A"]:
            y = (d[f"{v}_0"] - d[f"{v}_6"])[mask]
            rho, p = stats.spearmanr(x_all[mask], y)
            out(f"{marker} {grp:10s} (n={mask.sum()}) {v}: rho {rho:.3f}, p {fmt_p(p)}")

# ---------------------------------------------------------------- Table 5
out("\n=== TABLE 5: Multiple linear regression (standardised beta) ===")
out("Change defined as month 6 minus baseline; adjusted for delta-PASI, age and sex.")
X = pd.DataFrame({"CRP change": d["CRP_6"] - d["CRP_0"],
                  "NLR change": d["NLR_6"] - d["NLR_0"],
                  "PASI change": d["PASI_6"] - d["PASI_0"],
                  "Age": d["age"], "Sex": d["sex"]})
for v in ["HADS-D", "HADS-A"]:
    tab, F, df1, df2, pF, r2, adj = std_ols(d[f"{v}_6"] - d[f"{v}_0"], X)
    out(f"{v} change: F({df1},{df2}) = {F:.2f}, p {fmt_p(pF)}, "
        f"R2 = {r2:.3f}, adjusted R2 = {adj:.3f}")
    for name, row in tab.iterrows():
        out(f"  {name:12s} beta {row.beta:6.3f}  95% CI {row.ci_lo:6.3f} to "
            f"{row.ci_hi:6.3f}  p {fmt_p(row.p)}")

with open(OUT, "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines) + "\n")
print(f"\nSaved to {OUT}")
