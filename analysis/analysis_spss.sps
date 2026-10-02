* Encoding: UTF-8.
* =====================================================================.
* JDDG ddg.70579 - IL-17A / IL-23p19 inhibitors, psychiatric outcomes.
* Reproduces all values in Tables 1-5, Figure 4 and the Results text.
* HOW TO RUN: put this file and psoriasis_data_anonymized.csv in the same
* folder, change the path in FILE= below, then Run > All.
* Save the output afterwards: File > Save As (.spv) and File > Export (PDF).
* =====================================================================.

GET DATA
  /TYPE=TXT
  /FILE='C:\your_folder\data\psoriasis_data_anonymized.csv'
  /ENCODING='UTF8'
  /DELIMITERS=","
  /QUALIFIER='"'
  /ARRANGEMENT=DELIMITED
  /FIRSTCASE=2
  /VARIABLES=
    id F4.0 age F3.0 sex F1.0 marital F1.0 education F1.0 comorbidity A30
    income F1.0 drug F1.0
    PASI_0 F8.2 PASI_6 F8.2 CRP_0 F8.2 CRP_6 F8.2 NLR_0 F8.3 NLR_6 F8.3
    DLQI_0 F3.0 DLQI_6 F3.0 HADSD_0 F3.0 HADSD_6 F3.0 HADSA_0 F3.0 HADSA_6 F3.0.
EXECUTE.

VALUE LABELS sex 1 'Female' 2 'Male'
  /marital 1 'Married' 2 'Single' 3 'Widowed'
  /education 1 'Primary' 2 'Middle school' 3 'High school' 4 'University'
  /income 1 '<10,000 TL' 2 '10,000-20,000 TL' 3 '>20,000 TL'
  /drug 1 'Secukinumab' 2 'Ixekizumab' 3 'Risankizumab' 4 'Guselkumab'.
* Note: one case has income code 4 (invalid entry) - check source record.

* ---------- Derived variables.
* Reductions (baseline - month 6), used for Spearman and Kruskal-Wallis.
COMPUTE red_PASI = PASI_0 - PASI_6.
COMPUTE red_CRP = CRP_0 - CRP_6.
COMPUTE red_NLR = NLR_0 - NLR_6.
COMPUTE red_DLQI = DLQI_0 - DLQI_6.
COMPUTE red_HADSD = HADSD_0 - HADSD_6.
COMPUTE red_HADSA = HADSA_0 - HADSA_6.
* Changes (month 6 - baseline), used for the regression models.
COMPUTE chg_PASI = PASI_6 - PASI_0.
COMPUTE chg_CRP = CRP_6 - CRP_0.
COMPUTE chg_NLR = NLR_6 - NLR_0.
COMPUTE chg_HADSD = HADSD_6 - HADSD_0.
COMPUTE chg_HADSA = HADSA_6 - HADSA_0.
COMPUTE PASI90 = ((PASI_0 - PASI_6) / PASI_0 >= 0.90).
COMPUTE comorb_any = (LTRIM(RTRIM(comorbidity)) NE '0').
COMPUTE HADSD0_8 = (HADSD_0 >= 8).
COMPUTE HADSD6_8 = (HADSD_6 >= 8).
COMPUTE HADSA0_8 = (HADSA_0 >= 8).
COMPUTE HADSA6_8 = (HADSA_6 >= 8).
VALUE LABELS PASI90 0 'Non-PASI90' 1 'PASI90'
  /comorb_any 0 'No comorbidity' 1 'At least one'.
VARIABLE LEVEL drug sex PASI90 (NOMINAL).
EXECUTE.

* ---------- TABLE 1.
DESCRIPTIVES VARIABLES=age /STATISTICS=MEAN STDDEV MIN MAX.
FREQUENCIES VARIABLES=sex marital education income comorb_any drug
  /ORDER=ANALYSIS.

* ---------- TABLE 2: median (IQR) and Wilcoxon signed-rank.
EXAMINE VARIABLES=PASI_0 PASI_6 CRP_0 CRP_6 NLR_0 NLR_6 DLQI_0 DLQI_6
    HADSD_0 HADSD_6 HADSA_0 HADSA_6
  /PERCENTILES(25,50,75)=HAVERAGE
  /STATISTICS=NONE
  /PLOT=NONE.
NPAR TESTS
  /WILCOXON=PASI_0 CRP_0 NLR_0 DLQI_0 HADSD_0 HADSA_0
    WITH PASI_6 CRP_6 NLR_6 DLQI_6 HADSD_6 HADSA_6 (PAIRED).
* Effect size r = |Z| / SQRT(235), computed from the Z values above.
* (Bootstrap CIs for r are not available in SPSS; see analysis.py.)

* ---------- HADS >= 8 at baseline and month 6, and remission.
FREQUENCIES VARIABLES=HADSD0_8 HADSD6_8 HADSA0_8 HADSA6_8.
CROSSTABS /TABLES=HADSD0_8 BY HADSD6_8 /CELLS=COUNT ROW.
CROSSTABS /TABLES=HADSA0_8 BY HADSA6_8 /CELLS=COUNT ROW.

* ---------- TABLES 3 and 4: within each biologic.
SORT CASES BY drug.
SPLIT FILE LAYERED BY drug.
EXAMINE VARIABLES=CRP_0 CRP_6 chg_CRP NLR_0 NLR_6 chg_NLR
    DLQI_0 DLQI_6 HADSD_0 HADSD_6 HADSA_0 HADSA_6
  /PERCENTILES(25,50,75)=HAVERAGE
  /STATISTICS=NONE
  /PLOT=NONE.
NPAR TESTS
  /WILCOXON=CRP_0 NLR_0 DLQI_0 HADSD_0 HADSA_0
    WITH CRP_6 NLR_6 DLQI_6 HADSD_6 HADSA_6 (PAIRED).
SPLIT FILE OFF.

* ---------- Between-group comparisons (Kruskal-Wallis).
NPAR TESTS
  /K-W=CRP_0 CRP_6 red_CRP NLR_0 NLR_6 red_NLR BY drug(1 4).

* ---------- FIGURE 4: pairwise comparisons (Dunn, Bonferroni-adjusted).
NPTESTS
  /INDEPENDENT TEST (red_DLQI red_CRP red_NLR) GROUP (drug)
    KRUSKAL_WALLIS(COMPARE=PAIRWISE)
  /MISSING SCOPE=ANALYSIS USERMISSING=EXCLUDE
  /CRITERIA ALPHA=0.05 CILEVEL=95.

* ---------- Spearman correlations by PASI90 status.
FREQUENCIES VARIABLES=PASI90.
SORT CASES BY PASI90.
SPLIT FILE LAYERED BY PASI90.
NONPAR CORR
  /VARIABLES=red_CRP red_NLR WITH red_HADSD red_HADSA
  /PRINT=SPEARMAN TWOTAIL NOSIG
  /MISSING=PAIRWISE.
SPLIT FILE OFF.

* ---------- TABLE 5: multiple linear regression.
* Standardised betas: read the "Standardized Coefficients Beta" column.
REGRESSION
  /STATISTICS COEFF OUTS CI(95) R ANOVA
  /DEPENDENT chg_HADSD
  /METHOD=ENTER chg_CRP chg_NLR chg_PASI age sex.
REGRESSION
  /STATISTICS COEFF OUTS CI(95) R ANOVA
  /DEPENDENT chg_HADSA
  /METHOD=ENTER chg_CRP chg_NLR chg_PASI age sex.

* 95% CIs for the STANDARDISED betas: rerun on z-scores.
* In these models B = standardised beta and CI(95) is its confidence interval.
DESCRIPTIVES VARIABLES=chg_HADSD chg_HADSA chg_CRP chg_NLR chg_PASI age sex
  /SAVE.
REGRESSION
  /STATISTICS COEFF CI(95)
  /DEPENDENT Zchg_HADSD
  /METHOD=ENTER Zchg_CRP Zchg_NLR Zchg_PASI Zage Zsex.
REGRESSION
  /STATISTICS COEFF CI(95)
  /DEPENDENT Zchg_HADSA
  /METHOD=ENTER Zchg_CRP Zchg_NLR Zchg_PASI Zage Zsex.
