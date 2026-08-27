#NB: Parts of this code were contributed by project team member Nollan Verges.

# Import
from statsmodels.stats.power import TTestIndPower
import pandas as pd
from scipy import stats
from itertools import combinations, product
from scipy.stats import shapiro, mannwhitneyu, ttest_ind, levene, bartlett, f_oneway, kruskal, chi2_contingency
import pingouin as pg
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.stats.multitest as smm


# definition of df per USP level

df_USP_150 = pd.read_excel("USP_150_cal.xlsx", dtype={'Date': str})
df_USP_220 = pd.read_excel("USP_220_cal.xlsx", dtype={'Date': str})
df_USP_240 = pd.read_excel("USP_240_cal.xlsx", dtype={'Date': str})
df_USP_260 = pd.read_excel("USP_260_cal.xlsx", dtype={'Date': str})
df_USP_280 = pd.read_excel("USP_280_cal.xlsx", dtype={'Date': str})
df_USP_300 = pd.read_excel("USP_300_cal.xlsx", dtype={'Date': str})
df_USP_320 = pd.read_excel("USP_320_cal.xlsx", dtype={'Date': str})
print("Load all excel files succesfully")

# visualisation of data
df_USP = [df_USP_150, df_USP_220, df_USP_240, df_USP_260,
          df_USP_280, df_USP_300, df_USP_320]
labels = ["USP 150", "USP 220", "USP 240",
          "USP 260", "USP 280", "USP 300", "USP 320"]


# Per USP levels repartition of types of failure mode
for i, df in enumerate(df_USP):
    label = labels[i]
    unique_failure_methods = df["Failure Mode"].value_counts()
    fig, ax = plt.subplots(figsize=(6, 4))
    unique_failure_methods.plot(kind='bar', ax=ax)
    ax.set_xlabel('Failure methods')
    ax.set_ylabel('Count')
    ax.set_title(label)
    plt.tight_layout()
    plt.savefig(f"./Res/{label}_repartition_of_failure_methods.png")
    plt.close()

print("Repartition of failure methods saved in Res folder")

# boxplot
df_USP_peak_force = [df["Peak Force - Cal (gf)"] for df in df_USP]
fig, ax = plt.subplots()
ax.set_ylabel('Peak force (gf)')
# correct parameter is "labels"
ax.boxplot(df_USP_peak_force, tick_labels=labels)
ax.set_title('Comparison of Peak force by USP_level')
plt.savefig("./Res/Comparison of Peak force by USP level.png")
plt.close()

print("Boxplot of comparison of Peak force by USP_level saved in Res folder")

# Mean and std per USP level
for i, df in enumerate(df_USP_peak_force):
    print(f"{labels[i]} - Mean: {df.mean():.4f}, Std: {df.std(ddof=1):.4f}")

# Is there a difference between USP level?
# Test of normality
normality_flags = []
for i, df in enumerate(df_USP_peak_force):
    stat, p_val = stats.shapiro(df)
    normality_flags.append(p_val >= 0.05)
    if p_val >= 0.05:
        print(
            f"{labels[i]} peak force follows a normal distribution (p = {p_val:.4f})")
    else:
        print(
            f"{labels[i]} peak force doesn't follow a normal distribution (p = {p_val:.4f})")

all_normal = all(normality_flags)
equal_var = False

df_list = []
for label, df in zip(labels, df_USP):
    temp = pd.DataFrame({
        'USP_Level': label,
        'Peak_Force': df["Peak Force - Cal (gf)"]
    })

    df_list.append(temp)

df_all = pd.concat(df_list, ignore_index=True)

if all_normal:
    pvar = bartlett(*df_USP_peak_force).pvalue
    equal_var = pvar > 0.05

    if equal_var:
        test = "ANOVA"
        pval = f_oneway(*df_USP_peak_force).pvalue
    else:
        test = "Welch-ANOVA"
        # pval = pg.welch_anova(dv="Peak_Force", between="USP_Level", data=df_all)['p-unc'].iloc[0]
        res_welch = pg.welch_anova(
            dv="Peak_Force", between="USP_Level", data=df_all)
        pval_col = [c for c in ['p-unc', 'p_unc', 'pval', 'p-val']
                    if c in res_welch.columns][0]
        pval = res_welch[pval_col].iloc[0]
else:
    test = "Kruskal-Wallis"
    pval = kruskal(*df_USP_peak_force).pvalue

print(f"\n--- Test executed : {test} | p-value = {pval:.4e} ---")

if pval < 0.05:
    print(" -> At least one level is different from others. Proceeding to Post-Hoc analysis...\n")

    # Option A : Welch-ANOVA (Games-Howell)
    if test == "Welch-ANOVA":
        print("=== Post-Hoc Analysis: Games-Howell ===")
        posthoc = pg.pairwise_gameshowell(
            dv='Peak_Force', between='USP_Level', data=df_all)
        # print(posthoc[['A', 'B', 'pval', 'pval_corr', 'sig']])
        print(posthoc)

    # Option B : ANOVA or Kruskal-Wallis (Pairwise t-test / Mann-Whitney + Holm)
    else:
        print(
            f"=== Post-Hoc Analysis: Pairwise Tests with Holm Correction ({'t-test' if all_normal else 'Mann-Whitney'}) ===")
        p_values = []
        comparisons = list(combinations(
            sorted(df_all['USP_Level'].unique()), 2))

        for a, b in comparisons:
            g1 = df_all.loc[df_all['USP_Level'] == a, 'Peak_Force']
            g2 = df_all.loc[df_all['USP_Level'] == b, 'Peak_Force']

            if all_normal:
                stat, p = stats.ttest_ind(g1, g2, equal_var=equal_var)
            else:
                stat, p = stats.mannwhitneyu(g1, g2)
            p_values.append(p)

        # Holm corrected pairwise comparison between USP groups
        reject, pvals_corr, _, _ = smm.multipletests(
            p_values, alpha=0.05, method='holm')

        for (a, b), p, p_corr, rej in zip(comparisons, p_values, pvals_corr, reject):
            conclusion = " -> different distribution (p_corr < 0.05)" if rej else " -> similar distribution (p_corr ≥ 0.05)"
            print(f"{a} vs {b}: p = {p:.4f}, p_corr = {p_corr:.4f} {conclusion}")

else:
    print(" -> No significant difference between USP levels.")


# Power Analysis using Cohen's d

if pval < 0.05:
    print("\n Power Analysis (Cohen's d)")
    analysis = TTestIndPower()
    alpha = 0.05
    target_power = 0.80

    comparisons_power = list(combinations(
        sorted(df_all['USP_Level'].unique()), 2))

    power_results = []

    for a, b in comparisons_power:
        g1 = df_all.loc[df_all['USP_Level'] == a, 'Peak_Force']
        g2 = df_all.loc[df_all['USP_Level'] == b, 'Peak_Force']

        n1, n2 = len(g1), len(g2)

        pooled_std = np.sqrt(
            ((n1 - 1) * g1.var(ddof=1) + (n2 - 1) * g2.var(ddof=1)) / (n1 + n2 - 2)
        )
        d = abs(g1.mean() - g2.mean()) / pooled_std

        achieved_power = analysis.power(
            effect_size=d, nobs1=n1, alpha=alpha, ratio=n2 / n1
        )

        n_required = analysis.solve_power(
            effect_size=d, power=target_power, alpha=alpha, ratio=1.0
        )

        power_results.append({
            'Comparison': f"{a} vs {b}",
            "Cohen's d": round(d, 3),
            'n1': n1, 'n2': n2,
            'Achieved Power': round(achieved_power, 3),
            f'n required per group (power={target_power})': int(np.ceil(n_required))
        })

    power_df = pd.DataFrame(power_results)
    print(power_df.to_string(index=False))
    power_df.to_csv("./Res/power_analysis.csv", index=False)
    print("\nPower analysis saved to ./New_Res150/power_analysis.csv")

    print("\n--- Sample size needed per group to detect a range of effect sizes "
          f"(power={target_power}, alpha={alpha}) ---")
    for d_target, label in [(0.2, 'small'), (0.5, 'medium'), (0.8, 'large')]:
        n_req = analysis.solve_power(
            effect_size=d_target, power=target_power, alpha=alpha, ratio=1.0
        )
        print(f"d = {d_target} ({label}): n = {int(np.ceil(n_req))} per group")

    if not all_normal:
        print(
            "\nNote: peak force did not meet the Shapiro-Wilk normality assumption "
            "for at least one USP level, so the t-test-based power calculation above "
            "is an approximation."
        )
else:
    print(" No significant difference between USP levels; power analysis skipped.")
