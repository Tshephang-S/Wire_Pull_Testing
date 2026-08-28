
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from statsmodels.stats.inter_rater import fleiss_kappa, aggregate_raters


Input_file = "FM_Data_Initial.xlsx"  # transposed Attribute agreement responses
Output_File = "Attribute_Agreement_Results.xlsx"
Reference_Name = "Reference"       # reference

df = pd.read_excel(Input_file)


def intra_rater_reliability(df):
    intra_results = []  # empty list for storing results
    appraisers = sorted(appr for appr in df["Appraiser"].unique(
    ) if appr != Reference_Name)  # sort appraisers alphabetically, exclude Reference
    trials = sorted(df["Trial"].unique())

    for appraiser in appraisers:
        # select data relating to specific appraiser
        appr_data = df[df["Appraiser"] == appraiser]

        wire_data = appr_data.pivot(
            index="Wire ID", columns="Trial", values="FM")
        t1, t2 = trials[0], trials[1]
        wire_data = wire_data.dropna(subset=[t1, t2])

        # compute kappa for trial 1 vs trial 2
        kappa = cohen_kappa_score(wire_data[t1], wire_data[t2])
        # calculate percentaga of identical classifications
        pct_agree = (wire_data[t1] == wire_data[t2]).mean() * 100

        intra_results.append({
            "Appraiser": appraiser,
            "N wires compared": len(wire_data),
            "% Agreement (Trial1 vs Trial2)": round(pct_agree, 1),
            "Cohen's Kappa (self-agreement)": round(kappa, 3), })
    return pd.DataFrame(intra_results)


def inter_rater_reliability(df):
    rater_df = df[df["Appraiser"] != Reference_Name].copy()
    rater_df["Subject"] = rater_df["Wire ID"].astype(
        # unique identifier for a wire and trial number (1_T1, 1_T2...)
        str) + "_T" + rater_df["Trial"].astype(str)

    wide = rater_df.pivot(index="Subject", columns="Appraiser", values="FM")
    wide = wide.dropna()  # only keep subjects every appraiser rated

    # classifications into counts for each FM
    counts_table, categories = aggregate_raters(wide.values)
    # compute kappa between the appraisers
    kappa = fleiss_kappa(counts_table, method="fleiss")

    n_appraisers = wide.shape[1]
    n_subjects = wide.shape[0]

    inter_results = pd.DataFrame([{
        "N appraisers compared": n_appraisers,
        "N subjects (wire x trial)": n_subjects,
        "N failure mode categories": len(categories),
        "Fleiss' Kappa (inter-rater)": round(kappa, 3),
    }])
    return inter_results


def accuracy_vs_reference(df):

    ref = df[df["Appraiser"] == Reference_Name][["Wire ID", "Trial", "FM"]]
    ref = ref.rename(columns={"FM": "Reference_FM"})

    results = []
    appraisers = sorted(
        a for a in df["Appraiser"].unique() if a != Reference_Name)

    for appraiser in appraisers:
        sub = df[df["Appraiser"] == appraiser][["Wire ID", "Trial", "FM"]]
        merged = sub.merge(ref, on=["Wire ID", "Trial"])

        pct_agree = (merged["FM"] == merged["Reference_FM"]).mean() * 100
        kappa = cohen_kappa_score(merged["FM"], merged["Reference_FM"])

        results.append({
            "Appraiser": appraiser,
            "N classifications checked": len(merged),
            "% Accuracy vs Reference": round(pct_agree, 1),
            "Cohen's Kappa vs Reference": round(kappa, 3),
        })

    return pd.DataFrame(results)


def system_agreement(df):
    """
    Strictest measure: for each wire/trial, do ALL appraisers agree with
    EACH OTHER and match the Reference? Returns the overall % of
    wire/trial combinations that pass, plus a breakdown of the failures.
    """
    ref = df[df["Appraiser"] == Reference_Name][["Wire ID", "Trial", "FM"]]
    ref = ref.rename(columns={"FM": "Reference_FM"})

    rater_df = df[df["Appraiser"] != Reference_Name]
    wide = rater_df.pivot(index=["Wire ID", "Trial"],
                          columns="Appraiser", values="FM")
    wide = wide.merge(ref.set_index(
        ["Wire ID", "Trial"]), left_index=True, right_index=True)
    wide = wide.dropna()

    rater_cols = [c for c in wide.columns if c != "Reference_FM"]

    def all_match(row):
        return all(row[c] == row["Reference_FM"] for c in rater_cols)

    wide["System_Agrees"] = wide.apply(all_match, axis=1)

    overall_pct = wide["System_Agrees"].mean() * 100
    summary = pd.DataFrame([{
        "N wire/trial combinations": len(wide),
        "N where ALL appraisers match Reference": int(wide["System_Agrees"].sum()),
        "% System Agreement": round(overall_pct, 1),
    }])

    # Detail table: which specific wire/trials failed, and why
    detail = wide.reset_index()
    detail = detail.rename(columns={"Reference_FM": "Reference"})
    return summary, detail


# def main():
#     df = load_data(Input_file)

print("1. INTRA-RATER RELIABILITY (Cohen's Kappa, self vs self)")

intra = intra_rater_reliability(df)
print(intra.to_string(index=False))

print("2. INTER-RATER RELIABILITY (Fleiss' Kappa, all appraisers)")

inter = inter_rater_reliability(df)
print(inter.to_string(index=False))

print("3. ACCURACY vs REFERENCE (per appraiser)")

accuracy = accuracy_vs_reference(df)
print(accuracy.to_string(index=False))

print("4. SYSTEM AGREEMENT (all appraisers agree AND match Reference)")

sys_summary, sys_detail = system_agreement(df)
print(sys_summary.to_string(index=False))

with pd.ExcelWriter(Output_File, engine="openpyxl") as writer:
    intra.to_excel(writer, sheet_name="Intra-rater (Cohen)", index=False)
    inter.to_excel(writer, sheet_name="Inter-rater (Fleiss)", index=False)
    accuracy.to_excel(
        writer, sheet_name="Accuracy vs Reference", index=False)
    sys_summary.to_excel(
        writer, sheet_name="System Agreement", index=False)
    sys_detail.to_excel(
        writer, sheet_name="System Agreement Detail", index=False)

print(f"\nAll results saved to: {Output_File}")


# if __name__ == "__main__":
#     main()
