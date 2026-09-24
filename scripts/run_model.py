import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

def main(region_choice, region_col, file_choice, output_prefix):
    print(f"Selected region: {region_choice} ({region_col})")

    df = pd.read_csv(file_choice, encoding="utf-8-sig")
    print("Rows loaded:", len(df))
    print("Columns:", list(df.columns))

    df["log_region_sales"] = np.log1p(df[region_col])
    y_col = "log_region_sales"

    x_cols = [
        "critic_c", "critic_missing","year_c",
        "d_action", "d_sports", "d_misc", "d_adventure", "d_shooter",
        "d_racing", "d_simulation", "d_platform", "d_fighting",
        "d_strategy", "d_puzzle", "d_actionadventure", "d_visualnovel",
        "d_music", "d_other",
        "d_sega", "d_nintendohandheld", "d_nintendohome",
        "d_pcmac", "d_mshome", "d_sonyhandheld",
    ]

    region_drops = {
        "NA": {
            "columns": ["d_sega", "d_visualnovel"],
            "counts": {"d_sega": 2, "d_visualnovel": 14}
        },
        "JP": {
            "columns": ["d_pcmac"],
            "counts": {"d_pcmac": 0}
        },
        "PAL": {
            "columns": ["d_sega", "d_visualnovel"],
            "counts": {"d_sega": 2, "d_visualnovel": 7}
        },
        "OTHER": {
            "columns": ["d_sega", "d_visualnovel"],
            "counts": {"d_sega": 15, "d_visualnovel": 16}
        },
        "OVR": {
            "columns": [],
            "counts": {}
        }
    }

    drops = region_drops[region_choice]
    drop_cols = drops["columns"]
    drop_counts = drops["counts"]

    for col in drop_cols: #remove rows where dropped dummy is 1
        if col in df.columns:
            before = len(df)
            df = df[df[col] == 0].copy()
            after = len(df)
            print(f"Dropped {before - after} rows where {col} = 1 "
                  f"(original count in {region_choice}: {drop_counts[col]})")

    x_cols = [c for c in x_cols if c not in drop_cols] #drop

    missing = [c for c in [y_col] + x_cols if c not in df.columns]# check every column exists
    if missing:
        raise SystemExit(f"Missing columns in CSV: {missing}")

    model_df = df[[y_col] + x_cols].dropna() #keep only y & x
    print("Modelling rows:", len(model_df))

    y = model_df[y_col]
    X = sm.add_constant(model_df[x_cols])

    # VIF Check

    vif_data = pd.DataFrame()
    vif_data["variable"] = X.columns
    vif_data["VIF"] = [
        variance_inflation_factor(X.values, i)
        for i in range(X.shape[1])
    ]

    vif_data = vif_data[vif_data["variable"] != "const"]
    vif_data = vif_data.sort_values("VIF", ascending=False)

    print("\nVariance Inflation Factors:")
    print(vif_data.to_string(index=False))

    if region_choice == "OVR":
        vif_data.to_csv("../output/vif/vif_overall.csv", index=False)
    else:
        vif_data.to_csv(f"../output/vif/vif_{region_choice}.csv", index=False)

    model = sm.OLS(y, X).fit(cov_type="HC3")

    print("Summary: ")
    print(model.summary())
    print("=" * 70)


    coef = pd.DataFrame({ #report gen
        "variable": model.params.index,
        "coef": model.params.values,
        "std_err": model.bse.values,
        "t": model.tvalues.values,
        "p": model.pvalues.values,
        "pct_effect": (np.exp(model.params.values) - 1) * 100, #should convert log coefficients to percent effects
    })

    if region_choice == "OVR":
        output_file = f"{output_prefix}.csv"
    else:
        output_file = f"{output_prefix}_{region_choice}.csv"
    coef.to_csv(output_file, index=False)
    print(f"\nSaved {output_file}")

    resid = model.resid
    fitted = model.fittedvalues
    print("Residual std:", round(resid.std(), 4))
    print("Residual min:", round(resid.min(), 4))
    print("Residual max:", round(resid.max(), 4))

    # top 10 worst misses respectively
    df_resid = df.loc[model_df.index].copy()
    df_resid["resid"] = resid
    print("\nTop 10 under-predicted (model said low, actual high):")
    print(df_resid[["title", "console", "year", "resid"]]
          .sort_values("resid", ascending=False).head(10).to_string(index=False))

    print("\nTop 10 over-predicted (model said high, actual low):")
    print(df_resid[["title", "console", "year", "resid"]]
          .sort_values("resid").head(10).to_string(index=False))


file_choice = "../data/VGCHARTZ_DATA_v1.csv"
output_prefix = "../output/coefficients/se_coefficients"
region_map = {"NA": "na_sales","JP": "jp_sales","PAL": "pal_sales","OTHER": "other_sales", "OVR": "total_sales"}

while True:
    region_choice = input("Choose region (NA / JP / PAL / OTHER), all four regions (ALLREG), or overall stats (OVR): ").upper()
    if region_choice == "ALLREG":
        for r in ["NA", "JP", "PAL", "OTHER"]:
            region_col = region_map[r]
            main(r, region_col, file_choice, output_prefix)
            print("\n" + ("=" * 70) + "\n")
        break
    if region_choice in region_map:
        region_col = region_map[region_choice]
        main(region_choice, region_col, file_choice, output_prefix)
        break

    else:
        print("Valid options: NA, JP, PAL, OTHER, OVR, ALLREG")