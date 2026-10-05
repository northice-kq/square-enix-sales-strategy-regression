import pandas as pd
import numpy as np
import statsmodels.api as sm
import os

os.makedirs("../output/coefficients/standardized", exist_ok=True)

file_choice = "../data/VGCHARTZ_DATA_v1.csv"
output_prefix = "../output/coefficients/standardized/standardized"

region_map = {
    "NA": "na_sales",
    "JP": "jp_sales",
    "PAL": "pal_sales",
    "OTHER": "other_sales",
    "OVR": "total_sales",
}

region_drops = {
    "NA": ["d_sega", "d_visualnovel"],
    "JP": ["d_pcmac"],
    "PAL": ["d_sega", "d_visualnovel"],
    "OTHER": ["d_sega", "d_visualnovel"],
    "OVR": [],
}

x_cols_all = [
    "critic_c", "critic_missing", "year_c",
    "d_action", "d_sports", "d_misc", "d_adventure", "d_shooter",
    "d_racing", "d_simulation", "d_platform", "d_fighting",
    "d_strategy", "d_puzzle", "d_actionadventure", "d_visualnovel",
    "d_music", "d_other",
    "d_sega", "d_nintendohandheld", "d_nintendohome",
    "d_pcmac", "d_mshome", "d_sonyhandheld",
]

# build standardised table

def build_table(region_choice):
    df = pd.read_csv(file_choice, encoding="utf-8-sig")

    region_col = region_map[region_choice]
    df["log_region_sales"] = np.log1p(df[region_col])
    y_col = "log_region_sales"

    drop_cols = region_drops[region_choice]
    for col in drop_cols:
        df = df[df[col] == 0].copy()

    x_cols = [c for c in x_cols_all if c not in drop_cols]

    model_df = df[[y_col] + x_cols].dropna()
    y = model_df[y_col]
    X = model_df[x_cols]
    Xc = sm.add_constant(X)

    model = sm.OLS(y, Xc).fit(cov_type="HC3")

    sd_y = y.std(ddof=1)
    sd_x = X.std(ddof=1)

    rows = []
    for var in model.params.index:
        b = model.params[var]
        se_b = model.bse[var]
        t = model.tvalues[var]
        p = model.pvalues[var]

        if var == "const":
            rows.append({
                "Variable": "Intercept",
                "B": b,
                "SE_B": se_b,
                "Beta": np.nan,
                "SE_Beta": np.nan,
                "t": t,
                "p": p,
            })
        else:
            beta = b * (sd_x[var] / sd_y)
            se_beta = se_b * (sd_x[var] / sd_y)
            rows.append({
                "Variable": var,
                "B": b,
                "SE_B": se_b,
                "Beta": beta,
                "SE_Beta": se_beta,
                "t": t,
                "p": p,
            })

    table = pd.DataFrame(rows)[
        ["Variable", "B", "SE_B", "Beta", "SE_Beta", "t", "p"]
    ]

    table[["B", "SE_B", "Beta", "SE_Beta", "t"]] = table[
        ["B", "SE_B", "Beta", "SE_Beta", "t"]
    ].round(4)

    return table

all_tables = {}
for region_choice in ["OVR", "NA", "JP", "PAL", "OTHER"]:
    print(f"Building table for {region_choice}...")
    table = build_table(region_choice)
    all_tables[region_choice] = table

    # Save individual
    table.to_excel(
        f"{output_prefix}_{region_choice}.xlsx", index=False
    )
    table.to_csv(
        f"{output_prefix}_{region_choice}.csv", index=False
    )

print("\nAll standardized coefficient tables saved.")

with pd.ExcelWriter(f"{output_prefix}_ALL.xlsx") as writer:
    for region, table in all_tables.items():
        table.to_excel(writer, sheet_name=region, index=False)

print(f"Combined workbook saved: {output_prefix}_ALL.xlsx")