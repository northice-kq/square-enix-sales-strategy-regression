import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

files = {
    "OVR": "../output/coefficients/se_coefficients.csv",
    "NA": "../output/coefficients/se_coefficients_NA.csv",
    "JP": "../output/coefficients/se_coefficients_JP.csv",
    "PAL": "../output/coefficients/se_coefficients_PAL.csv",
    "OTHER": "../output/coefficients/se_coefficients_OTHER.csv",
}

frames = {}
for region, path in files.items():
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = df.set_index("variable")
    frames[region] = df


# coefficient comparison table (exclude const)

coef_table = pd.DataFrame({
    region: frames[region]["coef"]
    for region in frames
}).drop("const", errors="ignore")


pct_table = np.exp(coef_table) - 1 # Add percent
pct_table = pct_table * 100

combined = coef_table.round(4).astype(str) + " (" + pct_table.round(2).astype(str) + "%)"
combined.to_csv("../output/coefficients/comparison_table.csv")
combined.to_excel("../output/coefficients/comparison_table.xlsx")

print("Comparison table saved.")


# p-value table (exclude const)
p_table = pd.DataFrame({
    region: frames[region]["p"]
    for region in frames
}).drop("const", errors="ignore")

p_table = p_table.round(4)
p_table.to_csv("../output/coefficients/pvalue_table.csv")

print("P-value table saved.")


# bar chart: coefficient comparison including const

all_coef_table = pd.DataFrame({
    region: frames[region]["coef"]
    for region in frames
})

all_coef_table.plot(kind="bar", figsize=(18, 8))
plt.title("Coefficient Comparison Across Regions")
plt.ylabel("Coefficient (log1p sales)")
plt.xlabel("Variable")
plt.axhline(0, color="black", linewidth=0.8)
plt.xticks(rotation=90)
plt.legend(title="Region", loc="upper right")
plt.tight_layout()
plt.savefig("../output/figures/coefficient_comparison.png", dpi=150, bbox_inches="tight")
plt.close()

print("Coefficient comparison chart saved.")


# heatmap (except const)
var_order = [
    "critic_c", "critic_missing", "year_c",
    "d_action", "d_sports", "d_misc", "d_adventure", "d_shooter",
    "d_racing", "d_simulation", "d_platform", "d_fighting",
    "d_strategy", "d_puzzle", "d_actionadventure", "d_visualnovel",
    "d_music", "d_other",
    "d_sega", "d_nintendohandheld", "d_nintendohome",
    "d_pcmac", "d_mshome", "d_sonyhandheld",
]

ordered_vars = [v for v in var_order if v in pct_table.index]

fig, ax = plt.subplots(figsize=(10, 15))

sns.heatmap(
    pct_table.loc[ordered_vars],
    annot=True,
    fmt=".1f",
    cmap="RdBu_r",
    center=0,
    cbar_kws={"label": "Effect vs baseline (%)"},
    ax=ax
)

ax.set_title("Percentage Heatmap by Variable & Region")

# reserve enough space at the bottom for  footnote
fig.subplots_adjust(bottom=0.14)

fig.text(# left-aligned footnote
    0.02, 0.07,
    "Baseline: Role-Playing Games on Sony Home, year 2010, critic score 7.3.\ncritic_c is centred at 7.3; year_c is centred at 2010. Values are percentages relative to that baseline.\nEmpty cells = variable dropped from that regional model because it had fewer than 30 observations in that region\n",
    ha="left", va="bottom", fontsize=9
)

plt.savefig("../output/figures/heatmap_pct_effect.png", dpi=150, bbox_inches="tight")
plt.close()

print("Heatmap saved.")


# forest plot: coefficient with confidence intervals (ovr and region)
groups = {
    "Quantitative": ["critic_c", "critic_missing", "year_c"],
    "Genre": [
        "d_action", "d_sports", "d_misc", "d_adventure", "d_shooter",
        "d_racing", "d_simulation", "d_platform", "d_fighting",
        "d_strategy", "d_puzzle", "d_actionadventure", "d_visualnovel",
        "d_music", "d_other",
    ],
    "Platform Category": [
        "d_sega", "d_nintendohandheld", "d_nintendohome",
        "d_pcmac", "d_mshome", "d_sonyhandheld",
    ],
}

group_colors = {
    "Quantitative": "tab:blue",
    "Genre": "tab:green",
    "Platform Category": "tab:red",
}

ordered_labels = []
ordered_coefs = []
ordered_errs = []
ordered_colors = []

for group_name, var_list in groups.items():
    for var in var_list:
        if var in frames["OVR"].index:
            ordered_labels.append(var)
            ordered_coefs.append(frames["OVR"].loc[var, "coef"])
            ordered_errs.append(frames["OVR"].loc[var, "std_err"])
            ordered_colors.append(group_colors[group_name])

plt.figure(figsize=(10, 14))
y_pos = np.arange(len(ordered_labels))

for i, (coef, err, color) in enumerate(zip(ordered_coefs, ordered_errs, ordered_colors)):
    plt.errorbar(
        coef, i,
        xerr=1.96 * err,
        fmt="o",
        color=color,
        capsize=4
    )

plt.yticks(y_pos, ordered_labels)
plt.axvline(0, color="black", linestyle="--", linewidth=1)
plt.xlabel("Coefficient (95% CI)")
plt.title("Overall Model: Coefficients by Variable Type")

legend_elements = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor=group_colors["Quantitative"],
           markersize=10, label="Quantitative (critic, year)"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor=group_colors["Genre"],
           markersize=10, label="Genre (vs Role-Playing)"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor=group_colors["Platform Category"],
           markersize=10, label="Platform Category (vs Sony Home)"),
]
plt.legend(handles=legend_elements, loc="upper right", framealpha=0.9)

plt.tight_layout()
plt.savefig("../output/figures/forest_OVR.png", dpi=150, bbox_inches="tight")
plt.close()
print("Overall Forest plot saved.")

for region in ["NA", "JP", "PAL", "OTHER"]: #for region, same theory
    ordered_labels = []
    ordered_coefs = []
    ordered_errs = []
    ordered_colors = []

    for group_name, var_list in groups.items():
        for var in var_list:
            if var in frames[region].index:
                ordered_labels.append(var)
                ordered_coefs.append(frames[region].loc[var, "coef"])
                ordered_errs.append(frames[region].loc[var, "std_err"])
                ordered_colors.append(group_colors[group_name])

    plt.figure(figsize=(10, 14))
    y_pos = np.arange(len(ordered_labels))

    for i, (coef, err, color) in enumerate(zip(ordered_coefs, ordered_errs, ordered_colors)):
        plt.errorbar(
            coef, i,
            xerr=1.96 * err,
            fmt="o",
            color=color,
            capsize=4
        )

    plt.yticks(y_pos, ordered_labels)
    plt.axvline(0, color="black", linestyle="--", linewidth=1)
    plt.xlabel("Coefficient (95% CI)")
    plt.title(f"{region} Model: Coefficients by Variable Type")

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=group_colors["Quantitative"],
               markersize=10, label="Quantitative (critic, year)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=group_colors["Genre"],
               markersize=10, label="Genre (vs Role-Playing)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=group_colors["Platform Category"],
               markersize=10, label="Platform Category (vs Sony Home)"),
    ]
    plt.legend(handles=legend_elements, loc="upper right", framealpha=0.9)

    plt.tight_layout()
    plt.savefig(f"../output/figures/forest_{region}.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Forest plot saved for {region}.")


# genre comparison
genre_vars = [
    "d_action", "d_sports", "d_misc", "d_adventure", "d_shooter",
    "d_racing", "d_simulation", "d_platform", "d_fighting",
    "d_strategy", "d_puzzle", "d_actionadventure", "d_visualnovel",
    "d_music", "d_other",
]

genre_pct = pct_table.loc[genre_vars].iloc[::-1]

fig, ax = plt.subplots(figsize=(12, 11))
genre_pct.plot(kind="barh", ax=ax)
ax.set_title("All Game Genres vs Role-Playing Games (RPG) by Region")
ax.set_xlabel("Effect vs RPG (%)")
ax.set_ylabel("Genre")
ax.axvline(0, color="black", linewidth=0.8)
ax.legend(title="Region", loc="lower right")

fig.subplots_adjust(bottom=0.13)
fig.text(
    0.12, 0.05,
    "Baseline: Role-Playing Games. Values show percent effect relative to RPG in each region.\nBars to the right = sells more than RPG; bars to the left = sells less.",
    ha="left", va="bottom", fontsize=9
)

fig.subplots_adjust(bottom=0.13)

plt.savefig("../output/figures/genre_comparison.png", dpi=150, bbox_inches="tight")
plt.close()

print("Genre chart saved.")

# Platform comparison

platform_vars = [
    "d_mshome", "d_nintendohome", "d_nintendohandheld",
    "d_sonyhandheld", "d_pcmac", "d_sega",
]

platform_pct = pct_table.loc[[v for v in platform_vars if v in pct_table.index]]

platform_pct.plot(kind="barh", figsize=(10, 6))
plt.title("Platform Effects vs Sony Home by Region")
plt.xlabel("Effect vs Sony Home (%)")
plt.ylabel("Platform Category")
plt.axvline(0, color="black", linewidth=0.8)
plt.legend(title="Region")
plt.tight_layout()
plt.savefig("../output/figures/platform_comparison.png", dpi=150, bbox_inches="tight")
plt.close()

print("Platform chart saved.")