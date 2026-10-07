import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
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
    "d_aa", "d_indie", "d_unknown",
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
    "Baseline: AAA-published Role-Playing Games on Sony Home, year 2010, critic score 7.3.\n\ncritic_c is centred at 7.3; year_c is centred at 2010. Values are percentages relative to that baseline.\nEmpty cells = variable dropped from that regional model because it had fewer than 30 observations in that region\n",
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
    "Publisher Class": [
        "d_aa", "d_indie", "d_unknown",
    ],
}

group_colors = {
    "Quantitative": "tab:blue",
    "Genre": "tab:green",
    "Platform Category": "tab:red",
    "Publisher Class": "tab:orange",
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
        Line2D([0], [0], marker="o", color="w", markerfacecolor=group_colors["Publisher Class"],
               markersize=10, label="Publisher Class (vs AAA)"),
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
        Line2D([0], [0], marker="o", color="w", markerfacecolor=group_colors["Publisher Class"],
               markersize=10, label="Publisher Class (vs AAA)"),
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

# residual map

RESID_FILES = {
    "OVR":   "../output/residuals/residuals.csv",
    "NA":    "../output/residuals/residuals_NA.csv",
    "JP":    "../output/residuals/residuals_JP.csv",
    "PAL":   "../output/residuals/residuals_PAL.csv",
    "OTHER": "../output/residuals/residuals_OTHER.csv",
}

resid_frames = {}
for region, path in RESID_FILES.items():
    if os.path.exists(path):
        resid_frames[region] = pd.read_csv(path, encoding="utf-8-sig")


def annotate_stack(ax, rows, side, color):
    """
    Draw leader lines from each labelled game to the outside margin,
    then draw the label text ON TOP with an opaque white bbox so lines
    never appear to cross through labels.
    """
    n = len(rows)
    if n == 0:
        return

    ylim = ax.get_ylim()
    y_range = ylim[1] - ylim[0]

    x_pts = rows.index.to_numpy().astype(float)
    y_pts = rows["resid"].to_numpy().astype(float)
    y_pts_frac = (y_pts - ylim[0]) / y_range

    # larger gap => visibly spread labels even when residuals are clustered
    gap_frac = 0.07

    if side == "right":
        order = np.argsort(-y_pts_frac)
    else:
        order = np.argsort(y_pts_frac)

    y_labels_frac = y_pts_frac.copy()
    for i in range(1, n):
        prev, cur = order[i - 1], order[i]
        if side == "right":
            if y_labels_frac[prev] - y_labels_frac[cur] < gap_frac:
                y_labels_frac[cur] = y_labels_frac[prev] - gap_frac
        else:
            if y_labels_frac[cur] - y_labels_frac[prev] < gap_frac:
                y_labels_frac[cur] = y_labels_frac[prev] + gap_frac

    y_labels_frac = np.clip(y_labels_frac, 0.02, 0.98)

    x_label_frac = 1.02 if side == "right" else -0.02
    ha = "left" if side == "right" else "right"

    # ---- Step 1: leader lines (drawn first, low zorder) ----
    for i in range(n):
        ax.annotate(
            "",
            xy=(x_pts[i], y_pts[i]),                 # arrow tip → data point
            xytext=(x_label_frac, y_labels_frac[i]), # start → outside margin
            textcoords="axes fraction",
            arrowprops=dict(
                arrowstyle="-",
                color=color,
                linewidth=0.6,
                alpha=0.7,
                shrinkA=0, shrinkB=3,
            ),
            annotation_clip=False,
            zorder=1,
        )

    # ---- Step 2: label text with opaque bbox (drawn on top) ----
    for i in range(n):
        row = rows.iloc[i]
        ax.text(
            x_label_frac, y_labels_frac[i],
            f"{row['title']} ({row['console']}, {row['year']})",
            transform=ax.transAxes,
            fontsize=7, va="center", ha=ha,
            bbox=dict(boxstyle="round,pad=0.18", fc="white",
                      ec="gray", alpha=1.0, linewidth=0.4),
            zorder=10,
            clip_on=False,
        )

for region, rdf in resid_frames.items():
    rdf = rdf.sort_values("resid").reset_index(drop=True)
    n = len(rdf)
    x = np.arange(n)

    fig, ax = plt.subplots(figsize=(16, 7))

    ax.scatter(x, rdf["resid"], s=3, alpha=0.35, color="steelblue",
               edgecolors="none", label="Game")

    ax.axhline(0, color="black", linewidth=0.9)

    top = rdf.nlargest(5, "resid").copy()
    ax.scatter(top.index, top["resid"], s=45, color="crimson",
               edgecolors="black", linewidths=0.6, zorder=5,
               label="Top 5 under-predicted")

    bot = rdf.nsmallest(5, "resid").copy()
    ax.scatter(bot.index, bot["resid"], s=45, color="darkorange",
               edgecolors="black", linewidths=0.6, zorder=5,
               label="Top 5 over-predicted")

    annotate_stack(ax, top, side="right", color="crimson")
    annotate_stack(ax, bot, side="left",  color="darkorange")

    ax.set_xlabel("Game (sorted by residual)")
    ax.set_ylabel("Residual (log1p sales)")
    ax.set_title(f"{region} Model — Residual Map for All Games (n = {n})")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.set_xlim(-5, n + 5)

    fig.subplots_adjust(bottom=0.14)
    fig.text(
        0.02, 0.03,
        "Each dot = one game-platform release. The left tail = most over-predicted, the right tail = most under-predicted.\n"
        "Residual = actual log1p(sales) − predicted log1p(sales)",
        ha="left", va="bottom", fontsize=8.5
    )

    plt.savefig(f"../output/figures/residual_map_{region}.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Residual map saved for {region}.")