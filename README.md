# Square Enix Sales Strategy: A Regional Regression Analysis of VGChartz 2000–2020

A linear regression analysis of historical video game sales data to advise Square Enix on platform, genre, regional production and distribution strategy.

## Overview
This project uses VGChartz sales data (2000–2020) to estimate how critic score, release year, genre, and platform category relate to game sales. 
Separate models are estimated for overall sales and for four regions: North America (NA), Japan (JP), PAL (Europe, Australia, New Zealand, and other PAL territories), and Other. 
The goal is to provide evidence-based guidance for Square Enix on where to release games and which genres perform best by region.

## Data
- **Source**: VGChartz (publicly available video game sales estimates)
- **Period:** 2000–2020
- **Unit of analysis:** Game-platform release
- **Rows:** ~15,842 (after cleaning, varies by region)
- **Key variables:**
  - `total_sales`, `na_sales`, `jp_sales`, `pal_sales`, `other_sales` (millions of copies)
  - `critic_score` (0–10 scale)
  - `genre`, `console`, `publisher`, `developer`
  - `release_date`

Raw data files are located in `data/`. A trimmed dataset is also included for robustness checks.
The trimmed dataset has cut top 1% and bottom 1% data by `total_sales`.

## Methodology

### Cleaning and preparation
- Filtered to 2000–2020.
- Removed aggregate platforms (`Series`, `All`) and digital storefronts (`PSN`, `XBL`, `VC`, `WW`).
- Grouped platforms into strategic categories: *Sony Home*, *Microsoft Home*, *Nintendo Home*, *Nintendo Handheld/Hybrid*, *Sony Handheld*, *PC/Mac*, *Sega*.
- Collapsed rare genres into `Other`.
- Created `critic_clean` (median-filled for missing scores) and `critic_missing` flag.
- Centered critic score at 7.3 and year at 2010.
- Log-transformed sales using `log1p` to handle skew and zero values.

### Model specification
- **Dependent variable (y)**: `log_region_sales` for overall or regional sales.
- **Independent variables (x)**:
  - Quantitative: `critic_c` centred at 7.3 (median), `critic_missing`, `year_c` (centred at 2010)
  - Genre dummies (reference: *Role-Playing*)
  - Platform category dummies (reference: *Sony Home*)
- **Estimation**: OLS with HC3 robust standard errors.
- **Validation**: VIF diagnostics (all < 5), condition numbers (< 110).
- **Regional models**: Separate regressions for `NA`, `JP`, `PAL`, and `Other`. Categories with fewer than 30 observations in a region were dropped to avoid unstable estimates.  

Therefore, the baseline of this model is a *Role-Playing Game* on *Sony Home*, released in year 2010, with a critic score of 7.3.

## Why Sony Home and Role-Playing are the reference dummy categories
In a regression with dummy variables, one category from each group must be dropped to avoid perfect collinearity (the dummy variable trap). 
The dropped category is the reference or *baseline*. 
Every coefficient for the remaining categories is interpreted as a difference relative to that baseline.
The choice of reference determines how every coefficient in the report is read. 
Sony Home and Role-Playing were chosen because they match Square Enix’s core business and the client’s decision context.

### Sony Home was chosen as the platform reference because
- *PlayStation* home consoles were Square Enix’s primary platform through the period covered by the data, particularly for *Final Fantasy Series* and *Kingdom Hearts*.
- *Sony Home* accounts for the largest share of total sales in the dataset and the highest average sales per title, making it the most stable baseline.

### Role-Playing was chosen as the genre reference because
- Square Enix is fundamentally an RPG company. Its flagship franchises *Final Fantasy*, *Dragon Quest*, and *Kingdom Hearts*, are RPGs or RPG-adjacent.
- Even though it does not hold the largest share of genres, it is the most decision-relevant framing for the client to set genre coefficient  as “compared to an RPG”.

## How to Run

1. **Install dependencies**
    ```bash
   pip install -r requirements.txt
   ```
   
2. **Run the model**  
From the `scripts/` folder:
    ```bash
   python run_model.py
    ```
   Choose `OVR` for overall sales, `ALLREG` for all four region sales respectively, or region code (`JP`, `NA`, `PAL`, `OTHER`)for a specific region.    
  

3. **Generate tables and figures**  
From the `scripts/` folder:
    ```bash
   python table_and_graph_making.py
    ```
    This reads the coefficient CSVs created by `run_model.py` and produces comparison tables and plots.

## Outputs
- `output/coefficients/` - coefficient tables, comparison tables, p-value tables
- `output/vif/` - Variance Inflation Factor (VIF) diagnostics
- `output/figures/` - heatmap, forest plots, genre comparison, platform comparison, coefficient comparison

## Key Findings
- **Critic score** has a positive association with sales *(11.8%)*, strongest in `NA` *(8.2%)* and `PAL` *(6.0%)*, weak in `JP` *(1.3%)*
- **Missing critic score** is a large negative signal overall *(-23.3%)*, but small in `JP` *(-1.8%)*
- Effect of **Year** is practically negligible *(-0.1%)* and confounded with time-on-market. Older games have had more time to accumulate lifetime sales, which *may* slightly inflate their totals
- For **Game Genre**, in `JP`, RPG dominates, almost every other genre sells less. In `NA` and `PAL`, Shooter, Sports, and Racing sell more than RPG
- For **Platform**, Sony Home is the strongest platform overall. Microsoft Home is statistically equal to Sony Home in `NA`, while Nintendo Handheld/Hybrid is equal to Sony Home in `JP`. PC/Mac is undercounted by VGChartz and should not be interpreted as weak

## Limitations
- Available data from 2000 to 2020 only
- Results are associations, not causal effects
- VGChartz undercounts digital, PC, mobile, and indie sales
- IP strength and marketing budgets are not in the data
- Regional samples differ due to data availability
- Model explains 14.2% to 25.0% of variation in log sales (R<sup>2</sup> = 0.142 in `JP`, R<sup>2</sup> = 0.250 in `OVR`), the rest is unobserved

## Requirements
- Python 3.9+
- Packages: pandas, numpy, statsmodels, matplotlib, seaborn, openpyxl

## Author
**North Ice** *(northice-kq)*

## License
For academic and portfolio use only. Data sourced from VGChartz, redistribution may be subject to VGChartz terms.