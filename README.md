# Electricity Access in Southern Africa: Advanced, Multivariate, Temporal and Spatial Analysis

Notebook and code for **CSC 6732 Data Visualisation, Assignment 3** (University of Zambia, MSc Data Science).
It extends the Assignment 1 dataset audit and the Assignment 4 dashboard
([zambia-electricity-dashboard](https://github.com/lwandokasuba/zambia-electricity-dashboard)).

**Open the analysis:** [`assignment3_analysis.ipynb`](assignment3_analysis.ipynb). It is executed, so GitHub renders every output and figure.

| Task | What the notebook does | Figures |
|---|---|---|
| 1. Multivariate | Spearman correlations computed four ways (pooled, between countries, within countries, detrended year-to-year changes); a 4-variable bubble plot; parallel coordinates | 1–3 |
| 2. Temporal | Raw vs moving averages MAₜ = (1/k)Σxₜ₋ᵢ (k = 3, 5); OLS trends; piecewise-linear structural breaks; leave-one-out anomaly detection; why seasonality cannot be studied with annual data | 4–6 |
| 3. Geospatial | Rate choropleth vs proportional-symbol count map; rank reversal between rate and count; regional count over time | 7–10 |
| 4. Alternative representations | Choropleth vs ranked bar (ranking task); one multi-line chart vs small multiples (trajectory task) | 7, 11; 12, 5 |
| 5. Findings | Eight findings, each split into observation and interpretation | all |
| Limitations | Methodological limitations of data and methods | – |

## Run it

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pytest                      # tests for the analysis functions
.venv/bin/jupyter nbconvert --to notebook --execute --inplace assignment3_analysis.ipynb
```

Static map export uses Plotly + Kaleido, which needs a local Chrome/Chromium install.

## Structure

```
assignment3_analysis.ipynb   the analysis (narrative, code, outputs)
src/data.py                  load + clean the workbook, derived variables (shared with Assignment 4)
src/analysis.py              correlation views, trends, breakpoints, anomalies, rates vs counts
src/style.py                 shared palette and figure helpers
figures/                     every figure as PNG + numbers.json (values quoted in the report)
tests/                       pytest checks for the analysis functions
data/                        World Bank WDI extract (9 countries x 2000-2024)
```

## Data

World Bank, World Development Indicators (EG.ELC.ACCS.ZS, EG.FEC.RNEW.ZS, NY.GDP.PCAP.CD,
SP.URB.TOTL.IN.ZS, SP.DYN.LE00.IN, SP.POP.TOTL), indicators updated 13 July 2026, licensed
[CC BY 4.0](https://datacatalog.worldbank.org/public-licenses#cc-by).
