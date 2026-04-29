import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from networkx.algorithms.bipartite.basic import color
from sympy.abc import alpha

#Globa plot style
sns.set_theme(style = "whitegrid", palette = "muted")
plt.rcParams["figure.figsize"]=(12,6)
plt.rcParams["axes.titlesize"]=14
plt.rcParams["axes.titleweight"]="bold"


#Load the datasize directly from source
url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies-financials/master/data/constituents-financials.csv"
df = pd.read_csv(url)


#First look
print("Shape: ", df.shape)
print("\n Columns\n", df.columns.tolist())
print("\n Data types:\n", df.dtypes)
print("\n Print first 5 rows:\n",df.head())

# rename columns to snake case
df.columns = ["symbol", "name", "sector", "price", "pe_ratio", "dividend_yield", "eps", "week_52_low", "week_52_high", "market_cap", "ebitda", "price_sales", "price_book", "sec_filings"]



# drop column "sec_filings" because it's just url and of no use'
df.drop(columns = ["sec_filings"], inplace = True)


#missing values
print("--missing values per column--")
print(df.isnull().sum())
print(f"\n total rows:{len(df)}")
print(f"\n rows with any missing value:{df.isnull().any(axis=1).sum()}")


# nonsensical values -
print("\n Negative EPS - Companies losing money")
print(f"\n Count:{(df['eps']<0).sum()}")

print("\n Negative or zero PE ratio")
print(f"\n Count: {(df['pe_ratio']<=0).sum()}")

print("\n descriptive stats on key numerical values")
print(df[["price", "pe_ratio","eps","market_cap","ebitda"]].describe().round(2))

print("\n Unique sectors")
print(df["sector"].value_counts())

print(f"\n Companies with missing sector: {df["sector"].isnull().sum()}")
print(df[df["sector"].isnull()][["symbol" ,"name"]].head(10))

# ── Step 1: Map granular sub-sectors → 11 broad GICS sectors ──────────────
gics_map = {
    # Information Technology
    "Semiconductors": "Information Technology",
    "Systems Software": "Information Technology",
    "Application Software": "Information Technology",
    "IT Consulting & Other Services": "Information Technology",
    "Technology Hardware, Storage & Peripherals": "Information Technology",
    "Electronic Equipment & Instruments": "Information Technology",
    "Internet Services & Infrastructure": "Information Technology",
    "Data Processing & Outsourced Services": "Information Technology",
    "Communications Equipment": "Information Technology",
    "Semiconductor Materials & Equipment": "Information Technology",
    "Electronic Components": "Information Technology",
    "Electronic Manufacturing Services": "Information Technology",
    "Technology Distributors": "Information Technology",
    # Health Care
    "Health Care Equipment": "Health Care",
    "Pharmaceuticals": "Health Care",
    "Biotechnology": "Health Care",
    "Health Care Services": "Health Care",
    "Health Care Facilities": "Health Care",
    "Managed Health Care": "Health Care",
    "Life Sciences Tools & Services": "Health Care",
    "Health Care Supplies": "Health Care",
    "Health Care Technology": "Health Care",
    "Health Care Distributors": "Health Care",
    "Drug Retail": "Health Care",
    # Financials
    "Diversified Banks": "Financials",
    "Investment Banking & Brokerage": "Financials",
    "Asset Management & Custody Banks": "Financials",
    "Insurance": "Financials",
    "Property & Casualty Insurance": "Financials",
    "Life & Health Insurance": "Financials",
    "Consumer Finance": "Financials",
    "Regional Banks": "Financials",
    "Financial Exchanges & Data": "Financials",
    "Mortgage REITs": "Financials",
    "Diversified Financial Services": "Financials",
    "Multi-Sector Holdings": "Financials",
    "Reinsurance": "Financials",
    "Thrifts & Mortgage Finance": "Financials",
    "Transaction & Payment Processing Services": "Financials",
    "Insurance Brokers": "Financials",
    "Multi-line Insurance": "Financials",
    # Industrials
    "Industrial Machinery & Supplies & Components": "Industrials",
    "Aerospace & Defense": "Industrials",
    "Air Freight & Logistics": "Industrials",
    "Airlines": "Industrials",
    "Building Products": "Industrials",
    "Construction & Engineering": "Industrials",
    "Electrical Components & Equipment": "Industrials",
    "Environmental & Facilities Services": "Industrials",
    "Human Resource & Employment Services": "Industrials",
    "Office Services & Supplies": "Industrials",
    "Passenger Ground Transportation": "Industrials",
    "Professional Services": "Industrials",
    "Research & Consulting Services": "Industrials",
    "Security & Alarm Services": "Industrials",
    "Trading Companies & Distributors": "Industrials",
    "Trucking": "Industrials",
    "Waste Management": "Industrials",
    "Marine Transportation": "Industrials",
    "Farm & Construction Machinery": "Industrials",
    "Construction Machinery & Heavy Transportation Equipment": "Industrials",
    "Diversified Support Services": "Industrials",
    "Passenger Airlines": "Industrials",
    "Rail Transportation": "Industrials",
    "Industrial Conglomerates": "Industrials",
    "Agricultural & Farm Machinery": "Industrials",
    "Cargo Ground Transportation": "Industrials",
    "Heavy Electrical Equipment": "Industrials",
    # Consumer Discretionary
    "Automotive Retail": "Consumer Discretionary",
    "Casinos & Gaming": "Consumer Discretionary",
    "Home Improvement Retail": "Consumer Discretionary",
    "Hotels, Resorts & Cruise Lines": "Consumer Discretionary",
    "Household Durables": "Consumer Discretionary",
    "Internet & Direct Marketing Retail": "Consumer Discretionary",
    "Leisure Products": "Consumer Discretionary",
    "Restaurants": "Consumer Discretionary",
    "Specialty Retail": "Consumer Discretionary",
    "Textiles, Apparel & Luxury Goods": "Consumer Discretionary",
    "Auto Components": "Consumer Discretionary",
    "Automobile Manufacturers": "Consumer Discretionary",
    "Distributors": "Consumer Discretionary",
    "Footwear": "Consumer Discretionary",
    "General Merchandise Stores": "Consumer Discretionary",
    "Home Furnishings": "Consumer Discretionary",
    "Homebuilding": "Consumer Discretionary",
    "Movies & Entertainment": "Consumer Discretionary",
    "Publishing": "Consumer Discretionary",
    "Apparel, Accessories & Luxury Goods": "Consumer Discretionary",
    "Apparel Retail": "Consumer Discretionary",
    "Broadline Retail": "Consumer Discretionary",
    "Automotive Parts & Equipment": "Consumer Discretionary",
    "Interactive Home Entertainment": "Consumer Discretionary",
    "Other Specialty Retail": "Consumer Discretionary",
    "Computer & Electronics Retail": "Consumer Discretionary",
    "Consumer Electronics": "Consumer Discretionary",
    # Consumer Staples
    "Beverages": "Consumer Staples",
    "Food & Staples Retailing": "Consumer Staples",
    "Food Products": "Consumer Staples",
    "Household Products": "Consumer Staples",
    "Personal Products": "Consumer Staples",
    "Tobacco": "Consumer Staples",
    "Hypermarkets & Super Centers": "Consumer Staples",
    "Packaged Foods & Meats":                     "Consumer Staples",
    "Consumer Staples Merchandise Retail":         "Consumer Staples",
    "Soft Drinks & Non-alcoholic Beverages":       "Consumer Staples",
    "Agricultural Products & Services":            "Consumer Staples",
    "Distillers & Vintners":                       "Consumer Staples",
    "Food Retail":                                 "Consumer Staples",
    "Brewers":                                     "Consumer Staples",
    "Food Distributors":                           "Consumer Staples",
    "Personal Care Products":                      "Consumer Staples",
    # Energy
    "Integrated Oil & Gas": "Energy",
    "Oil & Gas Exploration & Production": "Energy",
    "Oil & Gas Refining & Marketing": "Energy",
    "Oil & Gas Equipment & Services": "Energy",
    "Oil & Gas Storage & Transportation": "Energy",
    "Coal & Consumable Fuels": "Energy",
    # Utilities
    "Electric Utilities": "Utilities",
    "Multi-Utilities": "Utilities",
    "Gas Utilities": "Utilities",
    "Water Utilities": "Utilities",
    "Independent Power Producers & Energy Traders": "Utilities",
    "Renewable Electricity": "Utilities",
    # Real Estate
    "Diversified REITs": "Real Estate",
    "Industrial REITs": "Real Estate",
    "Office REITs": "Real Estate",
    "Residential REITs": "Real Estate",
    "Retail REITs": "Real Estate",
    "Specialized REITs": "Real Estate",
    "Real Estate Management & Development": "Real Estate",
    "Timber REITs": "Real Estate",
    "Hotel & Resort REITs": "Real Estate",
    "Health Care REITs": "Real Estate",
    "Multi-Family Residential REITs":              "Real Estate",
    "Telecom Tower REITs":                         "Real Estate",
    "Self-Storage REITs":                          "Real Estate",
    "Data Center REITs":                           "Real Estate",
    "Real Estate Services":                        "Real Estate",
    "Other Specialized REITs":                     "Real Estate",
    "Single-Family Residential REITs":             "Real Estate",
    # Communication Services
    "Interactive Media & Services": "Communication Services",
    "Cable & Satellite": "Communication Services",
    "Integrated Telecommunication Services": "Communication Services",
    "Wireless Telecommunication Services": "Communication Services",
    "Broadcasting": "Communication Services",
    "Advertising": "Communication Services",

    # Materials
    "Specialty Chemicals":                         "Materials",
    "Fertilizers & Agricultural Chemicals":        "Materials",
    "Paper & Plastic Packaging Products & Materials": "Materials",
    "Industrial Gases":                            "Materials",
    "Metal, Glass & Plastic Containers":           "Materials",
    "Commodity Chemicals":                         "Materials",
    "Chemicals": "Materials",
    "Construction Materials": "Materials",
    "Containers & Packaging": "Materials",
    "Metals & Mining": "Materials",
    "Paper & Forest Products": "Materials",
    "Steel": "Materials",
    "Aluminum": "Materials",
    "Copper": "Materials",
    "Diversified Metals & Mining": "Materials",
    "Gold": "Materials",
    "Silver": "Materials",
}

df["broad_sector"] = df["sector"].map(gics_map).fillna("Other")

print("-----Broad sector distribution----")
print(df["broad_sector"].value_counts())
print(f"\n Unmapped to 'Other' :{(df['broad_sector']=='Other').sum()} companies")


#convert market cap and ebidta to billions
df["market_cap_b"] = df["market_cap"]/ 1e9
df["ebitda_b"] = df["ebitda"]/ 1e9


# add derived columns useful for EDA
df["price_range_pct"] = ((df["week_52_high"]-df["week_52_low"])/ df["week_52_high"] *100).round(2)
df["pays_dividend"] = df["dividend_yield"].notna() & (df["dividend_yield"]>0)


# flag outliers (not dropping)
pe_cap = df["pe_ratio"].quantile(0.95)
eps_cap = df["eps"].quantile(0.95)
mc_cap = df["market_cap_b"].quantile(0.95)

df["pe_capped"] = df["pe_ratio"].clip(upper=pe_cap)
df["eps_capped"] = df["eps"].clip(lower = df["eps"].quantile(0.05), upper = eps_cap)
df["mc_capped"] = df["market_cap_b"].clip(upper = mc_cap)


#final dataset summary
print("\n-----Final cleaned dataset-----")
print(f"Shape: {df.shape}")
print(f"Columns :{df.columns.tolist()}")
print(f"Companies paying dividends: {df['pays_dividend'].sum()}")
print(f"\n Companies with negative EPS: {(df['eps'] <0).sum()}")
print(f"\n P/E 95th percentile cap: {pe_cap:.1f}")
print(f"EPS 95th percentile cap: {eps_cap:.2f}")
print(f"Market cap 95th percentile cap: {mc_cap:.1f}")


print("\n----Cleaned descriptive stats----")
print(df[["price", "pe_capped","eps_capped", "market_cap_b", "dividend_yield"]].describe().round(2))


other_sectors = df[df["broad_sector"]=="Other"]["sector"].value_counts()
print("---subsectors currently unmapped falling into 'Others'---")
print(other_sectors.to_string())

cols_of_interest = ["price", "pe_ratio","eps","dividend_yield","market_cap_b", "ebitda_b","price_sales","price_book"]

stats = df[cols_of_interest].describe().T
stats["skewness"] = df[cols_of_interest].skew()
stats["median"] = df[cols_of_interest].median()
stats["iqr"] = df[cols_of_interest].quantile(0.75) - df[cols_of_interest].quantile(0.25)

print("----descriptive statistics (full dataset)----")
print(stats[["count","mean","median","std","skewness","iqr","min","max"]].round(2).to_string())

# per sector descriptive statsfor key metrics
print("\n---Median EPS by sector---")
print(df.groupby("broad_sector")["eps"].median().sort_values(ascending=False).round(2).to_string())

print("\n---Median P/E by sector---")
print(df.groupby("broad_sector")["pe_ratio"].median().sort_values(ascending=False).round(2).to_string())

print("\n---Median market cap $b by sector---")
print(df.groupby("broad_sector")["market_cap_b"].median().sort_values(ascending=False).round(2).to_string())


# descriptive plots

fig, axes = plt.subplots(2,2,figsize=(14,10))
fig.suptitle("Distribution Analysis - S&P 500 Financial Metrics", fontsize = 16, fontweight = "bold", y=1.01)

# Plot 1 eps distribution (capped)
ax=axes[0,0]
sns.histplot(df["eps_capped"].dropna(), bins =40, kde = True, ax=ax, color="#4C72B0")
ax.axvline(df["eps_capped"].median(), color = "crimson", linestyle="--",linewidth=1.5, label =f'Median: ${df["eps_capped"].median():.2f}' )
ax.axvline(df["eps_capped"].mean(), color = "orange", linestyle = "--", linewidth = 1.5, label = f'Mean: ${df["eps_capped"].mean():.2f}')
ax.set_title("EPS Distribution (95th pct capped)")
ax.set_xlabel("Earnings Per Share ($)")
ax.set_ylabel("Count")
ax.legend(fontsize=9)

# Plot 2 P/E ratio distribution (capped)
ax= axes[0,1]
sns.histplot(df["pe_capped"].dropna(), bins=40, kde=True, ax=ax, color ="#55A868")
ax.axvline(df["pe_capped"].median(), color="crimson", linestyle="--", linewidth=1.5, label=f'Median: {df["pe_capped"].median():.1f}')
ax.axvline(df["pe_capped"].mean(), color = "orange", linestyle = "--", linewidth=1.5, label=f'Mean: {df["pe_capped"].mean():.1f}')
ax.set_title("P/E Ratio Distribution (95th pct capped)")
ax.set_xlabel("Price / Earnings Ratio")
ax.set_ylabel("Count")
ax.legend(fontsize=9)


# Plot 3 market cap distribution (log scale)
ax=axes[1,0]
mc_clean = df["market_cap_b"].dropna()
sns.histplot(mc_clean, bins= 50, kde=True,ax=ax, color="#C44E52")
ax.axvline(mc_clean.median(), color="crimson", linestyle="--", linewidth=1.5, label=f'Median: ${mc_clean.median():.0f}B')
ax.axvline(mc_clean.mean(), color="orange", linestyle="--", linewidth=1.5, label=f'mean: ${mc_clean.mean():.0f}B')
ax.set_title("Market Cap Distribution ($B)")
ax.set_xlabel("Market Cap ($B)")
ax.set_ylabel("Count")
ax.set_xlim(0,600) # zoom in - extreme outliers (Apple, etc) noted separately
ax.legend(fontsize=9)
ax.annotate("Tail extends to $3,573B\n(outliers not shown)", xy=(560, ax.get_ylim()[1]*0.85),
            fontsize=8, color="grey", ha="right")


# Plot 4 Dividend yield distribution
ax=axes[1,1]
dy_clean=df["dividend_yield"].dropna()
dy_clean = dy_clean[dy_clean>0] # paying 0 dividend
sns.histplot(dy_clean, bins=35, kde=True, ax=ax, color="#8172B2")
ax.axvline(dy_clean.median(), color="crimson", linestyle="--", linewidth=1.5, label=f'Median: {dy_clean.median()*100:.2f}')
ax.axvline(dy_clean.mean(), color="orange", linestyle="--", linewidth=1.5, label=f'Mean: {dy_clean.mean()*100:.2f}')
ax.set_title("Dividend Yield Distribution (payers only)")
ax.set_xlabel("Dividend Yield")
ax.set_ylabel("Count")
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x*100:.1f}%"))
ax.legend(fontsize=9)


plt.tight_layout()
plt.savefig("distributions.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: distributions.png")


# Relationship analysis

# Correlation heatmap

fig,ax= plt.subplots(figsize=(10,8))

corr_cols=["price","pe_capped","eps_capped","dividend_yield","market_cap_b", "ebitda_b", "price_sales","price_book"]
corr_labels=["Price","P/E","EPS","Dividend Yield","Mkt Cap","EBITDA","P/S","P/B"]

corr_matrix=df[corr_cols].corr()
corr_matrix.index=corr_labels
corr_matrix.columns=corr_labels

mask=np.triu(np.ones_like(corr_matrix, dtype=bool)) #hide upper triangle

sns.heatmap(
    corr_matrix, mask=mask,annot=True, fmt=".2f",
    cmap="RdYlGn", center=0, vmin=-1,vmax=1,
    linewidth=0.5,ax=ax, annot_kws={"size":10}
)
ax.set_title("Correlation Matrix - S&P 500Financial Metrics", fontsize=14, fontweight="bold", pad=15)
plt.tight_layout()
plt.savefig("phase5a_correlation.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: phse5a_correlation.png")


# eps vs market cap (colored by sector)
fig, ax = plt.subplots(figsize=(13,7))

sector_sorted = df["broad_sector"].value_counts().index.tolist()
palette = sns.color_palette("tab10", n_colors=len(sector_sorted))
color_map = dict(zip(sector_sorted,palette))

for sector, grp in df.groupby("broad_sector"):
    grp_clean=grp.dropna(subset=["eps_capped", "mc_capped"])
    ax.scatter(
        grp_clean["eps_capped"], grp_clean["mc_capped"],
        label=sector, color=color_map[sector],
        alpha=0.7, edgecolors = "white", linewidth=0.4, s=60
    )

ax.set_title("EPS vs Market Cap - colored by sector", fontsize=14, fontweight="bold")
ax.set_xlabel("Earnings Per Share - capped at 95th pct ($)")
ax.set_ylabel("Market Cap - capped at 95th pct ($B)")
ax.legend(bbox_to_anchor=(1.01,1), loc="upper left", fontsize=8, frameon=False)
plt.tight_layout()
plt.savefig("phase5b_eps_vs_mktcap.png",dpi=150, bbox_inches="tight")
plt.show()
print("Saved: phase5b_eps_vs_mktcap.png")


# p/e vs eps - does higher earnings = lower valuation?

fig, ax = plt.subplots(figsize=(13,7))

for sector, grp in df.groupby("broad_sector"):
    grp_clean = grp.dropna(subset=["eps_capped", "pe_capped"])
    ax.scatter(
        grp_clean["eps_capped"], grp_clean["pe_capped"],
        label=sector, color=color_map[sector],
        alpha=0.7, edgecolors="white", linewidths=0.4, s=60
    )

# overlay a regression line across the full dataset
clean = df[["eps_capped", "pe_capped"]].dropna()
m, b = np.polyfit(clean["eps_capped"], clean["pe_capped"],1)
x_line= np.linspace(clean["eps_capped"].min(), clean["eps_capped"].max(), 200)
ax.plot(x_line, m*x_line+b, color="black", linewidth=1.5,
        linestyle="--",label=f"Trend (slope={m:.2f})")

ax.set_title("P/E Ratio vs EPS - does high earnings compress valuation?", fontsize=14, fontweight="bold")
ax.set_xlabel("Earnings Per Share - capped at 95th pct ($)")
ax.set_ylabel("P/E Ratio - capped at 95th pct")
ax.legend(bbox_to_anchor=(1.01,1), loc = "upper left", fontsize=8,frameon=False)
plt.tight_layout()
plt.savefig("phase5c_pe_vs_eps.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: phase5c_pe_vs_eps.png")

# price/book vs p/e - growth vsvalue quadrant map
fig, ax = plt.subplots(figsize=(13,7))

for sector, grp in df.groupby("broad_sector"):
    grp_clean=grp.dropna(subset=["price_book", "pe_capped"])
    grp_clean=grp_clean[grp_clean["price_book"]<= 30]  # cap p/e for readablity
    ax.scatter(
        grp_clean["price_book"],grp_clean["pe_capped"],
        label=sector, color=color_map[sector],
        alpha=0.7, edgecolors="white", linewidth=0.4, s=60
    )

#quadrant line at medians
pb_med = df["price_book"].median()
pe_med = df["pe_capped"].median()
ax.axvline(pb_med, color="gray", linestyle=":", linewidth=1.2)
ax.axhline(pe_med,color="gray", linestyle=":", linewidth=1.2)

#quadrant labels
ax.text(0.5, pe_med +1, "Value", fontsize=9, color="gray")
ax.text(15, pe_med +1, "Premium",fontsize=9, color="gray")
ax.text(0.5, pe_med -5, "Deep Value", fontsize=9, color="gray")
ax.text(15, pe_med -5, "High-earn/Low-P/E", fontsize=9, color="gray")

ax.set_title("Price/Book vs P/E - value vs growth quadrant map", fontsize=14, fontweight="bold")
ax.set_xlabel("Price/Book Ratio (capped at 30x)")
ax.set_ylabel("P/E Ratio - capped at 95th pct")
ax.legend(bbox_to_anchor=(1.01,1), loc="upper left", fontsize=8, frameon=False)
plt.tight_layout()
plt.savefig("phase5d_pb_vs_pe.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: phase5d_pb_vs_pe.png")

# quick correlation summary
print("\n--- Top correlations with market cap---")
print(corr_matrix["Mkt Cap"].sort_values(ascending=False).round(3).to_string())

print("\n---top correlations with P/E---")
print(corr_matrix["P/E"].sort_values(ascending=False).round(3).to_string())

print("\n---Top correlations with EPS---")
print(corr_matrix["EPS"].sort_values(ascending=False).round(3).to_string())


# sector deep dive
sector_order = (df.groupby("broad_sector")["market_cap_b"]
                .median()
                .sort_values(ascending=False)
                .index.tolist())

# --- box plots by sector---
fig,ax=plt.subplots(figsize=(14,6))
sns.boxplot(
    data=df, x="broad_sector", y="eps_capped",
    order=sector_order, palette="tab10",
    width=0.55, fliersize=3, ax=ax
)
ax.axhline(df["eps_capped"].median(), color="black", linestyle="--",
           linewidth=1.2, label=f'Overall median: ${df["eps_capped"].median():.2f}')
ax.set_title("EPS Distribution by Sector (95th pct capped)", fontsize=14, fontweight="bold")
ax.set_xlabel("")
ax.set_ylabel("Earnings Per Share ($)")
ax.tick_params(axis="x", rotation=35)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig("phase6a_eps_by_sector.png",dpi=150, bbox_inches="tight")
plt.show()
print("Saved: phase6a_eps_by_sector.png")

#---box plots - p/e by sector---
fig, ax=plt.subplots(figsize=(14,6))
sns.boxplot(
    data=df, x="broad_sector", y="pe_capped",
    order=sector_order, palette="tab10",
    width=0.55, fliersize=3, ax=ax
)
ax.axhline(df["pe_capped"].median(), color="black", linestyle="--",
           linewidth=1.2, label=f'Overall median: {df["pe_capped"].median():.1f}x')
ax.set_title("P/E Ratio Distribution by Sector (95th pct capped)", fontsize=14, fontweight="bold")
ax.set_xlabel("")
ax.set_ylabel("Price / Earning Ratio")
ax.tick_params(axis="x", rotation=35)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig("phase6b_pe_by_sector.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: phase6b_pe_by_sector.png")


# grouped bar - median p/e vs median eps side by side

sector_summary = df.groupby("broad_sector").agg(
    median_pe=("pe_capped", "median"),
    median_eps=("eps_capped", "median"),
    median_mc=("market_cap_b", "median"),
    median_dy=("dividend_yield", "median"),
    pct_dividend=("pays_dividend", "mean"),
    count=("symbol", "count")
).loc[sector_order].round(2)

print("\n---Sector Summary Table---")
print(sector_summary.to_string())

x=np.arange(len(sector_order))
width=0.38

fig,ax1 = plt.subplots(figsize=(14,6))
ax2=ax1.twinx()

bars1=ax1.bar(x-width/2, sector_summary["median_pe"],
              width,label="Median P/E", color="#4C72B0", alpha=0.85)
bars2=ax2.bar(x+width/2,sector_summary["median_eps"],
              width, label="Median EPS ($)", color="#C44E52", alpha=0.85)
ax1.set_ylabel("Median P/E Ratio", color="#4C72B0")
ax2.set_ylabel("Median EPS ($)", color="#C44E52")
ax1.tick_params(axis="y", labelcolor="#4C72B0")
ax2.tick_params(axis="y", labelcolor="#C44E52")
ax1.set_xticks(x)
ax1.set_xticklabels(sector_order, rotation=35, ha="right")
ax1.set_title("Median P/E vs Median EPS by Sector", fontsize=14, fontweight="bold")

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=9 )
plt.tight_layout()
plt.savefig("phase6c_pe_vs_eps_sector.png")
plt.show()
print("Saved: phase6c_pe_vs_eps_sector.png")


# dividend payers % + median yield by sector

fig, ax1 = plt.subplots(figsize=(14,6))
ax2=ax1.twinx()

pct_div = sector_summary["pct_dividend"]*100
med_dy = sector_summary["median_dy"]*100

ax1.bar(x,pct_div, width=0.55, color="#55A868", alpha=0.8, label="% paying dividend")
ax2.plot(x, med_dy, color="#DD8452", marker="o",linewidth=2,
         markersize=7, label="Median yield (%)")

ax1.set_ylabel("% of sector paying dividend", color="#55A868")
ax2.set_ylabel("Median Dividend yield(%)", color="#DD8452")
ax1.tick_params(axis="y", labelcolor="#55A868")
ax2.tick_params(axis="y", labelcolor="#DD8452")
ax1.set_xticks(x)
ax1.set_xticklabels(sector_order, rotation=35, ha="right")
ax1.set_title("dividend Profile by Sector - payers % and median yield", fontsize=14, fontweight="bold")

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 =ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2,loc="lower right", fontsize=9)
plt.tight_layout()
plt.savefig("phase6d_dividends_by_sector.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: phase6d_dividends_by_sector.png")

# 52 week price range % by sector (volatility roxy)
fig, ax=plt.subplots(figsize=(14,6))
sns.boxplot(
    data=df, x="broad_sector", y="price_range_pct",
    order=sector_order, palette="tab10",
    width=0.55, fliersize=3, ax=ax
)

ax.axhline(df["price_range_pct"].median(), color="black", linestyle="--",
           linewidth=1.2, label=f'Overall median: {df["price_range_pct"].median():.1f}%')
ax.set_title("52-Week Price Range % by Sector (volatility proxy)", fontsize=14, fontweight="bold")
ax.set_xlabel("")
ax.set_ylabel("(52W High - 52W Low)/ 52W Low * 100")
ax.tick_params(axis="x", rotation=35)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig("phase6e_volatility_by _sector", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: phase6e_volatility_by _sector")