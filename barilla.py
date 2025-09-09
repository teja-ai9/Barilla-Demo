import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Barilla Promo Strategy – Game Theory Simulator", layout="wide")

# ===== Category presets (ranges, defaults, grids) =====
# Ranges reflect typical EU grocery price bands & sensitivities:
# - Pasta (Core): high volume, more elastic; deeper promo room
# - Pasta (Premium): lower volume, less elastic; shallower promos
# - Sauces: mid elasticity; sometimes deeper TPRs than premium pasta
CATEGORY_PRESETS = {
    "Pasta (Core)": {
        "base_demand": (5000, 40000, 18000, 1000),   # units
        "wholesale":   (0.80, 2.20, 1.20, 0.05),     # € per pack
        "cogs":        (0.35, 1.10, 0.55, 0.05),     # € per pack
        "own_elast":   (-3.0, -0.9, -2.2, 0.1),      # more elastic
        "cross_elast": (0.2, 2.0, 0.9, 0.1),
        "disc_grid":   np.round(np.arange(0.00, 0.80, 0.05), 2)
    },
    "Pasta (Premium)": {
        "base_demand": (1500, 15000, 6000, 500),
        "wholesale":   (1.20, 3.20, 2.10, 0.05),
        "cogs":        (0.60, 2.20, 1.10, 0.05),
        "own_elast":   (-2.0, -0.6, -1.3, 0.1),      # less elastic
        "cross_elast": (0.1, 1.5, 0.6, 0.1),
        "disc_grid":   np.round(np.arange(0.00, 0.60, 0.05), 2)
    },
    "Sauces": {
        "base_demand": (3000, 25000, 10000, 500),
        "wholesale":   (1.00, 3.50, 1.80, 0.05),
        "cogs":        (0.50, 1.80, 0.80, 0.05),
        "own_elast":   (-2.6, -0.8, -1.8, 0.1),      # mid elasticity
        "cross_elast": (0.2, 2.0, 0.8, 0.1),
        "disc_grid":   np.round(np.arange(0.00, 0.90, 0.05), 2)
    }
}

# ========== SIDEBAR ==========
st.sidebar.markdown("### Barilla – Commercial Data Science Demo\n**Price & Promo Strategy (Game Theory)**\nby **Teja Bonthalakoti**")
st.sidebar.caption("Assumptions are illustrative and configurable for a live demo.")

st.sidebar.header("Category & Inputs")
category = st.sidebar.selectbox("Category", list(CATEGORY_PRESETS.keys()))
p = CATEGORY_PRESETS[category]

bd_lo, bd_hi, bd_def, bd_step = p["base_demand"]
base_demand = st.sidebar.slider("Base Demand (units)", bd_lo, bd_hi, bd_def, bd_step)

wp_lo, wp_hi, wp_def, wp_step = p["wholesale"]
wholesale_price = st.sidebar.slider("Barilla Wholesale Price (€)", wp_lo, wp_hi, wp_def, wp_step)

cg_lo, cg_hi, cg_def, cg_step = p["cogs"]
cogs = st.sidebar.slider("Barilla COGS (€)", cg_lo, cg_hi, cg_def, cg_step)

el_lo, el_hi, el_def, el_step = p["own_elast"]
elasticity = st.sidebar.slider("Own Price Elasticity", el_lo, el_hi, el_def, el_step)

xl_lo, xl_hi, xl_def, xl_step = p["cross_elast"]
cross_elasticity = st.sidebar.slider("Cross Elasticity", xl_lo, xl_hi, xl_def, xl_step)

# Strategy grids (symmetric)
barilla_disc_grid = p["disc_grid"]
pl_disc_grid = p["disc_grid"]

# ===== Asymmetry + simple penalty (Bol-style) =====
PL_BASE_FACTOR = 0.97      # Private Label base price is 3% cheaper than Barilla
MIN_PRICE = 0.01
GAP_THRESHOLD = 1.05       # 5% pricier
PENALTY = 0.70             # if Barilla > PL by 5%, multiply Barilla demand by 0.70

# ===== Simulation =====
rows = []
for bar_disc in barilla_disc_grid:
    for pl_disc in pl_disc_grid:
        # Baselines (tiny asymmetry ensures the 5% gap can trigger)
        bar_base = wholesale_price
        pl_base  = wholesale_price * PL_BASE_FACTOR

        # Transactional prices (simplified)
        bar_price = max(MIN_PRICE, bar_base - bar_disc)
        pl_price  = max(MIN_PRICE, pl_base  - pl_disc)

        # Simple price-gap penalty (one-way on Barilla)
        price_ratio = bar_price / pl_price if pl_price > 0 else 1.0
        penalty = PENALTY if price_ratio > GAP_THRESHOLD else 1.0

        # Demands (constant elasticities; normalized by each player's base)
        bar_demand = base_demand * (bar_price / bar_base) ** elasticity
        bar_demand *= (pl_price / pl_base) ** cross_elasticity
        bar_demand *= penalty

        pl_demand = base_demand * (pl_price / pl_base) ** elasticity
        pl_demand *= (bar_price / bar_base) ** cross_elasticity

        # Profits (illustrative; PL uses same COGS for simplicity)
        bar_profit = (bar_price - cogs) * bar_demand
        pl_profit  = (pl_price  - cogs) * pl_demand

        rows.append({
            "Category": category,
            "Barilla Disc (€)": bar_disc,
            "Private Label Disc (€)": pl_disc,
            "Barilla Profit (€)": bar_profit,
            "Private Label Profit (€)": pl_profit
        })

df = pd.DataFrame(rows)

# ===== Payoff matrices & Nash =====
barilla_matrix = df.pivot(index="Private Label Disc (€)", columns="Barilla Disc (€)", values="Barilla Profit (€)")
pl_matrix      = df.pivot(index="Private Label Disc (€)", columns="Barilla Disc (€)", values="Private Label Profit (€)")

barilla_best_response = barilla_matrix.idxmax(axis=1)
pl_best_response      = pl_matrix.idxmax(axis=0)

nash_points = []
for pl_disc in pl_disc_grid:
    if pl_disc not in barilla_best_response.index:
        continue
    bar_disc = barilla_best_response[pl_disc]
    if bar_disc not in pl_best_response.index:
        continue
    pl_best = pl_best_response[bar_disc]
    if np.isclose(pl_disc, pl_best):
        nash_points.append((pl_disc, bar_disc))

# ===== UI =====
st.title("Barilla Promo Strategy – Game Theory Simulator")

# KPIs
col1, col2 = st.columns(2)
with col1:
    best_row = df.loc[df["Barilla Profit (€)"].idxmax()]
    st.metric("Max Barilla Profit (€)", f"{best_row['Barilla Profit (€)']:.0f}")
with col2:
    best_bar_disc_avg = df.groupby("Barilla Disc (€)")["Barilla Profit (€)"].mean().idxmax()
    st.metric("Recommended Barilla Discount (avg)", f"€{best_bar_disc_avg:.2f}")

# Heatmap
st.subheader(f"Barilla Profit Payoff Matrix (€) — {category}")
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(barilla_matrix, annot=False, fmt=".0f", cmap="YlGnBu", ax=ax)

# Nash dots
for (pl_disc, bar_disc) in nash_points:
    x_idx = list(barilla_matrix.columns).index(bar_disc)
    y_idx = list(barilla_matrix.index).index(pl_disc)
    ax.plot(x_idx + 0.5, y_idx + 0.5, 'ro', markersize=10)

ax.set_xlabel("Barilla Trade Discount (€)")
ax.set_ylabel("Private Label Trade Discount (€)")
st.pyplot(fig)

# Nash list
st.subheader("Nash Equilibrium (Profit-Based)")
if nash_points:
    for pl_disc, bar_disc in nash_points:
        sub = df[(df["Private Label Disc (€)"] == pl_disc) & (df["Barilla Disc (€)"] == bar_disc)].iloc[0]
        st.info(f"Private Label Discount: €{pl_disc:.2f} | Barilla Discount: €{bar_disc:.2f} "
                f"→ Barilla Profit: €{sub['Barilla Profit (€)']:.0f}")
else:
    st.warning("No pure-strategy Nash equilibrium found under current assumptions.")

# Download
st.subheader("Download Simulation Data")
st.download_button(
    label="Download CSV",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="barilla_promo_simulation.csv",
    mime="text/csv"
)

# ===== Assumptions / Notes =====
st.caption(
    "This is a simplified demo for interview purposes:\n"
    "- Prices are modeled as effective transactional prices (wholesale minus trade discount), "
    "serving as a proxy for shelf price in this demo.\n"
    "- Private Label base price is set 3% cheaper than Barilla to reflect typical price gaps.\n"
    "- A simple fairness penalty applies when Barilla is >5% pricier than Private Label "
    "(Barilla demand × 0.70)\n"
    "- Private Label profit uses the same COGS slider for simplicity; in production, use category-specific COGS.\n"
    "- In practice, elasticities/costs are calibrated from price tests or MMM. \n"
)
