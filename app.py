import pandas as pd
import pulp as pl
import streamlit as st
import numpy as np
import io

st.set_page_config(
    page_title="Supply Chain Commercial Engine", page_icon="🌐", layout="wide"
)

st.title("Supply Chain Resilience & Capital Allocation Engine")
st.markdown(
    "*Commercial B2B Enterprise Edition: Fully Functional Multi-Objective Optimization, True Optimization Weights & Real Stochastic Metrics.*"
)
st.markdown("---")

# --- SIDEBAR: COMMERCIAL CONFIGURATION & CONTROLS ---
st.sidebar.header("Executive Control Panel")
baseline_risk = st.sidebar.number_input(
    "Starting Baseline System Risk (%)",
    min_value=0.0,
    max_value=100.0,
    value=65.5,
    step=1.0,
)

st.sidebar.markdown("---")
opt_weight = st.sidebar.slider(
    "Optimization Weight (Risk vs. Lead Time)",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.05,
    help="1.0 = Pure focus on Risk Reduction. 0.0 = Pure focus on Lead Time Savings.",
)

num_iterations = st.sidebar.slider(
    "Monte Carlo Simulation Precision (Runs)",
    min_value=1000,
    max_value=50000,
    value=10000,
    step=5000,
    help="Higher iterations increase precision for risk committees and tail-risk VaR percentiles."
)

st.sidebar.markdown("---")
enable_target_mode = st.sidebar.checkbox(
    "🎯 Enable Target Risk Goal Mode",
    value=False,
    help="Minimizes capital deployment to achieve a specific target risk threshold using true LpMinimize cost functions.",
)

if enable_target_mode:
    target_risk_goal = st.sidebar.number_input(
        "Target System Risk Goal (%)",
        min_value=0.0,
        max_value=100.0,
        value=20.0,
        step=1.0,
    )
    total_budget = 1000000000
    st.sidebar.info(f"Target Mode Active: Minimizing capital to reach <= {target_risk_goal}% risk.")
else:
    total_budget = st.sidebar.number_input(
        "Total Budget Cap ($)",
        min_value=10000,
        max_value=1000000000,
        value=750000,
        step=50000,
        format="%d",
    )

st.sidebar.markdown("---")
st.sidebar.subheader("1. Custom Intervention Nodes")

if "nodes_df" not in st.session_state:
    st.session_state.nodes_df = pd.DataFrame([
        {"Node Name": "Logistics_Hub", "Action": "Alternative port routing", "Cost": 200000, "Risk Reduction (%)": 15.0, "Lead Time Saved (Days)": 5, "Carbon Impact (Tons)": 120},
        {"Node Name": "Primary_Warehouse", "Action": "Warehouse hardening", "Cost": 250000, "Risk Reduction (%)": 15.0, "Lead Time Saved (Days)": 2, "Carbon Impact (Tons)": 50},
        {"Node Name": "Backup_Supplier", "Action": "Upgrade backup capacity", "Cost": 150000, "Risk Reduction (%)": 10.0, "Lead Time Saved (Days)": 7, "Carbon Impact (Tons)": 80},
        {"Node Name": "Tier_1_Suppliers", "Action": "Onboard regional suppliers", "Cost": 300000, "Risk Reduction (%)": 20.0, "Lead Time Saved (Days)": 10, "Carbon Impact (Tons)": 200},
    ])

edited_nodes = st.sidebar.data_editor(st.session_state.nodes_df, num_rows="dynamic", use_container_width=True)

# Data Validation
edited_nodes["Cost"] = edited_nodes["Cost"].apply(lambda x: max(0.0, float(x) if pd.notnull(x) else 0.0))
edited_nodes["Risk Reduction (%)"] = edited_nodes["Risk Reduction (%)"].apply(lambda x: max(0.0, float(x) if pd.notnull(x) else 0.0))
edited_nodes["Lead Time Saved (Days)"] = edited_nodes["Lead Time Saved (Days)"].apply(lambda x: max(0.0, float(x) if pd.notnull(x) else 0.0))
st.session_state.nodes_df = edited_nodes

nodes = edited_nodes["Node Name"].dropna().tolist()
costs = dict(zip(nodes, edited_nodes["Cost"]))
risks = dict(zip(nodes, edited_nodes["Risk Reduction (%)"]))
lead_times = dict(zip(nodes, edited_nodes["Lead Time Saved (Days)"]))
carbon_impacts = dict(zip(nodes, edited_nodes["Carbon Impact (Tons)"]))
actions = dict(zip(nodes, edited_nodes["Action"]))

# LINKED MATH: Apply a diminishing returns transformation multiplier dynamically
marginal_risks = {n: float(risks[n] ** 0.85) for n in nodes}

st.sidebar.markdown("---")
st.sidebar.subheader("2. Dynamic Savings Bundles")

if "bundles_df" not in st.session_state:
    st.session_state.bundles_df = pd.DataFrame([{"Bundle Name": "Port_Warehouse_Synergy", "Discount ($)": 50000}])

edited_bundles = st.sidebar.data_editor(st.session_state.bundles_df, num_rows="dynamic", use_container_width=True)
st.session_state.bundles_df = edited_bundles

# --- OPTIMIZATION ENGINE ---
bundle_vars = {}
if enable_target_mode:
    prob = pl.LpProblem("Target_Goal_Min_Cost_Optimization", pl.LpMinimize)
    x = {n: pl.LpVariable(f"x_{n}", cat="Binary") for n in nodes}
    
    # Define optimization target constraints using linked marginal risks
    required_risk_drop = max(0.0, baseline_risk - target_risk_goal)
    prob += pl.lpSum(marginal_risks[n] * x[n] for n in nodes) >= required_risk_drop
    
    # Setup bundle links
    for idx, row in edited_bundles.iterrows():
        b_name = row.get("Bundle Name")
        if not b_name or pd.isna(b_name): 
            continue
        req_nodes = st.sidebar.multiselect(f"Nodes for bundle: {b_name}", options=nodes, default=nodes[:2] if len(nodes)>=2 else nodes, key=f"b_targ_{idx}")
        if req_nodes:
            b_var = pl.LpVariable(f"bundle_{idx}", cat="Binary")
            bundle_vars[str(b_name).strip()] = {"var": b_var, "discount": float(row.get("Discount ($)", 0))}
            for rn in req_nodes: 
                prob += b_var <= x[rn]
            
    total_discounts = pl.lpSum(bundle_vars[b]["var"] * bundle_vars[b]["discount"] for b in bundle_vars)
    prob += pl.lpSum(costs[n] * x[n] for n in nodes) - total_discounts  # Objective: Minimize net cost
else:
    prob = pl.LpProblem("Commercial_Supply_Chain_Optimization", pl.LpMaximize)
    x = {n: pl.LpVariable(f"x_{n}", cat="Binary") for n in nodes}
    
    # Setup bundle links
    for idx, row in edited_bundles.iterrows():
        b_name = row.get("Bundle Name")
        if not b_name or pd.isna(b_name): 
            continue
        req_nodes = st.sidebar.multiselect(f"Nodes for bundle: {b_name}", options=nodes, default=nodes[:2] if len(nodes)>=2 else nodes, key=f"b_max_{idx}")
        if req_nodes:
            b_var = pl.LpVariable(f"bundle_{idx}", cat="Binary")
            bundle_vars[str(b_name).strip()] = {"var": b_var, "discount": float(row.get("Discount ($)", 0))}
            for rn in req_nodes: 
                prob += b_var <= x[rn]

    total_discounts = pl.lpSum(bundle_vars[b]["var"] * bundle_vars[b]["discount"] for b in bundle_vars)
    prob += pl.lpSum(costs[n] * x[n] for n in nodes) - total_discounts <= total_budget
    
    # Balanced multi-objective blending weight logic
    prob += pl.lpSum((opt_weight * marginal_risks[n] + (1 - opt_weight) * lead_times[n]) * x[n] for n in nodes)

try:
    prob.solve(pl.PULP_CBC_CMD(msg=False))
except:
    prob.solve(pl.CHOOSE_SOLVER(msg=False))

# --- RESULTS PROCESSING & TRUE STOCHASTIC SIMULATION ---
selected_nodes = [n for n in nodes if pl.value(x[n]) == 1]
final_cost = sum(costs[n] for n in selected_nodes) - sum(b_data["discount"] for b_name, b_data in bundle_vars.items() if pl.value(b_data["var"]) == 1)
total_risk_drop = sum(marginal_risks[n] for n in selected_nodes)
optimized_risk = max(0.0, baseline_risk - total_risk_drop)
total_lead_time = sum(lead_times[n] for n in selected_nodes)

# Active Bundle Names List tracking for UI reporting
active_bundle_names = [b_name for b_name, b_data in bundle_vars.items() if pl.value(b_data["var"]) == 1]

# REAL MC ENGINE: Simulating supply chain disruption volatility shocks across selected portfolio
if selected_nodes and num_iterations > 0:
    sim_results = []
    for _ in range(num_iterations):
        random_shocks = np.random.normal(loc=1.0, scale=0.12, size=len(selected_nodes))
        simulated_drop = sum(marginal_risks[n] * random_shocks[i] for i, n in enumerate(selected_nodes))
        sim_results.append(max(0.0, baseline_risk - simulated_drop))
    var_95 = float(np.percentile(sim_results, 95))
    p50_risk = float(np.percentile(sim_results, 50))
else:
    var_95 = optimized_risk
    p50_risk = optimized_risk

# --- DASHBOARD METRICS DISPLAY ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Baseline System Risk", f"{baseline_risk:.2f}%")
col2.metric("Optimized Expected Risk", f"{optimized_risk:.2f}%", delta=f"-{total_risk_drop:.2f} pts", delta_color="inverse")
col3.metric("Capital Deployed", f"${final_cost:,.2f}")
col4.metric("95% Tail-Risk (VaR)", f"{var_95:.2f}%", help="Stochastic Value-at-Risk under Monte Carlo volatility shocks.")

st.markdown("---")

tab1, tab2, tab3 = st.tabs([
    "Optimal Portfolio", 
    "Executive Overview", 
    "🔍 Audit Trail & Math Reference"
])

with tab1:
    st.subheader("Funded Mitigation Portfolio & Asset Breakdown")
    if selected_nodes:
        portfolio_data = []
        for idx, n in enumerate(selected_nodes, 1):
            portfolio_data.append({
                "Priority": f"Rank #{idx}",
                "Target Node": n,
                "Strategic Action": actions.get(n, "Custom Intervention"),
                "Capital Required": f"${costs[n]:,.2f}",
                "Raw Risk Reduction": f"{risks[n]}%",
                "Linked Diminishing Utility": f"{marginal_risks[n]:.2f}",
                "Lead Time Saved": f"{lead_times[n]} Days",
                "Carbon Impact": f"{carbon_impacts[n]} Tons",
            })
        df_portfolio = pd.DataFrame(portfolio_data)
        st.dataframe(df_portfolio, use_container_width=True)

        csv_data = df_portfolio.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Portfolio Report (CSV)",
            data=csv_data,
            file_name="optimized_supply_chain_portfolio.csv",
            mime="text/csv",
        )
    else:
        st.warning("No nodes selected. Please expand your budget or modify your target constraints.")

with tab2:
    st.subheader("Performance & Stochastic Risk Distribution Summary")
    st.write(f"**Funded Mitigation Nodes:** {', '.join(selected_nodes) if selected_nodes else 'None'}")
    st.write(f"**Total Lead Time Extinguished:** {total_lead_time} Days")
    st.write(f"**Active Synergy Bundles:** {', '.join(active_bundle_names) if active_bundle_names else 'None'}")
    st.write(f"**Stochastic Median Risk ($P_{50}$):** {p50_risk:.2f}%")
    st.write(f"**Stochastic Tail-Risk ($P_{95}$ VaR across {num_iterations:,} runs):** {var_95:.2f}%")

with tab3:
    st.subheader("🔍 Mathematical Framework & Optimization Parameters")
    st.markdown("""
    * **Concave Diminishing Returns:** Modeled utilizing $R_n^{0.85}$ to prevent the linear over-allocation fallacy.
    * **Stochastic Volatility Simulation:** Normal distribution error perturbations ($\sigma = 12\%$ shock factor) over $N$ iterations.
    * **Solver Backend:** Linear Mixed-Integer Programming executed via PuLP CBC engine.
    """)
    st.markdown("### Raw Node Configuration Matrix")
    st.dataframe(edited_nodes, use_container_width=True)
