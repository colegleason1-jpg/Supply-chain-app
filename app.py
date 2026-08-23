import pandas as pd
import pulp as pl
import streamlit as st

st.set_page_config(
    page_title="Supply Chain Resilience Engine", page_icon="⚙️", layout="wide"
)

st.title("Supply Chain Resilience & Capital Allocation Engine")
st.markdown(
    "*Enterprise Edition: Dynamic Operations Research ILP Solver with Custom"
    " Variables & Sensitivity Analysis.*"
)
st.markdown("---")

# --- SIDEBAR: DYNAMIC CONFIGURATION & CONTROLS ---
st.sidebar.header("Executive Control Panel")

total_budget = st.sidebar.slider(
    "Total Budget Cap ($)", 100000, 1000000, 600000, step=25000
)
enable_synergy = st.sidebar.checkbox(
    "Enable Multi-Node Synergy Bundle ($50k Discount)", value=True
)

st.sidebar.markdown("---")
st.sidebar.subheader("Custom Portfolio Nodes")
st.sidebar.markdown(
    "Add, rename, or configure your supply chain intervention targets below:"
)

# Initialize session state for custom nodes if not present
if "nodes_df" not in st.session_state:
  st.session_state.nodes_df = pd.DataFrame([
      {
          "Node Name": "Logistics_Hub",
          "Action": "Alternative port routing & freight contracts",
          "Cost": 200000,
          "Risk Reduction (%)": 20.0,
      },
      {
          "Node Name": "Primary_Warehouse",
          "Action": "Primary warehouse resilience hardening",
          "Cost": 250000,
          "Risk Reduction (%)": 22.0,
      },
      {
          "Node Name": "Backup_Supplier",
          "Action": "Upgrade backup logistics & capacity",
          "Cost": 150000,
          "Risk Reduction (%)": 15.0,
      },
      {
          "Node Name": "Tier_1_Suppliers",
          "Action": "Onboard redundant regional secondary suppliers",
          "Cost": 100000,
          "Risk Reduction (%)": 12.0,
      },
  ])

# Allow user to edit existing nodes or add new ones directly in data editor
edited_nodes = st.sidebar.data_editor(
    st.session_state.nodes_df, num_rows="dynamic", use_container_width=True
)
st.session_state.nodes_df = edited_nodes

# --- OPTIMIZATION ENGINE (ILP SOLVER) ---
nodes = edited_nodes["Node Name"].tolist()
costs = dict(zip(nodes, edited_nodes["Cost"]))
risks = dict(zip(nodes, edited_nodes["Risk Reduction (%)"]))
actions = dict(zip(nodes, edited_nodes["Action"]))

prob = pl.LpProblem("Supply_Chain_Optimization", pl.LpMaximize)
x = {n: pl.LpVariable(f"x_{n}", cat="Binary") for n in nodes}
bundle_active = pl.LpVariable("bundle_active", cat="Binary")

# Objective: Maximize total risk reduction
prob += pl.lpSum(risks[n] * x[n] for n in nodes)

# Cost Constraint with Synergy Discount
if enable_synergy and len(nodes) >= 3:
  prob += (
      pl.lpSum(costs[n] * x[n] for n in nodes)
      - 50000 * bundle_active
      <= total_budget
  )
  # Bundle logic: active only if at least 3 nodes are selected
  prob += pl.lpSum(x[n] for n in nodes) >= 3 * bundle_active
else:
  prob += pl.lpSum(costs[n] * x[n] for n in nodes) <= total_budget

prob.solve(pl.PULP_CBC_CMD(msg=False))

# Calculate results
selected_nodes = [n for n in nodes if pl.value(x[n]) == 1]
total_cost = sum(costs[n] for n in selected_nodes)
is_bundle_active = (
    bool(pl.value(bundle_active) == 1)
    if (enable_synergy and len(nodes) >= 3)
    else False
)
if is_bundle_active:
  total_cost -= 50000

total_risk_drop = sum(risks[n] for n in selected_nodes)
baseline_risk = 82.5
optimized_risk = max(0.0, baseline_risk - total_risk_drop)

# --- MAIN DASHBOARD DISPLAY ---
col1, col2, col3 = st.columns(3)
col1.metric("Baseline System Risk", f"{baseline_risk:.2f}%")
col2.metric(
    "Optimized System Risk",
    f"{optimized_risk:.2f}%",
    delta=f"-{total_risk_drop:.2f} pts",
    delta_color="inverse",
)
col3.metric(
    "Capital Deployed",
    f"${total_cost:,.2f}",
    delta=f"Cap: ${total_budget:,.2f}",
    delta_color="off",
)

if is_bundle_active:
  st.success("Multi-node bundle synergy active: $50,000 discount applied.")

tab1, tab2 = st.tabs(
    ["Optimal Portfolio", "CFO Budget Sensitivity & Visual Sweep"]
)

with tab1:
  st.subheader("Globally Optimal Capital Allocation Portfolio")
  if selected_nodes:
    portfolio_data = []
    for idx, n in enumerate(selected_nodes, 1):
      portfolio_data.append({
          "Priority": f"Rank #{idx}",
          "Target Node": n,
          "Strategic Action": actions.get(n, "Custom Intervention"),
          "Capital Required": f"${costs[n]:,.2f}",
          "Risk Reduction (%)": f"{risks[n]}%",
      })
    df_portfolio = pd.DataFrame(portfolio_data)
    st.dataframe(df_portfolio, use_container_width=True)

    # CSV Export Button
    csv_data = df_portfolio.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Optimal Portfolio CSV",
        data=csv_data,
        file_name="optimal_supply_chain_portfolio.csv",
        mime="text/csv",
    )
  else:
    st.warning(
        "No nodes selected. Increase your budget cap or adjust node costs."
    )

with tab2:
  st.subheader("Budget Sensitivity Analysis & Marginal ROI")
  st.markdown(
      "Evaluating how portfolio selections scale across variable budget"
      " thresholds:"
  )

  sweep_results = []
  budget_range = range(
      max(50000, total_budget - 100000), total_budget + 150000, 25000
  )

  for b in budget_range:
    sub_prob = pl.LpProblem(f"Sub_{b}", pl.LpMaximize)
    sub_x = {n: pl.LpVariable(f"sx_{n}", cat="Binary") for n in nodes}
    sub_bundle = pl.LpVariable(f"sb_{b}", cat="Binary")

    sub_prob += pl.lpSum(risks[n] * sub_x[n] for n in nodes)
    if enable_synergy and len(nodes) >= 3:
      sub_prob += (
          pl.lpSum(costs[n] * sub_x[n] for n in nodes)
          - 50000 * sub_bundle
          <= b
      )
      sub_prob += pl.lpSum(sub_x[n] for n in nodes) >= 3 * sub_bundle
    else:
      sub_prob += pl.lpSum(costs[n] * sub_x[n] for n in nodes) <= b

    sub_prob.solve(pl.PULP_CBC_CMD(msg=False))
    s_nodes = [n for n in nodes if pl.value(sub_x[n]) == 1]
    s_cost = sum(costs[n] for n in s_nodes)
    s_bundle_on = (
        bool(pl.value(sub_bundle) == 1)
        if (enable_synergy and len(nodes) >= 3)
        else False
    )
    if s_bundle_on:
      s_cost -= 50000
    s_risk = sum(risks[n] for n in s_nodes)

    sweep_results.append({
        "Budget Cap": f"${b:,.2f}",
        "Actual Spent": f"${s_cost:,.2f}",
        "Risk Drop (pts)": s_risk,
        "Bundle Active?": "Yes" if s_bundle_on else "No",
        "Selected Portfolio": ", ".join(s_nodes) if s_nodes else "None",
        "RawBudget": b,
        "RawRisk": s_risk,
    })

  df_sweep = pd.DataFrame(sweep_results)

  # Visual Chart for Executive Presentation
  st.line_chart(
      df_sweep.set_index("RawBudget")[["RawRisk"]],
      use_container_width=True,
  )
  st.caption("Figure: System Risk Reduction Curve across Budget Variations.")

  st.dataframe(
      df_sweep[[
          "Budget Cap",
          "Actual Spent",
          "Risk Drop (pts)",
          "Bundle Active?",
          "Selected Portfolio",
      ]],
      use_container_width=True,
  )