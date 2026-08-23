import pandas as pd
import pulp as pl
import streamlit as st

st.set_page_config(
    page_title="Supply Chain Commercial Engine", page_icon="🌐", layout="wide"
)

st.title("Supply Chain Resilience & Capital Allocation Engine")
st.markdown(
    "*Commercial Enterprise Edition: Multi-Objective Optimization, Dynamic"
    " Bundles & Custom Variables.*"
)
st.markdown("---")

# --- SIDEBAR: COMMERCIAL CONFIGURATION & CONTROLS ---
st.sidebar.header("Executive Control Panel")

# Editable Baseline Risk
baseline_risk = st.sidebar.number_input(
    "Starting Baseline System Risk (%)",
    min_value=0.0,
    max_value=100.0,
    value=65.5,
    step=1.0,
)

# Target Goal Mode Controls
st.sidebar.markdown("---")
enable_target_mode = st.sidebar.checkbox(
    "🎯 Enable Target Risk Goal Mode",
    value=False,
    help=(
        "Finds the minimum budget and optimal nodes needed to hit a specific"
        " target risk."
    ),
)

if enable_target_mode:
  target_risk_goal = st.sidebar.number_input(
      "Target System Risk Goal (%)",
      min_value=0.0,
      max_value=100.0,
      value=20.0,
      step=1.0,
  )
  total_budget = 20000000
  st.sidebar.info(
      f"Target Mode Active: Solving for minimum capital required to reach"
      f" <= {target_risk_goal}% risk."
  )
else:
  total_budget = st.sidebar.number_input(
      "Total Budget Cap ($)",
      min_value=10000,
      max_value=20000000,
      value=750000,
      step=25000,
      format="%d",
  )

st.sidebar.markdown("---")
st.sidebar.subheader("1. Custom Intervention Nodes")
st.sidebar.markdown(
    "Configure your supply chain nodes, costs, risk reduction, lead times,"
    " and ESG impact:"
)

# Initialize session state for custom nodes
if "nodes_df" not in st.session_state:
  st.session_state.nodes_df = pd.DataFrame([
      {
          "Node Name": "Logistics_Hub",
          "Action": "Alternative port routing & freight contracts",
          "Cost": 200000,
          "Risk Reduction (%)": 15.0,
          "Lead Time Saved (Days)": 5,
          "Carbon Impact (Tons)": 120,
      },
      {
          "Node Name": "Primary_Warehouse",
          "Action": "Primary warehouse resilience hardening",
          "Cost": 250000,
          "Risk Reduction (%)": 15.0,
          "Lead Time Saved (Days)": 2,
          "Carbon Impact (Tons)": 50,
      },
      {
          "Node Name": "Backup_Supplier",
          "Action": "Upgrade backup logistics & capacity",
          "Cost": 150000,
          "Risk Reduction (%)": 10.0,
          "Lead Time Saved (Days)": 7,
          "Carbon Impact (Tons)": 80,
      },
      {
          "Node Name": "Tier_1_Suppliers",
          "Action": "Onboard redundant regional secondary suppliers",
          "Cost": 300000,
          "Risk Reduction (%)": 20.0,
          "Lead Time Saved (Days)": 10,
          "Carbon Impact (Tons)": 200,
      },
  ])

edited_nodes = st.sidebar.data_editor(
    st.session_state.nodes_df, num_rows="dynamic", use_container_width=True
)
st.session_state.nodes_df = edited_nodes

st.sidebar.markdown("---")
st.sidebar.subheader("2. Dynamic Savings Bundles")
st.sidebar.markdown(
    "Define multi-node discount packages offered by vendors or logistics"
    " partners:"
)

if "bundles_df" not in st.session_state:
  st.session_state.bundles_df = pd.DataFrame([
      {
          "Bundle Name": "Port_Warehouse_Synergy",
          "Required Nodes (Comma Separated)": "Logistics_Hub, Primary_Warehouse",
          "Discount ($)": 50000,
      },
      {
          "Bundle Name": "Total_Resilience_Package",
          "Required Nodes (Comma Separated)": (
              "Logistics_Hub, Backup_Supplier, Tier_1_Suppliers"
          ),
          "Discount ($)": 90000,
      },
  ])

edited_bundles = st.sidebar.data_editor(
    st.session_state.bundles_df, num_rows="dynamic", use_container_width=True
)
st.session_state.bundles_df = edited_bundles

# --- DATA EXTRACTION & SAFE CLEANING ---
nodes = edited_nodes["Node Name"].dropna().tolist()
costs = dict(zip(nodes, edited_nodes["Cost"]))
risks = dict(zip(nodes, edited_nodes["Risk Reduction (%)"]))
lead_times = dict(zip(nodes, edited_nodes["Lead Time Saved (Days)"]))
carbon_impacts = dict(zip(nodes, edited_nodes["Carbon Impact (Tons)"]))
actions = dict(zip(nodes, edited_nodes["Action"]))

# --- OPTIMIZATION ENGINE (ADVANCED ILP SOLVER) ---
if enable_target_mode:
  prob = pl.LpProblem("Target_Goal_Optimization", pl.LpMaximize)
  x = {n: pl.LpVariable(f"x_{n}", cat="Binary") for n in nodes}

  bundle_vars = {}
  for idx, row in edited_bundles.iterrows():
    b_name = row.get("Bundle Name")
    if not b_name or pd.isna(b_name) or str(b_name).strip().lower() == "none":
      continue
    try:
      b_discount = float(row.get("Discount ($)", 0))
    except (ValueError, TypeError):
      b_discount = 0.0

    req_nodes_raw = str(row.get("Required Nodes (Comma Separated)", ""))
    req_nodes = [
        n.strip() for n in req_nodes_raw.split(",") if n.strip() in nodes
    ]

    if str(b_name).strip() and req_nodes:
      b_var = pl.LpVariable(f"bundle_{idx}", cat="Binary")
      bundle_vars[str(b_name).strip()] = {
          "var": b_var,
          "discount": b_discount,
          "nodes": req_nodes,
      }
      for rn in req_nodes:
        prob += b_var <= x[rn]

  total_discounts = pl.lpSum(
      bundle_vars[b]["var"] * bundle_vars[b]["discount"] for b in bundle_vars
  )
  prob += pl.lpSum(costs[n] * x[n] for n in nodes) - total_discounts

  required_risk_drop = max(0.0, baseline_risk - target_risk_goal)
  prob += (
      pl.lpSum(risks[n] * x[n] for n in nodes) >= required_risk_drop
  )

else:
  prob = pl.LpProblem("Commercial_Supply_Chain_Optimization", pl.LpMaximize)
  x = {n: pl.LpVariable(f"x_{n}", cat="Binary") for n in nodes}

  bundle_vars = {}
  for idx, row in edited_bundles.iterrows():
    b_name = row.get("Bundle Name")
    if not b_name or pd.isna(b_name) or str(b_name).strip().lower() == "none":
      continue
    try:
      b_discount = float(row.get("Discount ($)", 0))
    except (ValueError, TypeError):
      b_discount = 0.0

    req_nodes_raw = str(row.get("Required Nodes (Comma Separated)", ""))
    req_nodes = [
        n.strip() for n in req_nodes_raw.split(",") if n.strip() in nodes
    ]

    if str(b_name).strip() and req_nodes:
      b_var = pl.LpVariable(f"bundle_{idx}", cat="Binary")
      bundle_vars[str(b_name).strip()] = {
          "var": b_var,
          "discount": b_discount,
          "nodes": req_nodes,
      }
      for rn in req_nodes:
        prob += b_var <= x[rn]

  prob += pl.lpSum(
      risks[n] * x[n] + (0.5 * lead_times[n] * x[n]) for n in nodes
  )
  total_discounts = pl.lpSum(
      bundle_vars[b]["var"] * bundle_vars[b]["discount"] for b in bundle_vars
  )
  prob += (
      pl.lpSum(costs[n] * x[n] for n in nodes) - total_discounts <= total_budget
  )

prob.solve(pl.PULP_CBC_CMD(msg=False))

# --- RESULTS COMPUTATION ---
selected_nodes = [n for n in nodes if pl.value(x[n]) == 1]
base_cost_spent = sum(costs[n] for n in selected_nodes)
applied_bundle_discounts = 0
active_bundle_names = []

for b_name, b_data in bundle_vars.items():
  if pl.value(b_data["var"]) == 1:
    applied_bundle_discounts += b_data["discount"]
    active_bundle_names.append(b_name)

final_cost_spent = base_cost_spent - applied_bundle_discounts
total_risk_drop = sum(risks[n] for n in selected_nodes)

# Mathematically clamp risk reduction so it cannot exceed baseline risk
total_risk_drop = min(baseline_risk, total_risk_drop)
optimized_risk = max(0.0, baseline_risk - total_risk_drop)

total_lead_time_saved = sum(lead_times[n] for n in selected_nodes)
total_carbon_saved = sum(carbon_impacts[n] for n in selected_nodes)

# --- DASHBOARD METRICS DISPLAY ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Baseline System Risk", f"{baseline_risk:.2f}%")
col2.metric(
    "Optimized System Risk",
    f"{optimized_risk:.2f}%",
    delta=f"-{total_risk_drop:.2f} pts",
    delta_color="inverse",
)
col3.metric("Capital Deployed", f"${final_cost_spent:,.2f}")
col4.metric("Lead Time Saved", f"{total_lead_time_saved} Days")

sec1, sec2 = st.columns(2)
sec1.metric("ESG Carbon Offset", f"{total_carbon_saved} Metric Tons")
sec2.metric("Active Synergy Bundles", f"{len(active_bundle_names)} Applied")

if enable_target_mode:
  if optimized_risk <= target_risk_goal:
    st.success(
        f"🎯 Target Goal Achieved! Minimum budget required to reach"
        f" {target_risk_goal}% risk is ${final_cost_spent:,.2f}."
    )
  else:
    st.warning(
        "⚠️ Target Goal Unreachable with current nodes. Add more risk-reduction"
        " nodes or lower your target."
    )

if active_bundle_names:
  st.success(
      "Active Commercial Bundles Applied: " + ", ".join(active_bundle_names)
  )

st.markdown("---")
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
          "Lead Time Saved": f"{lead_times[n]} Days",
          "Carbon Impact": f"{carbon_impacts[n]} Tons",
      })
    df_portfolio = pd.DataFrame(portfolio_data)
    st.dataframe(df_portfolio, use_container_width=True)

    csv_data = df_portfolio.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Commercial Portfolio Report (CSV)",
        data=csv_data,
        file_name="commercial_supply_chain_optimization.csv",
        mime="text/csv",
    )
  else:
    st.warning(
        "No nodes selected. Increase your budget cap or adjust node costs."
    )

with tab2:
  st.subheader("Enterprise Budget Sensitivity Analysis")
  st.markdown("Evaluating portfolio scaling across variable capital caps:")

  sweep_results = []
  reference_budget = final_cost_spent if enable_target_mode else total_budget
  step_size = max(10000, int(reference_budget * 0.1))
  lower_bound = max(10000, reference_budget - (step_size * 4))
  upper_bound = reference_budget + (step_size * 4)
  budget_range = range(lower_bound, upper_bound, step_size)

  for b in budget_range:
    sub_prob = pl.LpProblem(f"Sub_{b}", pl.LpMaximize)
    sub_x = {n: pl.LpVariable(f"sx_{n}", cat="Binary") for n in nodes}

    sub_bundle_vars = {}
    for idx, row in edited_bundles.iterrows():
      b_name = row.get("Bundle Name")
      if not b_name or pd.isna(b_name) or str(b_name).strip().lower() == "none":
        continue
      try:
        b_discount = float(row.get("Discount ($)", 0))
      except (ValueError, TypeError):
        b_discount = 0.0

      req_nodes = [
          n.strip()
          for n in str(row.get("Required Nodes (Comma Separated)", "")).split(
              ","
          )
          if n.strip() in nodes
      ]
      if str(b_name).strip() and req_nodes:
        sb_var = pl.LpVariable(f"sb_{idx}_{b}", cat="Binary")
        sub_bundle_vars[str(b_name).strip()] = {
            "var": sb_var,
            "discount": b_discount,
            "nodes": req_nodes,
        }
        for rn in req_nodes:
          sub_prob += sb_var <= sub_x[rn]

    sub_prob += pl.lpSum(
        risks[n] * sub_x[n] + (0.5 * lead_times[n] * sub_x[n]) for n in nodes
    )
    sub_total_discounts = pl.lpSum(
        sub_bundle_vars[b]["var"] * sub_bundle_vars[b]["discount"]
        for b in sub_bundle_vars
    )
    sub_prob += (
        pl.lpSum(costs[n] * sub_x[n] for n in nodes) - sub_total_discounts <= b
    )

    sub_prob.solve(pl.PULP_CBC_CMD(msg=False))
    s_nodes = [n for n in nodes if pl.value(sub_x[n]) == 1]
    s_cost = sum(costs[n] for n in s_nodes)

    s_active_bundles = []
    s_discount_total = 0
    for b_name, b_data in sub_bundle_vars.items():
      if pl.value(b_data["var"]) == 1:
        s_active_bundles.append(b_name)
        s_discount_total += b_data["discount"]

    s_final_cost = s_cost - s_discount_total
    s_risk = sum(risks[n] for n in s_nodes)
    s_risk = min(baseline_risk, s_risk)
    s_optimized_risk = max(0.0, baseline_risk - s_risk)

    sweep_results.append({
        "Budget Cap": f"${b:,.2f}",
        "Actual Spent": f"${s_final_cost:,.2f}",
        "Risk Drop (pts)": s_risk,
        "Optimized Risk": f"{s_optimized_risk:.2f}%",
        "Bundles Active": (
            ", ".join(s_active_bundles) if s_active_bundles else "None"
        ),
        "Selected Portfolio": ", ".join(s_nodes) if s_nodes else "None",
        "RawBudget": b,
        "RawRisk": s_optimized_risk,
    })

  df_sweep = pd.DataFrame(sweep_results)

  st.line_chart(
      df_sweep.set_index("RawBudget")[["RawRisk"]],
      use_container_width=True,
  )
  st.caption("Figure: System Risk Reduction Curve across Enterprise Budgets.")

  st.dataframe(
      df_sweep[[
          "Budget Cap",
          "Actual Spent",
          "Optimized Risk",
          "Bundles Active",
          "Selected Portfolio",
      ]],
      use_container_width=True,
  )
