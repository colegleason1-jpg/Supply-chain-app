import streamlit as st
import pulp

st.set_page_config(page_title="Supply Chain Risk & Capital Allocator", layout="wide")

st.title("Supply Chain Resilience & Capital Allocation Engine")
st.markdown("*Enterprise Edition: Operations Research ILP Solver with Budget Sensitivity Sweep.*")
st.markdown("---")

st.sidebar.header("Executive Control Panel")
budget_cap = st.sidebar.slider("Total Budget Cap ($)", min_value=350000, max_value=600000, value=500000, step=25000)
macro_shock = st.sidebar.slider("Manufacturing Disruption Risk (%)", min_value=10, max_value=90, value=60, step=5)
shipping_shock = st.sidebar.slider("Shipping / Port Congestion Risk (%)", min_value=10, max_value=90, value=45, step=5)
enable_synergy = st.sidebar.checkbox("Enable Multi-Node Synergy Bundle ($50k Discount)", value=True)

def solve_ilp(cap, synergy_active):
    nodes = {
        'Logistics_Hub': {'cost': 200000, 'risk_drop': 20.0, 'desc': 'Alternative port routing & freight contracts'},
        'Primary_Warehouse': {'cost': 250000, 'risk_drop': 22.0, 'desc': 'Primary warehouse resilience hardening'},
        'Backup_Supplier': {'cost': 150000, 'risk_drop': 15.0, 'desc': 'Upgrade backup logistics & capacity'},
        'Tier_1_Suppliers': {'cost': 300000, 'risk_drop': 24.0, 'desc': 'Dual-sourcing redundant Tier-1 parts'}
    }
    prob = pulp.LpProblem("Supply_Chain_Capital_Allocation", pulp.LpMaximize)
    x = {node: pulp.LpVariable(f"x_{node}", cat='Binary') for node in nodes}
    prob += pulp.lpSum(nodes[node]['risk_drop'] * x[node] for node in nodes)

    if synergy_active:
        y_bundle = pulp.LpVariable("y_bundle", cat='Binary')
        prob += y_bundle <= x['Logistics_Hub']
        prob += y_bundle <= x['Primary_Warehouse']
        prob += y_bundle >= x['Logistics_Hub'] + x['Primary_Warehouse'] - 1
        base_cost = pulp.lpSum(nodes[node]['cost'] * x[node] for node in nodes)
        prob += base_cost - (50000 * y_bundle) <= cap
    else:
        y_bundle = None
        prob += pulp.lpSum(nodes[node]['cost'] * x[node] for node in nodes) <= cap

    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    selected = []
    total_base_cost = 0
    total_risk_drop = 0
    for node, var in x.items():
        if pulp.value(var) == 1:
            selected.append({'node': node, **nodes[node]})
            total_base_cost += nodes[node]['cost']
            total_risk_drop += nodes[node]['risk_drop']
    bundled = (y_bundle is not None and pulp.value(y_bundle) == 1)
    final_spent = total_base_cost - (50000 if bundled else 0)
    return selected, total_base_cost, final_spent, total_risk_drop, bundled, pulp.LpStatus[prob.status]

portfolio, base_spent, final_spent, risk_drop, bundled, solver_status = solve_ilp(budget_cap, enable_synergy)
baseline_risk = min(99.0, (macro_shock * 1.0) + (shipping_shock * 0.5))
final_risk = max(5.0, baseline_risk - risk_drop)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="Baseline System Risk", value=f"{baseline_risk:.2f}%")
with col2:
    st.metric(label="Optimized System Risk", value=f"{final_risk:.2f}%", delta=f"-{risk_drop:.2f} pts")
with col3:
    st.metric(label="Capital Deployed", value=f"${final_spent:,.2f}", delta=f"Cap: ${budget_cap:,.2f}")

st.markdown("---")
tab_portfolio, tab_sensitivity = st.tabs(["Optimal Portfolio", "CFO Budget Sensitivity Sweep"])

with tab_portfolio:
    st.subheader("Globally Optimal Capital Allocation Portfolio")
    if solver_status == "Optimal" and portfolio:
        if bundled:
            st.success("Multi-node bundle synergy active: **$50,000 discount** applied.")
        table_data = [{"Priority": f"Rank #{i+1}", "Target Node": item['node'], "Strategic Action": item['desc'], "Capital Required": f"${item['cost']:,}", "Risk Reduction": f"{item['risk_drop']} pts"} for i, item in enumerate(portfolio)]
        st.table(table_data)
    else:
        st.warning("The selected budget cap is too restrictive to fund resilience enhancements.")

with tab_sensitivity:
    st.subheader("Budget Sensitivity Analysis (±$50k Variance)")
    sweep_range = range(max(350000, budget_cap - 50000), budget_cap + 55000, 25000)
    sweep_data = []
    for b in sweep_range:
        sel, _, spent, r_drop, b_active, _ = solve_ilp(b, enable_synergy)
        nodes_str = ", ".join([item['node'] for item in sel]) if sel else "None"
        sweep_data.append({"Budget Cap": f"${b:,}", "Actual Spent": f"${spent:,}", "Risk Drop": f"{r_drop:.1f} pts", "Bundle Active?": "Yes" if b_active else "No", "Selected Portfolio": nodes_str})
    st.table(sweep_data)