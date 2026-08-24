import streamlit as st
import pandas as pd
import numpy as np
import scipy.optimize as sco
import pulp as pl
import networkx as nx
import re
import time
import io
from supabase import create_client, Client

# 1. Page Configuration
st.set_page_config(
    page_title="Supply Chain Commercial Engine", 
    page_icon="⚡", 
    layout="wide"
)

# 2. Initialize Supabase Connection using Streamlit Secrets
@st.cache_resource
def init_supabase() -> Client:
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)

supabase = init_supabase()

# --- SECURITY: INPUT SANITIZATION HELPER ---
def sanitize_input(text: str) -> str:
    """Removes potential script injections and malicious characters."""
    if not isinstance(text, str):
        return ""
    cleaned = re.sub(r'<[^>]*?>', '', text)
    return cleaned[:500].strip()

# 3. Authentication & Security State Management Wrapper (Isolated Gate)
if "user" not in st.session_state:
    st.session_state.user = None
if "login_attempts" not in st.session_state:
    st.session_state.login_attempts = 0
if "lockout_time" not in st.session_state:
    st.session_state.lockout_time = 0

# Check for brute-force lockout status
if st.session_state.login_attempts >= 5:
    time_left = int(60 - (time.time() - st.session_state.lockout_time))
    if time_left > 0:
        st.error(f"🛡️ **Security Alert:** Too many failed login attempts. System locked for protection. Try again in {time_left} seconds.")
        st.stop()
    else:
        st.session_state.login_attempts = 0

if not st.session_state.user:
    st.title("🔐 Enterprise Corporate Portal")
    st.markdown("Please log in with your authorized corporate credentials. *Self-registration is disabled.*")
    
    signin_email = st.text_input("Corporate Email", key="si_email")
    signin_pass = st.text_input("Password", type="password", key="si_pass")
    
    if st.button("Log In to Portal"):
        sanitized_email = sanitize_input(signin_email)
        
        if not sanitized_email or not signin_pass:
            st.error("Please provide valid login credentials.")
        else:
            try:
                res = supabase.auth.sign_in_with_password({"email": sanitized_email, "password": signin_pass})
                st.session_state.user = res.user
                st.session_state.login_attempts = 0 
                st.success("Authenticated successfully!")
                st.rerun()
            except Exception as e:
                st.session_state.login_attempts += 1
                if st.session_state.login_attempts >= 5:
                    st.session_state.lockout_time = time.time()
                st.error(f"Authentication failed. Security log updated (Attempt {st.session_state.login_attempts}/5).")
            
    st.stop()

# --- LOGGED IN USER APP INTERFACE WRAPPER ---
st.sidebar.write(f"👤 Logged in as: {st.session_state.user.email}")

with st.sidebar.expander("🔑 Account & Security"):
    new_pass = st.text_input("New Password", type="password", key="new_pwd")
    if st.button("Update Password"):
        if len(new_pass) < 8:
            st.error("Password must be at least 8 characters long.")
        else:
            try:
                supabase.auth.update_user({"password": new_pass})
                st.success("Password updated securely!")
            except Exception as e:
                st.error("Failed to update password.")
            
    if st.button("Log Out"):
        supabase.auth.sign_out()
        st.session_state.user = None
        st.rerun()

st.title("Supply Chain Resilience & Capital Allocation Engine")
st.markdown(
    "*Enterprise Hybrid Architecture: PuLP Mixed-Integer Core (MES & Bundles) + SciPy/NumPy Monte Carlo Analytics.*"
)
st.markdown("------")

# --- SIDEBAR: EXECUTIVE CONTROL PANEL ---
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
uploaded_file = st.sidebar.file_uploader("Upload Node Data (CSV/Excel)", type=["csv", "xlsx"])

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

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            st.session_state.nodes_df = pd.read_csv(uploaded_file)
        else:
            st.session_state.nodes_df = pd.read_excel(uploaded_file)
        st.sidebar.success("File uploaded successfully!")
    except Exception as e:
        st.sidebar.error(f"Error reading file: {e}")

edited_nodes = st.sidebar.data_editor(
    st.session_state.nodes_df, num_rows="dynamic", use_container_width=True
)

# Robust Data Validation Layer
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

# Diminishing Returns Transformation (Strict Master Math Retention)
marginal_risks = {n: float(risks[n] ** 0.85) for n in nodes}

st.sidebar.markdown("---")
st.sidebar.subheader("2. Node Correlation Matrix (Institutional Risk)")
st.sidebar.markdown("Define cross-node disruption correlation coefficients ($\rho \in [-1, 1]$):")

if "corr_df" not in st.session_state or set(st.session_state.corr_df.columns[1:]) != set(nodes):
    corr_data = {"Node Name": nodes}
    for n in nodes:
        corr_data[n] = [1.0 if n == o else 0.35 for o in nodes]
    st.session_state.corr_df = pd.DataFrame(corr_data)

edited_corr = st.sidebar.data_editor(
    st.session_state.corr_df, num_rows="fixed", use_container_width=True, key="corr_editor"
)
st.session_state.corr_df = edited_corr

st.sidebar.markdown("---")
st.sidebar.subheader("3. Node Dependencies & Savings Bundles")

if "deps_df" not in st.session_state:
    st.session_state.deps_df = pd.DataFrame(columns=["Dependent Node", "Prerequisite Node"])
edited_deps = st.sidebar.data_editor(st.session_state.deps_df, num_rows="dynamic", use_container_width=True, key="deps_editor")
st.session_state.deps_df = edited_deps

if "bundles_df" not in st.session_state:
    st.session_state.bundles_df = pd.DataFrame([{"Bundle Name": "Port_Warehouse_Synergy", "Discount ($)": 50000}])
edited_bundles = st.sidebar.data_editor(st.session_state.bundles_df, num_rows="dynamic", use_container_width=True)
st.session_state.bundles_df = edited_bundles

# --- UNTOUCHED MASTER HYBRID OPTIMIZATION ENGINE: PULP (MIXED-INTEGER, MES, BUNDLES) ---
bundle_vars = {}
if enable_target_mode:
    prob = pl.LpProblem("Target_Goal_Hybrid_Optimization", pl.LpMinimize)
    y = {n: pl.LpVariable(f"y_{n}", cat="Binary") for n in nodes}
    x = {n: pl.LpVariable(f"x_{n}", lowBound=0.0, upBound=1.0, cat="Continuous") for n in nodes}

    for n in nodes:
        prob += x[n] >= 0.3 * y[n]
        prob += x[n] <= 1.0 * y[n]

    for idx, row in edited_bundles.iterrows():
        b_name = row.get("Bundle Name")
        if not b_name or pd.isna(b_name):
            continue
        b_discount = float(row.get("Discount ($)", 0) or 0)
        # BUG FIX 1: Dedicated multiselect key for Target Mode to prevent KeyError
        req_nodes = st.sidebar.multiselect(
            f"Target Nodes required for bundle: {b_name}",
            options=nodes,
            default=nodes[:2] if len(nodes) >= 2 else nodes,
            key=f"bundle_req_target_{idx}"
        )
        if str(b_name).strip() and req_nodes:
            b_var = pl.LpVariable(f"bundle_{idx}", cat="Binary")
            bundle_vars[str(b_name).strip()] = {"var": b_var, "discount": b_discount}
            for rn in req_nodes:
                if rn in nodes:
                    prob += b_var <= y[rn]

    for _, d_row in edited_deps.iterrows():
        dep_node, pre_node = d_row.get("Dependent Node"), d_row.get("Prerequisite Node")
        if dep_node in nodes and pre_node in nodes and dep_node != pre_node:
            prob += x[dep_node] <= x[pre_node]

    total_discounts = pl.lpSum(bundle_vars[b]["var"] * bundle_vars[b]["discount"] for b in bundle_vars)
    prob += pl.lpSum(costs[n] * x[n] for n in nodes) - total_discounts
    required_risk_drop = max(0.0, baseline_risk - target_risk_goal)
    prob += pl.lpSum(marginal_risks[n] * x[n] for n in nodes) >= required_risk_drop

else:
    prob = pl.LpProblem("Commercial_Supply_Chain_Hybrid_Optimization", pl.LpMaximize)
    y = {n: pl.LpVariable(f"y_{n}", cat="Binary") for n in nodes}
    x = {n: pl.LpVariable(f"x_{n}", lowBound=0.0, upBound=1.0, cat="Continuous") for n in nodes}

    for n in nodes:
        prob += x[n] >= 0.3 * y[n]
        prob += x[n] <= 1.0 * y[n]

    for idx, row in edited_bundles.iterrows():
        b_name = row.get("Bundle Name")
        if not b_name or pd.isna(b_name):
            continue
        b_discount = float(row.get("Discount ($)", 0) or 0)
        req_nodes = st.sidebar.multiselect(
            f"Nodes required for bundle: {b_name}",
            options=nodes,
            default=nodes[:2] if len(nodes) >= 2 else nodes,
            key=f"bundle_req_{idx}"
        )
        if str(b_name).strip() and req_nodes:
            b_var = pl.LpVariable(f"bundle_{idx}", cat="Binary")
            bundle_vars[str(b_name).strip()] = {"var": b_var, "discount": b_discount}
            for rn in req_nodes:
                if rn in nodes:
                    prob += b_var <= y[rn]

    for _, d_row in edited_deps.iterrows():
        dep_node, pre_node = d_row.get("Dependent Node"), d_row.get("Prerequisite Node")
        if dep_node in nodes and pre_node in nodes and dep_node != pre_node:
            prob += x[dep_node] <= x[pre_node]

    prob += pl.lpSum((opt_weight * marginal_risks[n] + (1 - opt_weight) * lead_times[n]) * x[n] for n in nodes)
    
    total_discounts = pl.lpSum(bundle_vars[b]["var"] * bundle_vars[b]["discount"] for b in bundle_vars)
    prob += pl.lpSum(costs[n] * x[n] for n in nodes) - total_discounts <= total_budget

try:
    prob.solve(pl.PULP_CBC_CMD(msg=False))
except:
    prob.solve(pl.CHOOSE_SOLVER(msg=False))

# --- RESULTS COMPUTATION ---
allocation_scales = {n: float(pl.value(x[n])) if pl.value(x[n]) is not None else 0.0 for n in nodes}
active_nodes = {n: scale for n, scale in allocation_scales.items() if scale > 0.001}

base_cost_spent = sum(costs[n] * scale for n, scale in allocation_scales.items())
applied_bundle_discounts = 0
active_bundle_names = []

for b_name, b_data in bundle_vars.items():
    if pl.value(b_data["var"]) == 1:
        applied_bundle_discounts += b_data["discount"]
        active_bundle_names.append(b_name)

final_cost_spent = base_cost_spent - applied_bundle_discounts
total_risk_drop = sum(marginal_risks[n] * scale for n, scale in allocation_scales.items())
total_risk_drop = min(baseline_risk, total_risk_drop)
optimized_risk = max(0.0, baseline_risk - total_risk_drop)
total_lead_time_saved = sum(lead_times[n] * scale for n, scale in allocation_scales.items())

# --- ADVANCED CHOLESKY MONTE CARLO ENGINE ---
def run_hybrid_monte_carlo(base_risk, scales_dict, marginal_dict, corr_df_matrix, nodes_list, iterations):
    np.random.seed(42)
    k = len(nodes_list)
    if k == 0 or iterations <= 0:
        return {"P50_Risk": base_risk, "P90_Risk": base_risk, "Std_Dev": 0.0}
    
    corr_matrix = np.eye(k)
    for i, n_row in enumerate(nodes_list):
        for j, n_col in enumerate(nodes_list):
            try:
                val = float(corr_df_matrix.loc[corr_df_matrix["Node Name"] == n_row, n_col].values[0])
                corr_matrix[i, j] = max(-0.99, min(0.99, val))
            except Exception:
                corr_matrix[i, j] = 1.0 if i == j else 0.35

    try:
        chol = np.linalg.cholesky(corr_matrix)
    except np.linalg.LinAlgError:
        min_eig = np.min(np.real(np.linalg.eigvals(corr_matrix)))
        corr_matrix += np.eye(k) * (-min_eig + 1e-5)
        chol = np.linalg.cholesky(corr_matrix)
        
    simulated_outcomes = []
    for _ in range(iterations):
        uncorrelated_z = np.random.normal(0, 1, k)
        correlated_z = np.dot(chol, uncorrelated_z)
        shocks = 1.0 + (0.12 * correlated_z)
        simulated_drop = sum(marginal_dict.get(n, 0) * scales_dict.get(n, 0.0) * max(0.4, shocks[i]) for i, n in enumerate(nodes_list))
        sim_risk = max(0.0, base_risk - simulated_drop)
        simulated_outcomes.append(sim_risk)
        
    simulated_outcomes = np.array(simulated_outcomes)
    return {
        "P50_Risk": np.percentile(simulated_outcomes, 50),
        "P90_Risk": np.percentile(simulated_outcomes, 90),
        "Std_Dev": np.std(simulated_outcomes)
    }

mc_results = run_hybrid_monte_carlo(baseline_risk, allocation_scales, marginal_risks, edited_corr, nodes, num_iterations)

# --- DASHBOARD METRICS DISPLAY ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Baseline System Risk", f"{baseline_risk:.2f}%")
col2.metric("Optimized Expected Risk", f"{optimized_risk:.2f}%", delta=f"-{total_risk_drop:.2f} pts", delta_color="inverse")
col3.metric("Capital Deployed", f"${final_cost_spent:,.2f}")
col4.metric("Lead Time Saved", f"{total_lead_time_saved:.1f} Days")

sec1, sec2, sec3 = st.columns(3)
sec1.metric(f"Institutional P50 Risk ({num_iterations:,} runs)", f"{mc_results['P50_Risk']:.2f}%")
sec2.metric("Institutional P90 Tail-Risk (VaR)", f"{mc_results['P90_Risk']:.2f}%")
sec3.metric("Active Synergy Bundles", f"{len(active_bundle_names)} Applied")

st.markdown("---")
tab1, tab2, tab3 = st.tabs([
    "Hybrid Portfolio Allocation", 
    "Budget Sensitivity Sweep", 
    "📜 Full Institutional Audit & Equation Ledger"
])

with tab1:
    st.subheader("Hybrid Mixed-Integer Portfolio (MES Thresholds & Bundles Enforced)")
    if active_nodes:
        portfolio_data = []
        for idx, (n, scale) in enumerate(active_nodes.items(), 1):
            portfolio_data.append({
                "Priority": f"Rank #{idx}",
                "Target Node": n,
                "Strategic Action": actions.get(n, "Custom Intervention"),
                "Funding Scale (%)": f"{scale * 100:.1f}%",
                "Capital Allocated": f"${costs[n] * scale:,.2f}",
                "Effective Risk Reduction": f"{risks[n] * scale:.2f}%",
                "Linked Diminishing Utility": f"{marginal_risks[n] * scale:.2f}",
                "Lead Time Saved": f"{lead_times[n] * scale:.1f} Days",
            })
        df_portfolio = pd.DataFrame(portfolio_data)
        st.dataframe(df_portfolio, use_container_width=True)

        csv_data = df_portfolio.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Hybrid Portfolio Report (CSV)",
            data=csv_data,
            file_name="hybrid_supply_chain_optimization.csv",
            mime="text/csv",
        )
    else:
        st.warning("No capital allocated. Adjust budget constraints or target goals.")

with tab2:
    st.subheader("Enterprise Budget Sensitivity Analysis (Fully Synchronized Engine)")
    sweep_results = []
    reference_budget = final_cost_spent if enable_target_mode else total_budget
    step_size = max(10000, int(reference_budget * 0.1))
    lower_bound = max(10000, reference_budget - (step_size * 4))
    upper_bound = reference_budget + (step_size * 4)
    budget_range = range(lower_bound, upper_bound, step_size)

    for b in budget_range:
        sub_prob = pl.LpProblem(f"Sub_{b}", pl.LpMaximize)
        sub_y = {n: pl.LpVariable(f"sy_{n}", cat="Binary") for n in nodes}
        sub_x = {n: pl.LpVariable(f"sx_{n}", lowBound=0.0, upBound=1.0, cat="Continuous") for n in nodes}
        
        for n in nodes:
            sub_prob += sub_x[n] >= 0.3 * sub_y[n]
            sub_prob += sub_x[n] <= 1.0 * sub_y[n]

        sub_bundle_vars = {}
        for idx, row in edited_bundles.iterrows():
            b_name = row.get("Bundle Name")
            if not b_name or pd.isna(b_name):
                continue
            b_discount = float(row.get("Discount ($)", 0) or 0)
            req_nodes = st.session_state.get(f"bundle_req_{idx}", nodes[:2] if len(nodes) >= 2 else nodes)
            if str(b_name).strip() and req_nodes:
                b_var = pl.LpVariable(f"sub_bundle_{idx}", cat="Binary")
                sub_bundle_vars[str(b_name).strip()] = {"var": b_var, "discount": b_discount}
                for rn in req_nodes:
                    if rn in nodes:
                        sub_prob += b_var <= sub_y[rn]

        for _, d_row in edited_deps.iterrows():
            dep_node, pre_node = d_row.get("Dependent Node"), d_row.get("Prerequisite Node")
            if dep_node in nodes and pre_node in nodes and dep_node != pre_node:
                sub_prob += sub_x[dep_node] <= sub_x[pre_node]

        sub_prob += pl.lpSum((opt_weight * marginal_risks[n] + (1 - opt_weight) * lead_times[n]) * sub_x[n] for n in nodes)
        
        sub_total_discounts = pl.lpSum(sub_bundle_vars[sb]["var"] * sub_bundle_vars[sb]["discount"] for sb in sub_bundle_vars)
        sub_prob += pl.lpSum(costs[n] * sub_x[n] for n in nodes) - sub_total_discounts <= b
        sub_prob.solve(pl.PULP_CBC_CMD(msg=False))
        
        s_scales = {n: pl.value(sub_x[n]) or 0.0 for n in nodes}
        s_base_cost = sum(costs[n] * scale for n, scale in s_scales.items())
        
        s_applied_discounts = 0
        for sb_name, sb_data in sub_bundle_vars.items():
            if pl.value(sb_data["var"]) == 1:
                s_applied_discounts += sb_data["discount"]
                
        s_cost = s_base_cost - s_applied_discounts
        s_risk_drop = sum(marginal_risks[n] * scale for n, scale in s_scales.items())
        s_optimized_risk = max(0.0, baseline_risk - s_risk_drop)

        sweep_results.append({
            "Budget Cap": f"${b:,.2f}",
            "Actual Spent": f"${s_cost:,.2f}",
            "Optimized Risk": f"{s_optimized_risk:.2f}%",
            "RawBudget": b,
            "RawRisk": s_optimized_risk,
        })

    df_sweep = pd.DataFrame(sweep_results)
    st.line_chart(df_sweep.set_index("RawBudget")[["RawRisk"]], use_container_width=True)
    st.dataframe(df_sweep[["Budget Cap", "Actual Spent", "Optimized Risk"]], use_container_width=True)

with tab3:
    st.subheader("📜 Full Institutional Audit & Master Equation Ledger")
    with st.expander("📌 View Complete System Master Equation (Zero Omissions)", expanded=True):
        st.markdown(r"""
        $$\begin{aligned}
        \textbf{I. Hybrid Objective Function Selection (Mode Dependent):} \\[0.3em]
        \mathcal{U}_{\text{system}} &= 
        \begin{cases} 
        \displaystyle\sum_{n \in N} \left( C_n x_n \right) - \sum_{b \in B} \left( D_b \cdot b_v \right) & (\text{Target Mode: } \text{LpMinimize}) \\[1em]
        \displaystyle\sum_{n \in N} \left[ w \cdot R_n^{0.85} + (1-w) \cdot L_n \right] x_n & (\text{Commercial Mode: } \text{LpMaximize})
        \end{cases}
        \\[1em]
        \textbf{II. Structural Operational & Feasibility Constraints (PuLP Mixed-Integer):} \\[0.3em]
        1. \text{ Semi-Continuous Scaling & MES Thresholds:} \quad & 0.3 \, y_n \le x_n \le 1.0 \, y_n, \quad \forall n \in N, \quad y_n \in \{0, 1\}, \quad x_n \in [0, 1] \\[0.5em]
        2. \text{ Capital Budget Cap & Bundle Discounts:} \quad & \sum_{n \in N} \left( C_n x_n \right) - \sum_{b \in B} \left( D_b \cdot b_v \right) \le B_{\text{total}} \\[0.5em]
        3. \text{ Prerequisite Dependency Cascade Capping:} \quad & x_{\text{dependent}} \le x_{\text{prerequisite}}, \quad \forall (\text{dep}, \text{pre}) \in \text{Edges} \\[0.5em]
        4. \text{ Synergy Bundle Logic Prerequisites:} \quad & b_v \le y_n, \quad \forall n \in \text{Nodes}(b), \quad b_v \in \{0, 1\} \\[0.5em]
        5. \text{ Target Risk Threshold Lower Bound:} \quad & \sum_{n \in N} \left( R_n^{0.85} x_n \right) \ge \max\left(0, \text{Risk}_{\text{baseline}} - \text{Risk}_{\text{target}}\right)
        \\[1em]
        \textbf{III. Multivariate Correlated Monte Carlo Simulation Engine (NumPy / SciPy Matrix Core):} \\[0.3em]
        1. \text{ User Matrix Definiteness Correction:} \quad & \Sigma_{\text{custom}} = \text{User Matrix} + \epsilon \cdot \mathbf{I} \quad (\text{Eigenvalue-clipped}) \\[0.5em]
        2. \text{ Cholesky Factorization Matrix:} \quad & \Sigma_{\text{custom}} = L L^T \\[0.5em]
        3. \text{ Correlated Normal Perturbation Vector:} \quad & Z_{\text{corr}} = L \cdot Z_{\text{normal}}, \quad Z_{\text{normal}} \sim \mathcal{N}(0, \mathbf{I}) \\[0.5em]
        4. \text{ Stochastic Outcome Distribution:} \quad & \text{Risk}_{\text{sim}}^{(i)} = \max\left(0, \text{Risk}_{\text{baseline}} - \sum_{n \in N} R_n^{0.85} x_n \max\left(0.4, 1.0 + 0.12 \, Z_{\text{corr}, n}^{(i)}\right)\right) \\[0.5em]
        5. \text{ Institutional Tail-Risk Value-at-Risk ($P_{90}$):} \quad & \text{VaR}_{90} = \text{Percentile}_{90}\left( \left\{ \text{Risk}_{\text{sim}}^{(1)}, \dots, \text{Risk}_{\text{sim}}^{(N)} \right\} \right)
        \end{aligned}$$
        """)

    with st.expander("2. Runtime Performance Ledger & Parameters"):
        st.markdown(f"""
        * **Active Weight Parameters:** $w = {opt_weight}$, $(1-w) = {1 - opt_weight}$
        * **Total Resolved Capital Outlay:** `${final_cost_spent:,.2f}`
        * **Deterministic System Risk Outcome:** $\mathbf{{ {optimized_risk:.2f}\% }}$
        * **Institutional Tail-Risk ($P_{90}$ VaR, $N={num_iterations:,}$):** $\mathbf{{{mc_results['P90_Risk']:.2f}\\%}}$
        """)

    with st.expander("3. Raw Data Input Verification Table"):
        st.dataframe(edited_nodes, use_container_width=True)

# --- MASTER AI EXPERT COPILOT WRAPPER (ISOLATED SIDE-PANEL ASSISTANT) ---
with st.sidebar:
    st.markdown("---")
    st.subheader("🤖 Master Supply Chain AI Expert")
    st.markdown("*Read-only copilot wrapper. Inspects state or guides adjustments.*")

    if "ai_messages" not in st.session_state:
        st.session_state.ai_messages = [
            {
                "role": "assistant",
                "content": "Hello! I am your isolated AI assistant wrapper. I can explain metrics, review your model status, or help verify parameters without altering your master equations.",
            }
        ]

    chat_container = st.container(height=250)
    with chat_container:
        for message in st.session_state.ai_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    raw_user_query = st.chat_input("Ask copilot about model metrics...", key="copilot_input")

    if raw_user_query:
        user_query = sanitize_input(raw_user_query)
        st.session_state.ai_messages.append({"role": "user", "content": user_query})
        query_lower = user_query.lower()
        
        # BUG FIX 2: Safely extract number from list to prevent TypeError crash
        numbers = re.findall(r'\d+(?:\.\d+)?', query_lower)
        target_val = float(numbers[0]) if numbers else None
        
        if "risk" in query_lower:
            response = f"📊 Your current optimized system risk is **{optimized_risk:.2f}%** (starting from a baseline of {baseline_risk:.2f}%)."
        elif "budget" in query_lower or "cost" in query_lower:
            response = f"💰 Total capital currently deployed stands at **${final_cost_spent:,.2f}** out of your active cap."
        elif "monte carlo" in query_lower or "var" in query_lower:
            response = f"🎲 Monte Carlo simulation ran {num_iterations:,} iterations. P50 risk is {mc_results['P50_Risk']:.2f}% and P90 tail risk (VaR) is {mc_results['P90_Risk']:.2f}%."
        elif target_val is not None and ("day" in query_lower or "target" in query_lower):
            response = f"🎯 Parsed target instruction for value `{target_val}`. Please use the executive control panel to apply adjustments directly to the model state."
        else:
            response = "I'm monitoring your sealed master optimization model. Ask me about system risk, capital deployment, or simulation percentiles anytime!"

        st.session_state.ai_messages.append({"role": "assistant", "content": response})
        st.rerun()
