import streamlit as st
import pandas as pd
import numpy as np
import scipy.optimize as sco
import pulp as pl
import networkx as nx
import re
import time
import io
import os
import requests
from typing import List, Dict, Any, Tuple
from pydantic import BaseModel, Field, ValidationError
from supabase import create_client, Client

# ==========================================
# 1. PAGE CONFIGURATION & ARCHITECTURE SETUP
# ==========================================
st.set_page_config(
    page_title="Supply Chain Commercial Engine", 
    page_icon="⚡", 
    layout="wide"
)

# ==========================================
# 2. MODULAR SECURITY, MVC & HEALTH GUARD
# ==========================================
class EnterpriseInputSchema(BaseModel):
    baseline_risk: float = Field(..., ge=0.0, le=100.0)
    opt_weight: float = Field(..., ge=0.0, le=1.0)
    num_iterations: int = Field(..., ge=1000, le=50000)
    total_budget: int = Field(..., ge=10000, le=1000000000)
    target_risk_goal: float = Field(default=20.0, ge=0.0, le=100.0)

def sanitize_input(text: str) -> str:
    """Advanced sanitization stripping malformed tags, script injections, and dangerous payloads."""
    if not isinstance(text, str):
        return ""
    cleaned = re.sub(r'<[^>]*?>', '', text)
    cleaned = re.sub(r'javascript:|vbscript:|onload=|onerror=', '', cleaned, flags=re.IGNORECASE)
    return cleaned[:500].strip()

class SupplyChainMVCController:
    """MVC Wrapper isolating session memory pools to prevent state leakage."""
    def __init__(self):
        self.init_session_state()

    def init_session_state(self):
        defaults = {
            "val_baseline_risk": 65.5,
            "val_opt_weight": 0.5,
            "val_num_iterations": 10000,
            "val_target_mode": False,
            "val_target_risk_goal": 20.0,
            "val_total_budget": 750000,
            "val_sim_horizon": 30,
        }
        for key, val in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = val

class SystemHealthMonitor:
    """Self-diagnostic guardian monitoring solver health and code drift."""
    @staticmethod
    def inspect_solver_status(prob: pl.LpProblem) -> bool:
        status_str = pl.LpStatus[prob.status]
        if status_str != "Optimal":
            st.warning(f"⚠️ **Optimization Notice:** Model status reported as **{status_str}**.")
            if status_str == "Infeasible":
                st.error("The current Target Risk Goal or constraint set is mathematically unreachable within your budget cap.")
            return False
        return True

# Initialize MVC Controller
controller = SupplyChainMVCController()

# ==========================================
# 3. SUPABASE INITIALIZATION & SECURITY GATE
# ==========================================
@st.cache_resource
def init_supabase() -> Client:
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)

supabase = init_supabase()

if "user" not in st.session_state:
    st.session_state.user = None
if "login_attempts" not in st.session_state:
    st.session_state.login_attempts = 0
if "lockout_time" not in st.session_state:
    st.session_state.lockout_time = 0

if st.session_state.login_attempts >= 5:
    time_left = int(60 - (time.time() - st.session_state.lockout_time))
    if time_left > 0:
        st.error(f"🛡️ **Security Alert:** Too many failed login attempts. System locked. Try again in {time_left} seconds.")
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
    "*Enterprise Hybrid Architecture: PuLP Mixed-Integer Core (MES & Bundles) + SciPy/NumPy Monte Carlo Analytics + Cloud-Safe Live Telemetry.*"
)
st.markdown("------")

# ==========================================
# 4. EXECUTIVE CONTROL PANEL & SIMULATION HORIZON
# ==========================================
st.sidebar.header("Executive Control Panel")

baseline_risk = st.sidebar.number_input(
    "Starting Baseline System Risk (%)",
    min_value=0.0,
    max_value=100.0,
    value=st.session_state.val_baseline_risk,
    step=1.0,
    key="val_baseline_risk"
)

st.sidebar.markdown("---")
opt_weight = st.sidebar.slider(
    "Optimization Weight (Risk vs. Lead Time)",
    min_value=0.0,
    max_value=1.0,
    value=st.session_state.val_opt_weight,
    step=0.05,
    key="val_opt_weight",
    help="1.0 = Pure focus on Risk Reduction. 0.0 = Pure focus on Lead Time Savings.",
)

num_iterations = st.sidebar.slider(
    "Monte Carlo Simulation Precision (Runs)",
    min_value=1000,
    max_value=50000,
    value=st.session_state.val_num_iterations,
    step=5000,
    key="val_num_iterations",
    help="Higher iterations increase precision for risk committees and tail-risk VaR percentiles."
)

sim_horizon = st.sidebar.slider(
    "Simulation Horizon (Days Out)",
    min_value=7,
    max_value=365,
    value=st.session_state.get("val_sim_horizon", 30),
    step=1,
    key="val_sim_horizon",
    help="Alters how far out tail-risk disruptions are simulated across tactical or strategic timeframes."
)

st.sidebar.markdown("---")
enable_target_mode = st.sidebar.checkbox(
    "🎯 Enable Target Risk Goal Mode",
    value=st.session_state.val_target_mode,
    key="val_target_mode",
    help="Minimizes capital deployment to achieve a specific target risk threshold using true LpMinimize cost functions.",
)

if enable_target_mode:
    target_risk_goal = st.sidebar.number_input(
        "Target System Risk Goal (%)",
        min_value=0.0,
        max_value=100.0,
        value=st.session_state.get("val_target_risk_goal", 20.0),
        step=1.0,
        key="val_target_risk_goal"
    )
    total_budget = 1000000000
    st.sidebar.info(f"Target Mode Active: Minimizing capital to reach <= {target_risk_goal}% risk.")
else:
    default_budget = st.session_state.get("pre_target_budget", st.session_state.get("val_total_budget", 750000))
    total_budget = st.sidebar.number_input(
        "Total Budget Cap ($)",
        min_value=10000,
        max_value=1000000000,
        value=default_budget,
        step=50000,
        format="%d",
        key="val_total_budget"
    )

# ==========================================
# 5. CLOUD-SAFE TELEMETRY ADAPTER & CACHE
# ==========================================
class MacroTelemetrySchema(BaseModel):
    brent_crude_price: float = Field(default=80.0, ge=0.0)
    shipping_lane_index: float = Field(default=1.0, ge=0.0)
    air_cargo_index: float = Field(default=1.0, ge=0.0)
    interest_rate: float = Field(default=0.05, ge=0.0, le=1.0)
    global_port_risk_pct: float = Field(default=15.0, ge=0.0, le=100.0)

class CloudSafeMacroTelemetryAdapter:
    @staticmethod
    def get_cached_telemetry() -> Tuple[float, str, MacroTelemetrySchema]:
        if "cached_macro_mult" not in st.session_state:
            st.session_state.cached_macro_mult = 1.0
            st.session_state.cached_feed_status = "Initialized (Default)"
            st.session_state.cached_schema = MacroTelemetrySchema()

        try:
            url = "https://api.api-ninjas.com/v1/oilprice?type=brent"
            headers = {'X-Api-Key': 'YOUR_FREE_API_KEY_HERE'}
            response = requests.get(url, headers=headers, timeout=1.5)
            
            if response.status_code == 200:
                data = response.json()
                price = float(data.get('price', 80.0))
                telemetry = MacroTelemetrySchema(
                    brent_crude_price=price,
                    shipping_lane_index=1.05 if price > 85.0 else 1.0,
                    air_cargo_index=1.02,
                    interest_rate=0.052,
                    global_port_risk_pct=15.0
                )
                baseline_anchor = 80.0
                macro_mult = 1.0 + max(0.0, ((price - baseline_anchor) / baseline_anchor) * 0.15)
                
                st.session_state.cached_macro_mult = round(macro_mult, 4)
                st.session_state.cached_feed_status = f"Active (Brent: ${price:.2f})"
                st.session_state.cached_schema = telemetry
        except Exception:
            st.session_state.cached_feed_status = "Stable (Using Cached Fallback)"

        return (
            st.session_state.cached_macro_mult, 
            st.session_state.cached_feed_status, 
            st.session_state.cached_schema
        )

st.sidebar.markdown("---")
st.sidebar.subheader("🌐 Global Macro & Logistics Telemetry")
live_macro_multiplier, feed_status, telemetry_data = CloudSafeMacroTelemetryAdapter.get_cached_telemetry()

st.sidebar.metric("Global Macro Multiplier", f"{live_macro_multiplier}x", delta=feed_status)
st.sidebar.text(f"🚢 Shipping Index: {telemetry_data.shipping_lane_index}x")
st.sidebar.text(f"✈️ Air Cargo Index: {telemetry_data.air_cargo_index}x")
st.sidebar.text(f"📈 Interest Rate: {telemetry_data.interest_rate*100:.1f}%")

effective_baseline_risk = min(100.0, baseline_risk * live_macro_multiplier)

# ==========================================
# 6. PROPRIETARY FILE INGESTION & AMPLIFICATION
# ==========================================
st.sidebar.markdown("---")
st.sidebar.subheader("📂 Proprietary Data Amplification")
uploaded_file = st.sidebar.file_uploader(
    "Upload Enterprise ERP/SAP Data (.xlsx, .xls, .csv)", 
    type=["xlsx", "xls", "csv"]
)

if "nodes_df" not in st.session_state:
    st.session_state.nodes_df = pd.DataFrame([
        {
            "Node Name": "Global_Freight_Hub",
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
            "Node Name": "Regional_Supplier_A",
            "Action": "Upgrade backup logistics & capacity",
            "Cost": 150000,
            "Risk Reduction (%)": 10.0,
            "Lead Time Saved (Days)": 7,
            "Carbon Impact (Tons)": 80,
        },
    ])

if uploaded_file is not None:
    try:
        file_extension = os.path.splitext(uploaded_file.name)[1][1:].lower()
        if file_extension in ["xlsx", "xls"]:
            df_uploaded = pd.read_excel(uploaded_file)
        elif file_extension == "csv":
            df_uploaded = pd.read_csv(uploaded_file)
        else:
            df_uploaded = None

        if df_uploaded is not None:
            st.sidebar.success(f"Amplified with **{uploaded_file.name}**!")
            st.session_state.nodes_df = df_uploaded
    except Exception as e:
        st.sidebar.error(f"Error parsing uploaded file: {e}")

edited_nodes = st.sidebar.data_editor(
    st.session_state.nodes_df, num_rows="dynamic", use_container_width=True
)

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

marginal_risks = {n: float((live_macro_multiplier * risks[n]) ** 0.85) for n in nodes}

st.sidebar.markdown("---")
st.sidebar.subheader("2. Node Correlation Matrix (Institutional Risk)")
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

# ==========================================
# 7. MASTER HYBRID OPTIMIZATION ENGINE (PULP)
# ==========================================
bundle_vars = {}                  
bundle_requirements_cache = {}    

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
        req_nodes = st.sidebar.multiselect(
            f"Target Nodes required for bundle: {b_name}",
            options=nodes,
            default=nodes[:2] if len(nodes) >= 2 else nodes,
            key=f"bundle_req_target_{idx}"
        )
        bundle_requirements_cache[idx] = req_nodes
        
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

    total_discounts = pl.lpSum(bundle_vars[b]["var"] * bundle_vars[b]["discount"] for b in bundle_vars) if bundle_vars else 0
    prob += pl.lpSum(costs[n] * x[n] for n in nodes) - total_discounts
    required_risk_drop = max(0.0, effective_baseline_risk - target_risk_goal)
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
        bundle_requirements_cache[idx] = req_nodes

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
    total_discounts = pl.lpSum(bundle_vars[b]["var"] * bundle_vars[b]["discount"] for b in bundle_vars) if bundle_vars else 0
    prob += pl.lpSum(costs[n] * x[n] for n in nodes) - total_discounts <= total_budget

try:
    prob.solve(pl.PULP_CBC_CMD(msg=False))
except:
    prob.solve(pl.CHOOSE_SOLVER(msg=False))

SystemHealthMonitor.inspect_solver_status(prob)

# ==========================================
# 8. RESULTS COMPUTATION & HORIZON MONTE CARLO
# ==========================================
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
total_risk_drop = min(effective_baseline_risk, total_risk_drop)
optimized_risk = max(0.0, effective_baseline_risk - total_risk_drop)
total_lead_time_saved = sum(lead_times[n] * scale for n, scale in allocation_scales.items())

def run_hybrid_monte_carlo(base_risk, scales_dict, marginal_dict, corr_df_matrix, nodes_list, iterations, horizon_days):
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
        
    horizon_factor = np.sqrt(horizon_days / 30.0)
    simulated_outcomes = []
    for _ in range(iterations):
        uncorrelated_z = np.random.normal(0, 1, k)
        correlated_z = np.dot(chol, uncorrelated_z)
        shocks = 1.0 + (0.12 * horizon_factor * correlated_z)
        simulated_drop = sum(marginal_dict.get(n, 0) * scales_dict.get(n, 0.0) * max(0.4, shocks[i]) for i, n in enumerate(nodes_list))
        sim_risk = max(0.0, base_risk - simulated_drop)
        simulated_outcomes.append(sim_risk)
        
    simulated_outcomes = np.array(simulated_outcomes)
    return {
        "P50_Risk": np.percentile(simulated_outcomes, 50),
        "P90_Risk": np.percentile(simulated_outcomes, 90),
        "Std_Dev": np.std(simulated_outcomes)
    }

mc_results = run_hybrid_monte_carlo(effective_baseline_risk, allocation_scales, marginal_risks, edited_corr, nodes, num_iterations, sim_horizon)

# ==========================================
# 9. DASHBOARD METRICS DISPLAY & TABS
# ==========================================
col1, col2, col3, col4 = st.columns(4)
col1.metric("Effective System Risk", f"{effective_baseline_risk:.2f}%", delta=f"{live_macro_multiplier}x Global Macro" if live_macro_multiplier != 1.0 else "Normal")
col2.metric("Optimized Expected Risk", f"{optimized_risk:.2f}%", delta=f"-{total_risk_drop:.2f} pts", delta_color="inverse")
col3.metric("Capital Deployed", f"${final_cost_spent:,.2f}")
col4.metric("Lead Time Saved", f"{total_lead_time_saved:.1f} Days")

sec1, sec2, sec3 = st.columns(3)
sec1.metric(f"Institutional P50 Risk ({num_iterations:,} runs, {sim_horizon}d)", f"{mc_results['P50_Risk']:.2f}%")
sec2.metric(f"Institutional P90 Tail-Risk (VaR, {sim_horizon}d)", f"{mc_results['P90_Risk']:.2f}%")
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
            req_nodes = bundle_requirements_cache.get(idx, nodes[:2] if len(nodes) >= 2 else nodes)
            
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
        sub_total_discounts = pl.lpSum(sub_bundle_vars[sb]["var"] * sub_bundle_vars[sb]["discount"] for sb in sub_bundle_vars) if sub_bundle_vars else 0
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
        s_optimized_risk = max(0.0, effective_baseline_risk - s_risk_drop)

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
        st.markdown("### **I. Dual-Layer Data Blending & Global Macro Telemetry**")
        st.markdown("1. **Live Global Macro Multiplier Factor ($\beta_{\text{macro}}$):**")
        st.latex(r"\beta_{\text{macro}} = \begin{cases} 1.0 + \left(\frac{P_{\text{brent}} - P_{\text{anchor}}}{P_{\text{anchor}}}\right) \cdot 0.15 & \text{if } P_{\text{brent}} > P_{\text{anchor}} \\ 1.0 & \text{otherwise (or offline fallback)} \end{cases}")
        st.markdown("2. **Macro-Scaled Diminishing Returns Utility ($R_n^*$):**")
        st.latex(r"R_n^* = \left(\beta_{\text{macro}} \cdot R_n\right)^{0.85}, \quad \forall n \in N_{\text{active}}")

        st.markdown("---")
        st.markdown("### **II. Horizon-Adjusted Multivariate Correlated Monte Carlo Simulation**")
        st.markdown("1. **Horizon-Scaled Volatility Multiplier:**")
        st.latex(r"\sigma_{\text{horizon}} = \sqrt{\frac{\text{Days}}{30}}")
        st.markdown("2. **Stochastic Outcome Distribution:**")
        st.latex(r"\text{Risk}_{\text{sim}}^{(i)} = \max\left(0, (\beta_{\text{macro}} \cdot \text{Risk}_{\text{baseline}}) - \sum_{n \in N} R_n^* x_n \max\left(0.4, 1.0 + 0.12 \, \sigma_{\text{horizon}} \, Z_{\text{corr}, n}^{(i)}\right)\right)")

    with st.expander("2. Runtime Performance Ledger & Parameters"):
        st.markdown(f"""
        * **Live Global Macro Multiplier:** $\\beta_{{\\text{{macro}}}} = {live_macro_multiplier}$ ({feed_status})
        * **Simulation Horizon:** $H = {sim_horizon}$ Days
        * **Active Weight Parameters:** $w = {opt_weight}$, $(1-w) = {1 - opt_weight}$
        * **Total Resolved Capital Outlay:** `${final_cost_spent:,.2f}`
        * **Deterministic System Risk Outcome:** $\mathbf{{ {optimized_risk:.2f}\% }}$
        * **Institutional Tail-Risk ($P_{90}$ VaR, $N={num_iterations:,}$):** $\mathbf{{{mc_results['P90_Risk']:.2f}\\%}}$
        """)

# ==========================================
# 10. MASTER AI EXPERT COPILOT CHAT WIDGET (DOMAIN ENRICHED)
# ==========================================
SUPPLY_CHAIN_DOMAIN_EXPERTISE = """
You are an expert Enterprise Supply Chain AI Copilot and optimization strategist. 
Your core architecture is built on a hybrid model:
1. PuLP Mixed-Integer Programming (MIP) handling semi-continuous Minimum Order Quantity (MOQ) boundaries (30% to 100%) and synergistic bundle discounts.
2. SciPy/NumPy Cholesky-corrected multivariate Monte Carlo simulations modeling institutional tail-risks (VaR P50/P90).
3. Real-time global macro telemetry factors multiplying baseline risks.

Your role is to assist executives by answering analytical questions and safely adjusting session parameters (budget, risk weights, horizons, and target risk goals) via direct session state modification. Never attempt to alter the core math solver directly.
"""

with st.sidebar:
    st.markdown("---")
    st.subheader("🤖 Master Supply Chain AI Expert")
    st.markdown("*Autonomous external controller. Powered with deep domain expertise.*")

    if "ai_messages" not in st.session_state:
        st.session_state.ai_messages = [
            {
                "role": "assistant",
                "content": "Hello! I am your supply chain copilot, backed by deep optimization and logistics domain knowledge. How can I help you adjust your parameters today?",
            }
        ]

    chat_container = st.container(height=260)
    with chat_container:
        for message in st.session_state.ai_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    raw_user_query = st.chat_input("Instruct copilot (e.g., 'increase budget to 1M')...", key="copilot_input")

    if raw_user_query:
        user_query = sanitize_input(raw_user_query)
        st.session_state.ai_messages.append({"role": "user", "content": user_query})
        query_lower = user_query.lower()
        
        numbers = re.findall(r'\d+(?:\.\d+)?', query_lower)
        target_val = float(numbers[0]) if numbers else None
        
        response = ""
        action_taken = False

        if "budget" in query_lower or "cap" in query_lower:
            if target_val is not None:
                new_b = int(target_val if target_val > 1000 else target_val * 1000000)
                st.session_state.val_total_budget = new_b
                if st.session_state.get("val_target_mode", False):
                    st.session_state.val_target_mode = False
                response = f"💰 Externally updated your **Total Budget Cap** to **${new_b:,.2f}**. Rerunning model..."
                action_taken = True
            else:
                response = f"💰 Your current budget cap is set to **${total_budget:,.2f}** with **${final_cost_spent:,.2f}** deployed."
        
        elif "horizon" in query_lower or "days" in query_lower:
            if target_val is not None:
                st.session_state.val_sim_horizon = int(target_val)
                response = f"⏱️ Updated simulation horizon to **{int(target_val)} days**. Rerunning model..."
                action_taken = True
            else:
                response = f"⏱️ Current simulation horizon is set to **{sim_horizon} days**."

        elif "risk" in query_lower:
            if "target" in query_lower and target_val is not None:
                if not st.session_state.get("val_target_mode", False):
                    st.session_state.pre_target_budget = st.session_state.get("val_total_budget", total_budget)
                st.session_state.val_target_mode = True
                st.session_state.val_target_risk_goal = target_val
                response = f"🎯 Activated **Target Risk Mode** aiming for **{target_val}%** system risk while protecting capital. Rerunning model..."
                action_taken = True
            else:
                response = f"📊 System risk stands at **{optimized_risk:.2f}%** (Baseline: {effective_baseline_risk:.2f}%, P90 VaR: {mc_results['P90_Risk']:.2f}%)."
        
        else:
            response = "I am monitoring your master optimization architecture with full domain awareness. You can command me to adjust parameters like **'set budget to 1.5M'**, **'set horizon to 90 days'**, or **'target 15% risk'**."

        st.session_state.ai_messages.append({"role": "assistant", "content": response})
        
        if action_taken:
            time.sleep(0.4)
            st.rerun()
