import time
import io
import os
import re
import requests
from typing import List, Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np
import pulp as pl
import networkx as nx
import streamlit as st
from supabase import create_client, Client
import yfinance as yf  # moved to top level

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch

# ==========================================
# 1. PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="Supply Chain Resilience & Capital Allocation Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. SUPABASE + SECURITY
# ==========================================
@st.cache_resource
def init_supabase() -> Client:
    url = st.secrets.get("SUPABASE_URL", "")
    key = st.secrets.get("SUPABASE_KEY", "")
    if not url or not key:
        st.error("Supabase credentials missing in secrets.")
        st.stop()
    return create_client(url, key)

supabase = init_supabase()

def sanitize_input(text: str) -> str:
    if not isinstance(text, str):
        return ""
    cleaned = re.sub(r'<[^>]*?>', '', text)
    return cleaned[:500].strip()

if "user" not in st.session_state:
    st.session_state.user = None
if "login_attempts" not in st.session_state:
    st.session_state.login_attempts = 0
if "lockout_time" not in st.session_state:
    st.session_state.lockout_time = 0

if st.session_state.login_attempts >= 5:
    time_left = int(60 - (time.time() - st.session_state.lockout_time))
    if time_left > 0:
        st.error(f"🛡️ Too many failed attempts. Locked for {time_left}s.")
        st.stop()
    else:
        st.session_state.login_attempts = 0

if not st.session_state.user:
    st.title("🔐 Enterprise Corporate Portal")
    st.markdown("Authorized credentials only. Self-registration disabled.")
    email = st.text_input("Corporate Email")
    password = st.text_input("Password", type="password")
    if st.button("Log In"):
        email = sanitize_input(email)
        if not email or not password:
            st.error("Credentials required.")
        else:
            try:
                res = supabase.auth.sign_in_with_password({"email": email, "password": password})
                st.session_state.user = res.user
                st.session_state.login_attempts = 0
                st.rerun()
            except Exception:
                st.session_state.login_attempts += 1
                if st.session_state.login_attempts >= 5:
                    st.session_state.lockout_time = time.time()
                st.error(f"Authentication failed ({st.session_state.login_attempts}/5)")
    st.stop()

st.sidebar.write(f"👤 {st.session_state.user.email}")
with st.sidebar.expander("Account"):
    if st.button("Log Out"):
        supabase.auth.sign_out()
        st.session_state.user = None
        st.rerun()

# ==========================================
# 3. MARKET SECTOR DEFINITIONS
# ==========================================
MARKET_SECTORS = [
    {"Sector": "Information_Technology", "Ticker": "XLK", "Category": "Growth"},
    {"Sector": "Communication_Services", "Ticker": "XLC", "Category": "Growth"},
    {"Sector": "Energy", "Ticker": "XLE", "Category": "Cyclical"},
    {"Sector": "Materials", "Ticker": "XLB", "Category": "Cyclical"},
    {"Sector": "Industrials", "Ticker": "XLI", "Category": "Cyclical"},
    {"Sector": "Consumer_Discretionary", "Ticker": "XLY", "Category": "Cyclical"},
    {"Sector": "Financials", "Ticker": "XLF", "Category": "Cyclical"},
    {"Sector": "Real_Estate", "Ticker": "XLRE", "Category": "Cyclical"},
    {"Sector": "Health_Care", "Ticker": "XLV", "Category": "Defensive"},
    {"Sector": "Consumer_Staples", "Ticker": "XLP", "Category": "Defensive"},
    {"Sector": "Utilities", "Ticker": "XLU", "Category": "Defensive"},
]

def init_market_state():
    if "market_df" not in st.session_state:
        rows = []
        for s in MARKET_SECTORS:
            rows.append({
                "Sector": s["Sector"],
                "Ticker": s["Ticker"],
                "Category": s["Category"],
                "Price": 100.0,
                "ChangePct": 0.0,
                "Status": "Initialized"
            })
        st.session_state.market_df = pd.DataFrame(rows)
        st.session_state.last_market_sync = 0.0

def sync_markets(force: bool = False):
    now = time.time()
    if not force and (now - st.session_state.last_market_sync) < 1800:
        return
    df = st.session_state.market_df.copy()
    for i, row in df.iterrows():
        try:
            hist = yf.Ticker(row["Ticker"]).history(period="5d")
            if len(hist) >= 2 and "Close" in hist.columns:
                price = float(hist["Close"].iloc[-1])
                prev = float(hist["Close"].iloc[-2])
                change = ((price - prev) / prev) * 100
                df.at[i, "Price"] = round(price, 2)
                df.at[i, "ChangePct"] = round(change, 2)
                df.at[i, "Status"] = "Live"
            else:
                df.at[i, "Status"] = "Stale"
        except Exception:
            df.at[i, "Status"] = "Offline"
    st.session_state.market_df = df
    st.session_state.last_market_sync = now

# ==========================================
# 4. LIVE MACRO (OIL)
# ==========================================
@st.cache_data(ttl=1800)
def get_macro_multiplier() -> Tuple[float, str]:
    try:
        api_key = st.secrets.get("API_NINJAS_KEY", "")
        if not api_key:
            return 1.0, "No API key – using neutral"
        r = requests.get(
            "https://api.api-ninjas.com/v1/oilprice?type=brent",
            headers={"X-Api-Key": api_key},
            timeout=3
        )
        if r.status_code == 200:
            price = float(r.json().get("price", 80))
            if price > 85:
                mult = 1.0 + ((price - 85) / 85) * 0.18
                return round(mult, 4), f"Elevated (Brent ${price:.1f})"
            return 1.0, f"Normal (Brent ${price:.1f})"
    except Exception:
        pass
    return 1.0, "Offline – neutral"

# ==========================================
# 5. CORE DATA + MARKET FEEDBACK (CLEAN)
# ==========================================
def default_nodes() -> pd.DataFrame:
    return pd.DataFrame([
        {"Node Name": "Global_Freight_Hub", "Action": "Alternative routing & contracts", "Cost": 200000, "Risk Reduction (%)": 15.0, "Lead Time Saved (Days)": 5, "Carbon Impact (Tons)": 120, "Linked Sector": "Industrials"},
        {"Node Name": "Primary_Warehouse", "Action": "Resilience hardening", "Cost": 250000, "Risk Reduction (%)": 15.0, "Lead Time Saved (Days)": 2, "Carbon Impact (Tons)": 50, "Linked Sector": "Real_Estate"},
        {"Node Name": "Regional_Supplier_A", "Action": "Backup capacity upgrade", "Cost": 150000, "Risk Reduction (%)": 10.0, "Lead Time Saved (Days)": 7, "Carbon Impact (Tons)": 80, "Linked Sector": "Materials"},
        {"Node Name": "Semiconductor_Buffer", "Action": "Strategic inventory buffer", "Cost": 180000, "Risk Reduction (%)": 12.0, "Lead Time Saved (Days)": 4, "Carbon Impact (Tons)": 30, "Linked Sector": "Information_Technology"},
    ])

if "nodes_raw" not in st.session_state:
    st.session_state.nodes_raw = default_nodes()

# ==========================================
# SIDEBAR CONTROLS
# ==========================================
st.sidebar.header("Executive Controls")

baseline_risk = st.sidebar.number_input("Baseline System Risk (%)", 0.0, 100.0, 65.5, 1.0)
opt_weight = st.sidebar.slider("Risk vs Lead-Time Weight", 0.0, 1.0, 0.55, 0.05,
                               help="1.0 = pure risk focus")
carbon_weight = st.sidebar.slider("Carbon Weight in Objective", 0.0, 0.4, 0.12, 0.02)
mc_runs = st.sidebar.slider("Monte Carlo Runs", 2000, 30000, 10000, 2000)

target_mode = st.sidebar.checkbox("Target Risk Mode (minimize capital)")
if target_mode:
    target_risk = st.sidebar.number_input("Target Risk (%)", 5.0, 50.0, 22.0, 1.0)
    budget = 50_000_000
else:
    budget = st.sidebar.number_input("Budget Cap ($)", 50_000, 20_000_000, 750_000, 25_000)

macro_mult, macro_status = get_macro_multiplier()
st.sidebar.metric("Macro Multiplier", f"{macro_mult:.3f}x", macro_status)
effective_baseline = min(100.0, baseline_risk * macro_mult)

st.sidebar.markdown("---")
st.sidebar.subheader("Nodes")
uploaded = st.sidebar.file_uploader("Upload nodes (.xlsx / .csv)", type=["xlsx", "csv"])
if uploaded:
    try:
        if uploaded.name.endswith(".csv"):
            df_up = pd.read_csv(uploaded)
        else:
            df_up = pd.read_excel(uploaded)
        # light schema normalization
        rename_map = {c: c.strip() for c in df_up.columns}
        df_up = df_up.rename(columns=rename_map)
        st.session_state.nodes_raw = df_up
        st.sidebar.success("File loaded")
    except Exception as e:
        st.sidebar.error(f"Upload failed: {e}")

edited = st.sidebar.data_editor(
    st.session_state.nodes_raw,
    num_rows="dynamic",
    use_container_width=True,
    key="nodes_editor"
)

# Clean numeric columns
for col in ["Cost", "Risk Reduction (%)", "Lead Time Saved (Days)", "Carbon Impact (Tons)"]:
    if col in edited.columns:
        edited[col] = pd.to_numeric(edited[col], errors="coerce").fillna(0).clip(lower=0)

st.session_state.nodes_raw = edited.copy()

# ==========================================
# APPLY MARKET FEEDBACK (CLEAN PIPELINE)
# ==========================================
init_market_state()
sync_markets()

def apply_market_intelligence(nodes: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    out = nodes.copy()
    if "Linked Sector" not in out.columns:
        out["Linked Sector"] = ""
    vol_map = dict(zip(market["Sector"], market["ChangePct"].abs() / 100.0))
    for i, row in out.iterrows():
        sector = str(row.get("Linked Sector", ""))
        vol = vol_map.get(sector, 0.05)
        # Higher market volatility increases the value of risk reduction on that node
        boost = 1.0 + (vol * 0.25)
        out.at[i, "Risk Reduction (%)"] = float(row["Risk Reduction (%)"]) * boost
    return out

nodes_adj = apply_market_intelligence(edited, st.session_state.market_df)

# Guard
if nodes_adj.empty or "Node Name" not in nodes_adj.columns:
    st.error("No valid nodes. Add at least one node.")
    st.stop()

nodes = nodes_adj["Node Name"].dropna().astype(str).tolist()
if not nodes:
    st.error("Node names are empty.")
    st.stop()

costs = dict(zip(nodes, nodes_adj["Cost"]))
risks = dict(zip(nodes, nodes_adj["Risk Reduction (%)"]))
leads = dict(zip(nodes, nodes_adj["Lead Time Saved (Days)"]))
carbons = dict(zip(nodes, nodes_adj["Carbon Impact (Tons)"]))
actions = dict(zip(nodes, nodes_adj.get("Action", ["Intervention"] * len(nodes))))

# Diminishing utility
marginal_risk = {n: float((macro_mult * risks[n]) ** 0.85) for n in nodes}

# ==========================================
# DEPENDENCIES + BUNDLES + NETWORKX
# ==========================================
st.sidebar.markdown("---")
st.sidebar.subheader("Dependencies & Bundles")

if "deps_df" not in st.session_state:
    st.session_state.deps_df = pd.DataFrame(columns=["Dependent", "Prerequisite"])
deps = st.sidebar.data_editor(st.session_state.deps_df, num_rows="dynamic", key="deps")
st.session_state.deps_df = deps

if "bundles_df" not in st.session_state:
    st.session_state.bundles_df = pd.DataFrame([{"Bundle": "Core Resilience Package", "Discount": 45000}])
bundles = st.sidebar.data_editor(st.session_state.bundles_df, num_rows="dynamic", key="bundles")
st.session_state.bundles_df = bundles

# Build NetworkX graph
G = nx.DiGraph()
G.add_nodes_from(nodes)
for _, row in deps.iterrows():
    d = str(row.get("Dependent", "")).strip()
    p = str(row.get("Prerequisite", "")).strip()
    if d in nodes and p in nodes and d != p:
        G.add_edge(p, d)

has_cycle = not nx.is_directed_acyclic_graph(G)
criticality = {}
if not has_cycle and len(G) > 0:
    # simple criticality = number of downstream nodes
    for n in nodes:
        criticality[n] = len(nx.descendants(G, n))
else:
    criticality = {n: 0 for n in nodes}

# ==========================================
# OPTIMIZATION (PuLP)
# ==========================================
st.title("Supply Chain Resilience & Capital Allocation Engine")
st.caption("Hybrid MILP + Correlated Monte Carlo + Live Market & Macro Intelligence")

prob = pl.LpProblem("Capital_Allocation", pl.LpMinimize if target_mode else pl.LpMaximize)

y = pl.LpVariable.dicts("select", nodes, cat="Binary")
x = pl.LpVariable.dicts("scale", nodes, lowBound=0, upBound=1, cat="Continuous")

for n in nodes:
    prob += x[n] >= 0.30 * y[n]
    prob += x[n] <= y[n]

# Dependencies
for _, row in deps.iterrows():
    d = str(row.get("Dependent", "")).strip()
    p = str(row.get("Prerequisite", "")).strip()
    if d in nodes and p in nodes:
        prob += x[d] <= x[p]

# Bundles
bundle_vars = {}
for idx, row in bundles.iterrows():
    name = str(row.get("Bundle", f"B{idx}")).strip()
    disc = float(row.get("Discount", 0) or 0)
    if not name:
        continue
    bvar = pl.LpVariable(f"bundle_{idx}", cat="Binary")
    bundle_vars[name] = {"var": bvar, "discount": disc}
    # For simplicity require first two nodes (user can expand later)
    req = nodes[:2] if len(nodes) >= 2 else nodes
    for r in req:
        prob += bvar <= y[r]

total_discount = pl.lpSum(b["var"] * b["discount"] for b in bundle_vars.values()) if bundle_vars else 0

if target_mode:
    # Minimize capital subject to risk target
    prob += pl.lpSum(costs[n] * x[n] for n in nodes) - total_discount
    needed = max(0.0, effective_baseline - target_risk)
    prob += pl.lpSum(marginal_risk[n] * x[n] for n in nodes) >= needed
else:
    # Maximize utility
    # Utility = risk + lead time - carbon penalty
    util = []
    for n in nodes:
        u = (opt_weight * marginal_risk[n] +
             (1 - opt_weight) * leads[n] -
             carbon_weight * (carbons[n] / 100.0))
        util.append(u * x[n])
    prob += pl.lpSum(util)
    prob += pl.lpSum(costs[n] * x[n] for n in nodes) - total_discount <= budget

status = prob.solve(pl.PULP_CBC_CMD(msg=False, timeLimit=20))

# ==========================================
# RESULTS
# ==========================================
scales = {n: max(0.0, pl.value(x[n]) or 0.0) for n in nodes}
active = {n: s for n, s in scales.items() if s > 0.01}

base_spend = sum(costs[n] * s for n, s in scales.items())
disc_applied = sum(b["discount"] for b in bundle_vars.values() if pl.value(b["var"]) > 0.5)
final_spend = max(0.0, base_spend - disc_applied)

risk_drop = sum(marginal_risk[n] * scales[n] for n in nodes)
risk_drop = min(effective_baseline, risk_drop)
final_risk = max(0.0, effective_baseline - risk_drop)
lead_saved = sum(leads[n] * scales[n] for n in nodes)
carbon_total = sum(carbons[n] * scales[n] for n in nodes)

# Capital efficiency
eff = (risk_drop / (final_spend / 1000)) if final_spend > 0 else 0.0

# ==========================================
# MONTE CARLO (risk + cost)
# ==========================================
def run_mc(n_iter: int = 8000):
    np.random.seed(42)
    k = len(nodes)
    if k == 0:
        return final_risk, final_risk, final_spend, final_spend
    # simple correlation
    corr = np.eye(k) * 0.15 + np.eye(k) * 0.85
    try:
        L = np.linalg.cholesky(corr)
    except:
        L = np.eye(k)
    risks_out = []
    costs_out = []
    for _ in range(n_iter):
        z = L @ np.random.normal(size=k)
        shock_r = 1.0 + 0.11 * z
        shock_c = 1.0 + 0.08 * z
        drop = sum(marginal_risk[n] * scales[n] * max(0.35, shock_r[i]) for i, n in enumerate(nodes))
        c = sum(costs[n] * scales[n] * max(0.7, shock_c[i]) for i, n in enumerate(nodes))
        risks_out.append(max(0.0, effective_baseline - drop))
        costs_out.append(c)
    return (np.percentile(risks_out, 50), np.percentile(risks_out, 90),
            np.percentile(costs_out, 50), np.percentile(costs_out, 90))

p50_r, p90_r, p50_c, p90_c = run_mc(mc_runs)

# ==========================================
# DASHBOARD
# ==========================================
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Effective Baseline Risk", f"{effective_baseline:.1f}%")
c2.metric("Optimized Risk", f"{final_risk:.1f}%", delta=f"-{risk_drop:.1f} pts", delta_color="inverse")
c3.metric("Capital Deployed", f"${final_spend:,.0f}")
c4.metric("Capital Efficiency", f"{eff:.2f} pts / $1k")
c5.metric("Carbon Impact", f"{carbon_total:.0f} t")

st.markdown("##### Institutional Risk & Cost View")
r1, r2, r3, r4 = st.columns(4)
r1.metric("P50 Risk", f"{p50_r:.1f}%")
r2.metric("P90 Risk (VaR)", f"{p90_r:.1f}%")
r3.metric("P50 Cost", f"${p50_c:,.0f}")
r4.metric("P90 Cost", f"${p90_c:,.0f}")

if has_cycle:
    st.warning("Dependency cycle detected — review prerequisites.")

# Recommendation block (CFO language)
st.markdown("---")
st.subheader("CFO Recommendation Summary")
if final_spend == 0:
    st.info("No capital allocated under current constraints.")
else:
    st.success(
        f"**Recommended capital plan:** Deploy **${final_spend:,.0f}** across {len(active)} interventions. "
        f"Expected risk reduction of **{risk_drop:.1f} points** (to {final_risk:.1f}%). "
        f"Capital efficiency: **{eff:.2f} risk points per $1,000**. "
        f"P90 residual risk: **{p90_r:.1f}%**. "
        f"Associated carbon: **{carbon_total:.0f} tons**."
    )

# Portfolio table
if active:
    rows = []
    for n, s in sorted(active.items(), key=lambda x: -x[1]):
        rows.append({
            "Node": n,
            "Action": actions.get(n, ""),
            "Scale": f"{s*100:.0f}%",
            "Capital": f"${costs[n]*s:,.0f}",
            "Risk Red.": f"{risks[n]*s:.1f}%",
            "Lead Time": f"{leads[n]*s:.1f}d",
            "Carbon": f"{carbons[n]*s:.0f}t",
            "Criticality": criticality.get(n, 0)
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# Network insight
with st.expander("Dependency Graph Insights (NetworkX)"):
    st.write(f"Nodes: {G.number_of_nodes()} | Edges: {G.number_of_edges()}")
    if criticality:
        top = sorted(criticality.items(), key=lambda x: -x[1])[:5]
        st.write("Highest downstream criticality:", top)

st.caption("Engine uses: Supabase • yfinance • Oil macro • PuLP MILP • NumPy Cholesky MC • NetworkX • ReportLab • Pandas • Streamlit • File schema normalization")
