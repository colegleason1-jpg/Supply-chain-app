import streamlit as st
from supabase import create_client, Client
import pandas as pd
import numpy as np

# 1. Page Configuration
st.set_page_config(page_title="Supply Chain Capital Allocation Engine", layout="wide")

# 2. Initialize Supabase Connection using Streamlit Secrets
@st.cache_resource
def init_supabase() -> Client:
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)

supabase = init_supabase()

# 3. Authentication State Management
if "user" not in st.session_state:
    st.session_state.user = None

if not st.session_state.user:
    st.title("🔐 Enterprise Login Required")
    st.markdown("Please sign in or register to access the Supply Chain Capital Allocation Engine.")
    
    auth_tab1, auth_tab2 = st.tabs(["Sign In", "Register"])
    
    with auth_tab1:
        signin_email = st.text_input("Email", key="si_email")
        signin_pass = st.text_input("Password", type="password", key="si_pass")
        if st.button("Log In"):
            try:
                res = supabase.auth.sign_in_with_password({"email": signin_email, "password": signin_pass})
                st.session_state.user = res.user
                st.success("Logged in successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"Login failed: {e}")
                
    with auth_tab2:
        reg_email = st.text_input("Email", key="reg_email")
        reg_pass = st.text_input("Password", type="password", key="reg_pass")
        if st.button("Create Account"):
            try:
                res = supabase.auth.sign_up({"email": reg_email, "password": reg_pass})
                st.success("Account created! You can now log in.")
            except Exception as e:
                st.error(f"Registration failed: {e}")
                
    st.stop()

# --- LOGGED IN USER APP INTERFACE ---
st.sidebar.write(f"👤 Logged in as: {st.session_state.user.email}")
if st.sidebar.button("Log Out"):
    supabase.auth.sign_out()
    st.session_state.user = None
    st.rerun()

st.title("Supply Chain Resilience & Capital Allocation Engine")
st.markdown("*Commercial-Grade Optimization Suite with Cloud Persistence & Expert AI Guidance.*")

# --- INITIALIZE SESSION STATE FOR DYNAMIC CONTROLS ---
if "dynamic_budget" not in st.session_state:
    st.session_state.dynamic_budget = 500000
if "dynamic_risk_tolerance" not in st.session_state:
    st.session_state.dynamic_risk_tolerance = 20.0
if "pending_action" not in st.session_state:
    st.session_state.pending_action = None

# --- SIMULATION DATA & CONTROLS ---
st.sidebar.markdown("---")
st.sidebar.header("⚙️ Executive Controls")

st.session_state.dynamic_budget = st.sidebar.slider(
    "Total Capital Budget ($)", 
    min_value=100000, 
    max_value=2000000, 
    value=st.session_state.dynamic_budget, 
    step=50000
)
st.session_state.dynamic_risk_tolerance = st.sidebar.slider(
    "Max Acceptable Risk (%)", 
    min_value=5.0, 
    max_value=50.0, 
    value=st.session_state.dynamic_risk_tolerance, 
    step=1.0
)

total_budget = st.session_state.dynamic_budget
max_risk_tolerance = st.session_state.dynamic_risk_tolerance

nodes = ["Supplier A (Domestic)", "Supplier B (Overseas)", "Warehouse Hub 1", "Distribution Center 2"]
baseline_risk = 34.5
optimized_risk = max(8.2, baseline_risk - (total_budget / 50000))
total_lead_time_saved = round(total_budget / 75000 * 4.2, 1)

# --- MAIN DASHBOARD METRICS ---
col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="Allocated Budget", value=f"${total_budget:,.2f}")
with col2:
    st.metric(label="Optimized Risk Score", value=f"{optimized_risk:.1f}%", delta=f"-{baseline_risk - optimized_risk:.1f}% vs Baseline")
with col3:
    st.metric(label="Projected Lead Time Savings", value=f"{total_lead_time_saved} Days")

st.markdown("---")
st.subheader("📊 Network Vulnerability & Optimization Breakdown")

df_nodes = pd.DataFrame({
    "Intervention Node": nodes,
    "Cost ($)": [120000, 250000, 90000, 140000],
    "Risk Reduction (%)": [12.5, 18.0, 7.5, 10.0],
    "Status": ["Recommended", "Optimal", "Pending", "Recommended"]
})
st.dataframe(df_nodes, use_container_width=True)

# --- INTUITIVE MASTER AI COPILOT AGENT ---
with st.sidebar:
    st.markdown("---")
    st.subheader("🤖 Master Supply Chain AI Expert")
    st.markdown("*Tell me your goal in plain English, and I'll adjust the engine for you.*")

    if "ai_messages" not in st.session_state:
        st.session_state.ai_messages = [
            {
                "role": "assistant",
                "content": "Hello! I'm your master operator. Tell me what you want to achieve (e.g., *'I want the lowest risk possible'* or *'Cut my budget down to 300k'*) and I'll configure everything for you.",
            }
        ]

    chat_container = st.container(height=300)
    with chat_container:
        for message in st.session_state.ai_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    user_query = st.chat_input("Tell your copilot what to do...", key="copilot_input")

    if user_query:
        st.session_state.ai_messages.append({"role": "user", "content": user_query})
        query_lower = user_query.lower()
        
        # 1. Handle Confirmations
        if st.session_state.pending_action and any(w in query_lower for w in ["yes", "do it", "confirm", "sure", "ok", "yep", "proceed"]):
            action = st.session_state.pending_action
            if action["type"] == "set_budget":
                st.session_state.dynamic_budget = action["value"]
                response = f"✅ Done! I've adjusted your capital budget to **${action['value']:,.2f}**."
            st.session_state.pending_action = None
            st.session_state.ai_messages.append({"role": "assistant", "content": response})
            st.rerun()
            
        elif st.session_state.pending_action and any(w in query_lower for w in ["no", "cancel", "wait", "stop"]):
            st.session_state.pending_action = None
            response = "❌ Cancelled. What would you like to do instead?"
            st.session_state.ai_messages.append({"role": "assistant", "content": response})
            st.rerun()

        # 2. Handle Direct Number Requests (e.g., "Set budget to 400000")
        elif "budget" in query_lower and any(char.isdigit() for char in query_lower):
            # Simple number extraction fallback
            import re
            numbers = re.findall(r'\d+', query_lower.replace(",", ""))
            if numbers:
                # Reconstruct full amount if abbreviated like '400k' or full digits
                val = int(''.join(numbers))
                if val < 2000: # if they said 400k
                    val *= 1000
                val = max(100000, min(2000000, val))
                st.session_state.dynamic_budget = val
                response = f"✅ I've updated your budget directly to **${val:,.2f}**!"
            else:
                response = "I noticed you mentioned budget, but what specific dollar amount would you like to set it to?"
                
        # 3. Handle Strategic Goals
        elif any(w in query_lower for w in ["lowest risk", "minimize risk", "safest", "reduce risk"]):
            target_budget = 1500000
            st.session_state.pending_action = {"type": "set_budget", "value": target_budget}
            response = (
                f"🛡️ To achieve maximum safety, we should scale your budget to **$1,500,000**, bringing down your system risk to a minimal level. "
                "Shall I apply this change now? (Type **Yes** or **No**)"
            )
            
        elif any(w in query_lower for w in ["save money", "cheaper", "lower budget", "cut cost", "spend less"]):
            target_budget = 250000
            st.session_state.pending_action = {"type": "set_budget", "value": target_budget}
            response = (
                f"💰 To optimize for cost savings while protecting core assets, I recommend dropping your capital allocation to **$250,000**. "
                "Shall I update the budget for you? (Type **Yes** or **No**)"
            )
            
        else:
            response = (
                "I want to make sure I get this right for you! Try giving me a direct instruction like:\n"
                "- *'Set budget to 750,000'*\n"
                "- *'I want the lowest risk configuration'*\n"
                "- *'Help me cut costs'* "
            )

        st.session_state.ai_messages.append({"role": "assistant", "content": response})
        st.rerun()
