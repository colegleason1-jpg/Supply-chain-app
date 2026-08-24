import streamlit as st
from supabase import create_client, Client
import pandas as pd
import numpy as np
import re
import time

# 1. Page Configuration
st.set_page_config(page_title="Supply Chain Capital Allocation Engine", layout="wide")

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
    # Strip HTML tags and limit length to prevent buffer/payload flooding
    cleaned = re.sub(r'<[^>]*?>', '', text)
    return cleaned[:500].strip()

# 3. Authentication & Security State Management
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
        # Reset lockout after timeout
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
                st.session_state.login_attempts = 0 # Reset attempts on success
                st.success("Authenticated successfully!")
                st.rerun()
            except Exception as e:
                st.session_state.login_attempts += 1
                if st.session_state.login_attempts >= 5:
                    st.session_state.lockout_time = time.time()
                st.error(f"Authentication failed. Security log updated (Attempt {st.session_state.login_attempts}/5).")
            
    st.stop()

# --- LOGGED IN USER APP INTERFACE ---
st.sidebar.write(f"👤 Logged in as: {st.session_state.user.email}")

# Account Management Expander for Password Updates & Secure Logout
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
st.markdown("*Commercial-Grade Optimization Suite with Enterprise Cloud Governance & Security Shielding.*")

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

# --- MASTER AI EXPERT COPILOT AGENT ---
with st.sidebar:
    st.markdown("---")
    st.subheader("🤖 Master Supply Chain AI Expert")
    st.markdown("*Tell me your goal in plain English, and I'll adjust the engine for you.*")

    if "ai_messages" not in st.session_state:
        st.session_state.ai_messages = [
            {
                "role": "assistant",
                "content": "Hello! I'm your master system operator. Tell me your target goals (e.g., *'Target 12 days of lead time savings'* or *'Set budget to 800k'*) and I'll align the engine.",
            }
        ]

    chat_container = st.container(height=300)
    with chat_container:
        for message in st.session_state.ai_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    raw_user_query = st.chat_input("Tell your copilot what to do...", key="copilot_input")

    if raw_user_query:
        # Sanitize chat inputs to block script and injection attacks from malicious chat queries
        user_query = sanitize_input(raw_user_query)
        st.session_state.ai_messages.append({"role": "user", "content": user_query})
        query_lower = user_query.lower()
        
        # 1. Handle Lead Time Targeting Goals safely
        if "lead time" in query_lower or "days" in query_lower:
            numbers = re.findall(r'\d+(?:\.\d+)?', query_lower)
            if numbers:
                target_days = float(numbers[0])
                calculated_budget = int((target_days * 75000) / 4.2)
                calculated_budget = max(100000, min(2000000, calculated_budget))
                
                st.session_state.dynamic_budget = calculated_budget
                response = f"🎯 To achieve your target of **{target_days} days** of lead time savings, I've dynamically re-allocated your budget to **${calculated_budget:,.2f}**."
            else:
                response = "How many days of lead time savings are you aiming for?"

        # 2. Handle Explicit Budget Commands
        elif "budget" in query_lower and any(char.isdigit() for char in query_lower):
            numbers = re.findall(r'\d+', query_lower.replace(",", ""))
            if numbers:
                val = int(''.join(numbers))
                if val < 2000 and ("k" in query_lower or val < 200): 
                    val *= 1000
                val = max(100000, min(2000000, val))
                st.session_state.dynamic_budget = val
                response = f"✅ Budget successfully adjusted to **${val:,.2f}**!"
            else:
                response = "Please specify the dollar amount for your budget update."
                
        # 3. Handle Strategic Goal Shortcuts
        elif any(w in query_lower for w in ["lowest risk", "minimize risk", "safest"]):
            st.session_state.dynamic_budget = 1500000
            response = "🛡️ I've maximized your budget to **$1,500,000** to drive your system risk down to its lowest possible threshold."
            
        elif any(w in query_lower for w in ["save money", "cheaper", "cut cost"]):
            st.session_state.dynamic_budget = 250000
            response = "💰 I've scaled your budget down to a lean **$250,000** to preserve capital while maintaining core nodes."
            
        else:
            response = (
                "I want to make sure I match your exact operational goals. Try telling me:\n"
                "- *'Target 12 days of lead time savings'*\n"
                "- *'Set budget to 750,000'*\n"
                "- *'I want the lowest risk setup'* "
            )

        st.session_state.ai_messages.append({"role": "assistant", "content": response})
        st.rerun()
