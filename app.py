import streamlit as st
from supabase import create_client, Client
import pandas as pd
import numpy as np
import pulp as pl

# 1. Page Configuration (Must be the first Streamlit command)
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
                
    st.stop() # Halts app rendering until user is logged in

# --- LOGGED IN USER APP INTERFACE ---
st.sidebar.write(f"👤 Logged in as: {st.session_state.user.email}")
if st.sidebar.button("Log Out"):
    supabase.auth.sign_out()
    st.session_state.user = None
    st.rerun()

st.title("Supply Chain Resilience & Capital Allocation Engine")
st.markdown("*Commercial-Grade Optimization Suite with Cloud Persistence & AI Guidance.*")

# --- SIMULATION DATA & CONTROLS ---
st.sidebar.markdown("---")
st.sidebar.header("⚙️ Executive Controls")
total_budget = st.sidebar.slider("Total Capital Budget ($)", min_value=100000, max_value=2000000, value=500000, step=50000)
max_risk_tolerance = st.sidebar.slider("Max Acceptable Risk (%)", min_value=5.0, max_value=50.0, value=20.0, step=1.0)

# Mocking internal supply chain optimization values for demonstration
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

# Sample Dataframe for the optimization matrix
df_nodes = pd.DataFrame({
    "Intervention Node": nodes,
    "Cost ($)": [120000, 250000, 90000, 140000],
    "Risk Reduction (%)": [12.5, 18.0, 7.5, 10.0],
    "Status": ["Recommended", "Optimal", "Pending", "Recommended"]
})
st.dataframe(df_nodes, use_container_width=True)

# --- INTERACTIVE AI ASSISTANT AGENT ---
with st.sidebar:
    st.markdown("---")
    st.subheader("🤖 Supply Chain AI Copilot")
    st.markdown("*Ask me how to lower risk, optimize budget, or configure your constraints.*")

    # Initialize chat history in session state if it doesn't exist
    if "ai_messages" not in st.session_state:
        st.session_state.ai_messages = [
            {
                "role": "assistant",
                "content": "Hello! I'm your supply chain optimization copilot. How can I help you allocate your capital today?",
            }
        ]

    # Display chat history container
    chat_container = st.container(height=250)
    with chat_container:
        for message in st.session_state.ai_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Chat input box for the user
    user_query = st.chat_input("Ask your copilot...", key="copilot_input")

    if user_query:
        # Append user message
        st.session_state.ai_messages.append({"role": "user", "content": user_query})

        # Generate smart dynamic responses based on live app variables
        query_lower = user_query.lower()
        if "budget" in query_lower:
            response = f"Your current budget cap is set to ${total_budget:,.2f}. You can adjust this using the slider in the Executive Control Panel above."
        elif "risk" in query_lower:
            response = f"Your baseline system risk is {baseline_risk}%, and the optimized expected risk drops to {optimized_risk:.1f}%."
        elif "lead time" in query_lower or "time" in query_lower:
            response = f"Your active interventions are saving a total of {total_lead_time_saved} days across your logistics network."
        else:
            response = (
                f"I'm tracking your network with {len(nodes)} active intervention nodes. "
                "Try asking me about your **budget**, **risk reduction**, or **lead time**!"
            )

        # Append assistant response
        st.session_state.ai_messages.append({"role": "assistant", "content": response})
        st.rerun()
