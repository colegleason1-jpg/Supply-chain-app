else:
    total_budget = st.sidebar.number_input(
        "Total Budget Cap ($)",
        min_value=10000,
        max_value=20000000,  # <-- This is the 20 million cap
        value=750000,
        step=25000,
        format="%d",
    )
