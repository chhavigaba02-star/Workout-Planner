# app.py
import streamlit as st
# Importing features directly from your main file
from main import generate_plan, split_reasoning_and_plan, calculate_intensity_schedule

# Page Settings
st.set_page_config(page_title="AI Fitness Coach", page_icon="🏋️‍♂️", layout="centered")

st.title("🏋️‍♂️ Personalized AI Workout Planner")
st.write("Fill out your metrics to generate a custom routine and progressive overload timeline.")
st.divider()

# Input UI Controls 
col1, col2 = st.columns(2)

with col1:
    goal = st.text_input("What is your fitness goal?", placeholder="e.g., Weight loss, Hypertrophy")
    level = st.selectbox("Current Experience Level", ["Beginner", "Intermediate", "Advanced"])
    equipment = st.text_input("Equipment Available", placeholder="e.g., Dumbbells, Kettlebells, Gym, Bodyweight")

with col2:
    days = st.slider("Weekly Frequency (Days)", min_value=1, max_value=7, value=4)
    duration = st.slider("Session Duration (Minutes)", min_value=15, max_value=180, value=60, step=5)
    total_weeks = st.slider("Intensity Plan Length (Weeks)", min_value=3, max_value=12, value=6)

st.write("")
generate_btn = st.button("🚀 Generate My Custom Workout Plan", type="primary", use_container_width=True)

if generate_btn:
    if not goal or not equipment:
        st.error("Please provide both a training goal and your equipment options before submitting!")
    else:
        # Use loading spinners while local model processes inputs
        with st.spinner("Processing parameters with Coach AI and plotting charts..."):
            try:
                # 1. Trigger the logic from main.py
                raw_response = generate_plan(goal, level, equipment, days, duration)
                reasoning, plan = split_reasoning_and_plan(raw_response)
                
                # 2. Cleanup unexpected markdown asterisks programmatically
                reasoning_clean = reasoning.replace("**", "").replace("* ", "- ")
                plan_clean = plan.replace("**", "").replace("* ", "- ")
                
                # 3. Pull intensity calculations from main.py 
                intensity_df = calculate_intensity_schedule(level, total_weeks=total_weeks)
                
                st.success("Plan Generated Successfully!")
                st.divider()
                
                # Segment data out beautifully using clean Streamlit Tabs
                tab1, tab2, tab3 = st.tabs(["📋 Workout Routine", "📈 Progressive Intensity Schedule", "🧠 Coach Reasoning Log"])
                
                with tab1:
                    st.subheader("Your Custom Training Plan")
                    # st.text preserves raw newline structure without triggering broken markdown
                    st.text(plan_clean)
                    
                with tab2:
                    st.subheader("6-Week Progressive Overload Model")
                    st.write("Track your targets over time using this calculated dataset:")
                    st.dataframe(intensity_df, use_container_width=True)
                    st.caption("**What is RPE?** Rate of Perceived Exertion. A value of 7 implies finishing a set feeling like you could have executed exactly 3 more reps before muscular failure.")
                    
                with tab3:
                    st.subheader("Behind the Design Elements")
                    st.text(reasoning_clean)
                    
            except Exception as e:
                st.error(f"An error occurred while connecting to the core system engine: {e}")