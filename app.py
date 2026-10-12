
import streamlit as st
import pandas as pd
import joblib

# Set page configuration
st.set_page_config(
    page_title="Health Insurance Premium Estimator",
    page_icon="🏥",
    layout="centered"
)

st.title("🏥 Health Insurance Premium Estimator")
st.write("Enter your details below to calculate your estimated annual health insurance premium.")

# Part 2 & Part 3: Load the trained pipeline
@st.cache_resource
def load_model():
    return joblib.load("insurance_model.pkl")

try:
    model = load_model()
except FileNotFoundError:
    st.error("Error: `insurance_model.pkl` not found. Please run `train.py` first to generate the model file.")
    st.stop()

# Part 2: Arrange input controls in two side-by-side columns
col1, col2 = st.columns(2)

with col1:
    # Part 3: Safety Rules added (min_value, max_value, step) to prevent invalid inputs
    age = st.number_input(
        "Age", 
        min_value=18, 
        max_value=100, 
        value=30, 
        step=1, 
        help="Age must be between 18 and 100."
    )
    
    sex = st.selectbox(
        "Sex", 
        options=["male", "female"]
    )
    
    bmi = st.number_input(
        "BMI (Body Mass Index)", 
        min_value=10.0, 
        max_value=60.0, 
        value=25.0, 
        step=0.1, 
        format="%.1f",
        help="BMI must be between 10.0 and 60.0."
    )

with col2:
    children = st.number_input(
        "Number of Children / Dependents", 
        min_value=0, 
        max_value=10, 
        value=0, 
        step=1,
        help="Number of dependents must be between 0 and 10."
    )
    
    smoker = st.selectbox(
        "Smoker", 
        options=["no", "yes"]
    )
    
    region = st.selectbox(
        "Region", 
        options=["southwest", "southeast", "northwest", "northeast"]
    )

# Part 2: Prediction trigger button
st.markdown("---")
if st.button("Calculate Estimated Premium", type="primary", use_container_width=True):
    # Part 2: Input Data organized into a DataFrame with exact matching column names and data types
    input_data = pd.DataFrame([{
        "age": int(age),
        "sex": str(sex),
        "bmi": float(bmi),
        "children": int(children),
        "smoker": str(smoker),
        "region": str(region)
    }])
    
    # Generate prediction using loaded model pipeline
    prediction = model.predict(input_data)[0]
    
    # Display formatted result
    st.success(f"### Estimated Yearly Premium: **${prediction:,.2f}**")
