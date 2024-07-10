import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load models and preprocessor
with open('rf_model.pkl', 'rb') as rf_file:
    rf_model = pickle.load(rf_file)

with open('xgb_model.pkl', 'rb') as xgb_file:
    xgb_model = pickle.load(xgb_file)

with open('preprocessor.pkl', 'rb') as preprocessor_file:
    preprocessor = pickle.load(preprocessor_file)

with open('label_encoder.pkl', 'rb') as le_file:
    le = pickle.load(le_file)

# Page configuration
st.set_page_config(
    page_title="Bank Marketing Prediction",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Day mapping
day_mapping = {'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4, 'Friday': 5}

# App layout with columns
col1, col2, col3 = st.columns([1, 2, 1])

with col1:
    image_path_sidebar = 'bullseye.png'
    st.sidebar.image(image_path_sidebar, use_column_width=True)
    st.sidebar.title("Model Selection")
    model_name = st.sidebar.selectbox("Choose a Model", ["Random Forest", "XGBoost"])

with col2:
    # Use the absolute path to the image file
    image_path = 'C:/Users/gkmur/Downloads/Bootcamp/Python data/Supervised Machine Learning/Final_Project/investment.png'
    st.image(image_path, use_column_width=True)
    st.title("SavvyInvestments")

    # Collect user input
    duration = st.number_input("Duration(seconds)")
    campaign = st.number_input("Campaign", min_value=1, max_value=100, step=1)
    age = st.number_input('Age', min_value=0)
    balance = st.number_input("Balance (Euros)")
    day = st.selectbox("Day", list(day_mapping.keys()))
    previous_outcome = st.selectbox("Previous Outcome", ['unknown', 'success', 'failure', 'other'])
    month = st.selectbox("Month", ['apr', 'dec', 'jan', 'sep', 'feb', 'jun', 'unknown', 'nov', 'aug', 'oct', 'jul', 'mar', 'may'])
    previous = st.slider("Previous", 0, 100)
    pdays = st.number_input("Pdays", min_value=-1, max_value=999, step=1)
    education = st.selectbox("Education", ['primary', 'unknown', 'secondary', 'tertiary'])
    marital = st.selectbox("Marital Status", ['divorced', 'unknown', 'single', 'married'])
    job = st.selectbox("Job", ['admin.', 'housemaid', 'unemployed', 'self-employed', 'entrepreneur', 'retired', 'unknown', 'student', 'services', 'technician', 'management', 'blue-collar'])

# Create input DataFrame
input_data = pd.DataFrame({
    'duration': [duration],
    'campaign': [campaign],
    'age': [age],
    'balance': [balance],
    'day': [day_mapping[day]],
    'poutcome': [previous_outcome],
    'month': [month],
    'previous': [previous],
    'pdays': [pdays],
    'education': [education],
    'marital': [marital],
    'job': [job],
})

# Encode boolean columns as 0/1
input_data['default'] = 0  # Placeholder as it's not a top feature
input_data['housing'] = 0  # Placeholder as it's not a top feature
input_data['loan'] = 0  # Placeholder as it's not a top feature
input_data['contact'] = 0  # Placeholder as it's not a top feature

# Preprocess the input data
input_data.replace("unknown", np.nan, inplace=True)  # Replace 'unknown' with NaN
input_data_processed = preprocessor.transform(input_data)

# Select the top features
top_features = ['num__duration', 'num__campaign', 'num__age','num__balance', 'num__day', 
                'cat__poutcome_success', 'ord__month', 'num__previous', 'num__pdays', 
                'ord__education', 'ord__marital', 'ord__job']

# Get the indices of the top features
feature_names = preprocessor.get_feature_names_out()
top_feature_indices = [list(feature_names).index(feat) for feat in top_features]
input_data_top_features = input_data_processed[:, top_feature_indices]

with col3:
    if st.button("Predict"):
        # Predictions based on selected model
        if model_name == "Random Forest":
            prediction = rf_model.predict(input_data_top_features)[0]
        elif model_name == "XGBoost":
            prediction = xgb_model.predict(input_data_top_features)[0]

        # Decode the prediction
        prediction = le.inverse_transform([prediction])[0]

        # Display results
        if prediction == 'yes':
            st.markdown("<h2 style='text-align: center; color: green;'>The customer would be interested in a fixed deposit account. Please arrange a call with the customer.</h2>", unsafe_allow_html=True)
        else:
            st.markdown("<h2 style='text-align: center; color: red;'>The customer would not be interested in a fixed deposit account. No need to arrange a call with the customer.</h2>", unsafe_allow_html=True)
