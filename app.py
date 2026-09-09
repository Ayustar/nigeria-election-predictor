import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.graph_objects as go

# Load models and features
model_apc = joblib.load('models/model_apc.pkl')
model_pdp = joblib.load('models/model_pdp.pkl')
scaler = joblib.load('models/scaler_pdp.pkl')

with open('data/features.json', 'r') as f:
    features = json.load(f)

# State data
state_data = {
    'Abia': {'zone': 4, 'state_avg_apc': 7.59, 'state_avg_pdp': 66.90, 'prev_apc': 26.31, 'prev_pdp': 67.96, 'prev_third': 3.73, 'zone_year_apc': 8.45, 'zone_year_pdp': 21.54},
    'Adamawa': {'zone': 1, 'state_avg_apc': 42.33, 'state_avg_pdp': 51.23, 'prev_apc': 46.59, 'prev_pdp': 50.55, 'prev_third': 2.86, 'zone_year_apc': 43.21, 'zone_year_pdp': 48.32},
    'Akwa Ibom': {'zone': 5, 'state_avg_apc': 16.29, 'state_avg_pdp': 77.95, 'prev_apc': 30.31, 'prev_pdp': 65.38, 'prev_third': 4.31, 'zone_year_apc': 24.13, 'zone_year_pdp': 52.17},
    'Anambra': {'zone': 4, 'state_avg_apc': 5.12, 'state_avg_pdp': 55.63, 'prev_apc': 5.50, 'prev_pdp': 91.06, 'prev_third': 1.44, 'zone_year_apc': 8.45, 'zone_year_pdp': 21.54},
    'Bauchi': {'zone': 1, 'state_avg_apc': 56.74, 'state_avg_pdp': 38.52, 'prev_apc': 77.95, 'prev_pdp': 19.82, 'prev_third': 2.23, 'zone_year_apc': 43.21, 'zone_year_pdp': 48.32},
    'Bayelsa': {'zone': 5, 'state_avg_apc': 20.14, 'state_avg_pdp': 74.23, 'prev_apc': 36.93, 'prev_pdp': 61.65, 'prev_third': 1.42, 'zone_year_apc': 24.13, 'zone_year_pdp': 52.17},
    'Benue': {'zone': 2, 'state_avg_apc': 38.92, 'state_avg_pdp': 54.18, 'prev_apc': 47.70, 'prev_pdp': 48.41, 'prev_third': 3.89, 'zone_year_apc': 35.67, 'zone_year_pdp': 45.23},
    'Borno': {'zone': 1, 'state_avg_apc': 65.43, 'state_avg_pdp': 31.24, 'prev_apc': 90.94, 'prev_pdp': 7.45, 'prev_third': 1.61, 'zone_year_apc': 43.21, 'zone_year_pdp': 48.32},
    'Cross River': {'zone': 5, 'state_avg_apc': 22.18, 'state_avg_pdp': 71.34, 'prev_apc': 27.80, 'prev_pdp': 69.42, 'prev_third': 2.78, 'zone_year_apc': 24.13, 'zone_year_pdp': 52.17},
    'Delta': {'zone': 5, 'state_avg_apc': 18.43, 'state_avg_pdp': 75.82, 'prev_apc': 26.67, 'prev_pdp': 70.23, 'prev_third': 3.10, 'zone_year_apc': 24.13, 'zone_year_pdp': 52.17},
    'Ebonyi': {'zone': 4, 'state_avg_apc': 18.92, 'state_avg_pdp': 72.43, 'prev_apc': 21.40, 'prev_pdp': 76.82, 'prev_third': 1.78, 'zone_year_apc': 8.45, 'zone_year_pdp': 21.54},
    'Edo': {'zone': 5, 'state_avg_apc': 38.64, 'state_avg_pdp': 55.23, 'prev_apc': 52.41, 'prev_pdp': 44.32, 'prev_third': 3.27, 'zone_year_apc': 24.13, 'zone_year_pdp': 52.17},
    'Ekiti': {'zone': 3, 'state_avg_apc': 52.34, 'state_avg_pdp': 43.21, 'prev_apc': 53.68, 'prev_pdp': 43.82, 'prev_third': 2.50, 'zone_year_apc': 55.43, 'zone_year_pdp': 36.21},
    'Enugu': {'zone': 4, 'state_avg_apc': 6.43, 'state_avg_pdp': 88.32, 'prev_apc': 8.23, 'prev_pdp': 89.54, 'prev_third': 2.23, 'zone_year_apc': 8.45, 'zone_year_pdp': 21.54},
    'FCT': {'zone': 2, 'state_avg_apc': 32.14, 'state_avg_pdp': 52.43, 'prev_apc': 35.00, 'prev_pdp': 58.32, 'prev_third': 6.68, 'zone_year_apc': 35.67, 'zone_year_pdp': 45.23},
    'Gombe': {'zone': 1, 'state_avg_apc': 52.43, 'state_avg_pdp': 43.21, 'prev_apc': 62.30, 'prev_pdp': 34.82, 'prev_third': 2.88, 'zone_year_apc': 43.21, 'zone_year_pdp': 48.32},
    'Imo': {'zone': 4, 'state_avg_apc': 28.43, 'state_avg_pdp': 55.32, 'prev_apc': 42.13, 'prev_pdp': 51.23, 'prev_third': 6.64, 'zone_year_apc': 8.45, 'zone_year_pdp': 21.54},
    'Jigawa': {'zone': 0, 'state_avg_apc': 64.32, 'state_avg_pdp': 33.21, 'prev_apc': 76.23, 'prev_pdp': 21.54, 'prev_third': 2.23, 'zone_year_apc': 61.23, 'zone_year_pdp': 35.43},
    'Kaduna': {'zone': 0, 'state_avg_apc': 55.43, 'state_avg_pdp': 41.23, 'prev_apc': 66.28, 'prev_pdp': 31.23, 'prev_third': 2.49, 'zone_year_apc': 61.23, 'zone_year_pdp': 35.43},
    'Kano': {'zone': 0, 'state_avg_apc': 64.74, 'state_avg_pdp': 13.83, 'prev_apc': 77.45, 'prev_pdp': 20.71, 'prev_third': 1.84, 'zone_year_apc': 61.23, 'zone_year_pdp': 35.43},
    'Katsina': {'zone': 0, 'state_avg_apc': 63.21, 'state_avg_pdp': 33.54, 'prev_apc': 80.32, 'prev_pdp': 18.23, 'prev_third': 1.45, 'zone_year_apc': 61.23, 'zone_year_pdp': 35.43},
    'Kebbi': {'zone': 0, 'state_avg_apc': 58.43, 'state_avg_pdp': 38.21, 'prev_apc': 74.23, 'prev_pdp': 23.54, 'prev_third': 2.23, 'zone_year_apc': 61.23, 'zone_year_pdp': 35.43},
    'Kogi': {'zone': 2, 'state_avg_apc': 48.32, 'state_avg_pdp': 46.23, 'prev_apc': 55.43, 'prev_pdp': 40.32, 'prev_third': 4.25, 'zone_year_apc': 35.67, 'zone_year_pdp': 45.23},
    'Kwara': {'zone': 2, 'state_avg_apc': 48.23, 'state_avg_pdp': 47.32, 'prev_apc': 54.32, 'prev_pdp': 41.23, 'prev_third': 4.45, 'zone_year_apc': 35.67, 'zone_year_pdp': 45.23},
    'Lagos': {'zone': 3, 'state_avg_apc': 58.32, 'state_avg_pdp': 35.43, 'prev_apc': 66.32, 'prev_pdp': 28.54, 'prev_third': 5.14, 'zone_year_apc': 55.43, 'zone_year_pdp': 36.21},
    'Nasarawa': {'zone': 2, 'state_avg_apc': 46.23, 'state_avg_pdp': 49.32, 'prev_apc': 55.32, 'prev_pdp': 41.23, 'prev_third': 3.45, 'zone_year_apc': 35.67, 'zone_year_pdp': 45.23},
    'Niger': {'zone': 2, 'state_avg_apc': 52.43, 'state_avg_pdp': 43.21, 'prev_apc': 62.32, 'prev_pdp': 34.23, 'prev_third': 3.45, 'zone_year_apc': 35.67, 'zone_year_pdp': 45.23},
    'Ogun': {'zone': 3, 'state_avg_apc': 57.32, 'state_avg_pdp': 37.43, 'prev_apc': 65.43, 'prev_pdp': 30.23, 'prev_third': 4.34, 'zone_year_apc': 55.43, 'zone_year_pdp': 36.21},
    'Ondo': {'zone': 3, 'state_avg_apc': 52.43, 'state_avg_pdp': 37.32, 'prev_apc': 62.32, 'prev_pdp': 32.43, 'prev_third': 5.25, 'zone_year_apc': 55.43, 'zone_year_pdp': 36.21},
    'Osun': {'zone': 3, 'state_avg_apc': 52.32, 'state_avg_pdp': 42.43, 'prev_apc': 60.23, 'prev_pdp': 36.54, 'prev_third': 3.77, 'zone_year_apc': 55.43, 'zone_year_pdp': 36.21},
    'Oyo': {'zone': 3, 'state_avg_apc': 53.43, 'state_avg_pdp': 40.32, 'prev_apc': 62.43, 'prev_pdp': 33.21, 'prev_third': 4.57, 'zone_year_apc': 55.43, 'zone_year_pdp': 36.21},
    'Plateau': {'zone': 2, 'state_avg_apc': 44.32, 'state_avg_pdp': 50.43, 'prev_apc': 52.43, 'prev_pdp': 43.21, 'prev_third': 4.36, 'zone_year_apc': 35.67, 'zone_year_pdp': 45.23},
    'Rivers': {'zone': 5, 'state_avg_apc': 32.43, 'state_avg_pdp': 61.32, 'prev_apc': 37.32, 'prev_pdp': 56.43, 'prev_third': 6.25, 'zone_year_apc': 24.13, 'zone_year_pdp': 52.17},
    'Sokoto': {'zone': 0, 'state_avg_apc': 52.43, 'state_avg_pdp': 44.32, 'prev_apc': 60.23, 'prev_pdp': 37.43, 'prev_third': 2.34, 'zone_year_apc': 61.23, 'zone_year_pdp': 35.43},
    'Taraba': {'zone': 1, 'state_avg_apc': 38.32, 'state_avg_pdp': 57.43, 'prev_apc': 43.21, 'prev_pdp': 53.32, 'prev_third': 3.47, 'zone_year_apc': 43.21, 'zone_year_pdp': 48.32},
    'Yobe': {'zone': 1, 'state_avg_apc': 62.43, 'state_avg_pdp': 34.32, 'prev_apc': 78.32, 'prev_pdp': 19.43, 'prev_third': 2.25, 'zone_year_apc': 43.21, 'zone_year_pdp': 48.32},
    'Zamfara': {'zone': 0, 'state_avg_apc': 60.32, 'state_avg_pdp': 36.43, 'prev_apc': 72.32, 'prev_pdp': 25.43, 'prev_third': 2.25, 'zone_year_apc': 61.23, 'zone_year_pdp': 35.43},
}

# Page config
st.set_page_config(page_title="Nigeria Election Predictor", layout="wide")

st.title("🗳️ Nigeria Presidential Election Predictor")
st.caption("Predicting 2023 state-level vote share using ML — APC, PDP, and LP")

st.divider()

# Sidebar
with st.sidebar:
    st.header("State & Scenario Inputs")

    selected_state = st.selectbox("Select State", sorted(state_data.keys()))

    st.subheader("Adjust Scenario")
    turnout = st.slider("Voter Turnout (%)", 10.0, 60.0, 26.93)
    gov_party = st.selectbox("Governor's Party", ["APC", "PDP", "Other"])
    incumbent = st.selectbox("Federal Incumbent Party", ["APC", "PDP"])

# Encode inputs
gov_encoded = 0 if gov_party == "APC" else 1 if gov_party == "PDP" else 2
incumbent_encoded = 1 if incumbent == "APC" else 0
gov_aligns = 1 if gov_party == incumbent else 0

state = state_data[selected_state]

security = {2023: 751}[2023]

input_data = pd.DataFrame([{
    'Zone_Encoded': state['zone'],
    'Turnout_%': turnout,
    'Gov_Party_Encoded': gov_encoded,
    'Incumbent_Encoded': incumbent_encoded,
    'Gov_Aligns_Incumbent': gov_aligns,
    'National_Security_Incidents': security,
    'Prev_APC_%': state['prev_apc'],
    'Prev_PDP_%': state['prev_pdp'],
    'Prev_Third_Party_%': state['prev_third'],
    'Third_Party_%': 28.53,
    'State_Avg_APC_%': state['state_avg_apc'],
    'State_Avg_PDP_%': state['state_avg_pdp'],
    'Zone_Year_APC_%': state['zone_year_apc'],
    'Zone_Year_PDP_%': state['zone_year_pdp'],
}])

# Predictions
apc_pred = float(model_apc.predict(input_data)[0])
pdp_pred = float(model_pdp.predict(scaler.transform(input_data))[0])
lp_pred = max(0, 100 - apc_pred - pdp_pred)
others_pred = max(0, 100 - apc_pred - pdp_pred - lp_pred)

# Normalize
total = apc_pred + pdp_pred + lp_pred + others_pred
apc_pred = (apc_pred / total) * 100
pdp_pred = (pdp_pred / total) * 100
lp_pred = (lp_pred / total) * 100

# Main display
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("APC", f"{apc_pred:.1f}%")
with col2:
    st.metric("PDP", f"{pdp_pred:.1f}%")
with col3:
    st.metric("LP", f"{lp_pred:.1f}%")

st.divider()

# Bar chart
fig = go.Figure(go.Bar(
    x=['APC', 'PDP', 'LP'],
    y=[apc_pred, pdp_pred, lp_pred],
    marker_color=['#004C97', '#CC0000', '#008000'],
    text=[f"{apc_pred:.1f}%", f"{pdp_pred:.1f}%", f"{lp_pred:.1f}%"],
    textposition='outside'
))

fig.update_layout(
    title=f"Predicted 2023 Vote Share — {selected_state}",
    yaxis_title="Vote Share (%)",
    yaxis=dict(range=[0, 100]),
    plot_bgcolor='white',
    height=400
)

st.plotly_chart(fig, use_container_width=True)

st.caption("⚠️ LP% is derived from remaining vote share after APC and PDP predictions. Model trained on 2011–2019 elections, validated against 2023.")
