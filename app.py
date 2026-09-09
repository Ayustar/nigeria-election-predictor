import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.graph_objects as go

# Load models and data
model_apc = joblib.load('models/model_apc.pkl')
model_pdp = joblib.load('models/model_pdp.pkl')
scaler = joblib.load('models/scaler_pdp.pkl')

with open('data/features.json', 'r') as f:
    feature_names = json.load(f)

state_df = pd.read_csv('data/state_features_2023.csv')

# Page config
st.set_page_config(page_title="Nigeria Election Predictor", layout="wide")

st.title("🗳️ Nigeria Presidential Election Predictor")
st.caption("Predicting 2023 state-level vote share using ML — APC, PDP, and LP")
st.divider()

# Sidebar
with st.sidebar:
    st.header("State & Scenario Inputs")

    selected_state = st.selectbox("Select State", sorted(state_df['State'].unique()))

    st.subheader("Adjust Scenario")
    turnout = st.slider("Voter Turnout (%)", 10.0, 60.0, 26.93)
    gov_party = st.selectbox("Governor's Party", ["APC", "PDP", "Other"])
    incumbent = st.selectbox("Federal Incumbent Party", ["APC", "PDP"])

# Get the selected state's actual features
row = state_df[state_df['State'] == selected_state].iloc[0]

# Encode scenario inputs
gov_encoded = 0 if gov_party == "APC" else 1 if gov_party == "PDP" else 2
incumbent_encoded = 1 if incumbent == "APC" else 0
gov_aligns = 1 if gov_party == incumbent else 0

# Build input row — historical features from CSV, scenario inputs overridden
input_dict = row[feature_names].to_dict()
input_dict['Turnout_%'] = turnout
input_dict['Gov_Party_Encoded'] = gov_encoded
input_dict['Incumbent_Encoded'] = incumbent_encoded
input_dict['Gov_Aligns_Incumbent'] = gov_aligns

input_data = pd.DataFrame([input_dict])[feature_names]

# Predictions
apc_pred = float(model_apc.predict(input_data)[0])
pdp_pred = float(model_pdp.predict(scaler.transform(input_data))[0])
lp_pred = max(0.0, 100.0 - apc_pred - pdp_pred)

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

st.caption("⚠️ LP% is derived from remaining vote share after APC and PDP predictions. "
           "Model trained on 2011–2019 elections, validated against 2023.")
