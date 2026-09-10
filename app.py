import streamlit as st
import pandas as pd
import numpy as np
import json
import requests
import plotly.graph_objects as go

# ── Config ───────────────────────────────────────────────────────────────────
API_URL = "https://nigeria-election-predictor-api.onrender.com/predict"

st.set_page_config(page_title="Nigeria Election Predictor", layout="wide")

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    features = pd.read_csv('data/state_features_2023.csv')
    master   = pd.read_csv('data/election_predictor_master_v2.csv')
    return features, master

state_df, master = load_data()

# Clean actual results
for col in ['Tinubu_%', 'Atiku_%', 'Obi_%', 'Others_%']:
    master[col] = pd.to_numeric(master[col].astype(str).str.replace('%', ''), errors='coerce')

actual_2023 = master[master['Year'] == 2023][
    ['State', 'Tinubu_%', 'Atiku_%', 'Obi_%', 'Others_%', 'Total_Valid_Votes']
].copy()
actual_2023.columns = ['State', 'APC_%', 'PDP_%', 'LP_%', 'Others_%', 'Votes']

GOV_DECODE = {0: 'APC', 1: 'PDP', 2: 'Other'}
GOV_ENCODE = {'APC': 0, 'PDP': 1, 'Other': 2}

# ── API call ──────────────────────────────────────────────────────────────────
def call_predict_api(row_dict):
    payload = {
        "Zone_Encoded":               float(row_dict['Zone_Encoded']),
        "Turnout_pct":                float(row_dict['Turnout_%']),
        "Gov_Party_Encoded":          float(row_dict['Gov_Party_Encoded']),
        "Incumbent_Encoded":          float(row_dict['Incumbent_Encoded']),
        "Gov_Aligns_Incumbent":       float(row_dict['Gov_Aligns_Incumbent']),
        "National_Security_Incidents":float(row_dict['National_Security_Incidents']),
        "Prev_APC_pct":               float(row_dict['Prev_APC_%']),
        "Prev_PDP_pct":               float(row_dict['Prev_PDP_%']),
        "Prev_Third_Party_pct":       float(row_dict['Prev_Third_Party_%']),
        "Third_Party_pct":            float(row_dict['Third_Party_%']),
        "State_Avg_APC_pct":          float(row_dict['State_Avg_APC_%']),
        "State_Avg_PDP_pct":          float(row_dict['State_Avg_PDP_%']),
        "Zone_Year_APC_pct":          float(row_dict['Zone_Year_APC_%']),
        "Zone_Year_PDP_pct":          float(row_dict['Zone_Year_PDP_%']),
    }
    response = requests.post(API_URL, json=payload, timeout=30)
    response.raise_for_status()
    return response.json()

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🗳️ Nigeria Presidential Election Predictor")
st.caption("2023 state-level vote share predictions — trained on 2011–2019, validated against 2023")
st.divider()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("State & Scenario Inputs")

    selected_state = st.selectbox("Select State", sorted(state_df['State'].unique()))
    row = state_df[state_df['State'] == selected_state].iloc[0]

    st.subheader("Adjust Scenario")
    turnout   = st.slider("Voter Turnout (%)", 10.0, 60.0, float(row['Turnout_%']))
    party_options = ["APC", "PDP", "Other"]
    gov_party = st.selectbox(
        "Governor's Party",
        party_options,
        index=party_options.index(GOV_DECODE[int(row['Gov_Party_Encoded'])])
    )
    incumbent   = st.selectbox("Federal Incumbent Party", ["APC", "PDP"], index=0)
    show_actual = st.checkbox("Show actual 2023 results", value=True)

# ── Build scenario for all states ────────────────────────────────────────────
gov_encoded       = GOV_ENCODE[gov_party]
incumbent_encoded = 1 if incumbent == 'APC' else 0

scen = state_df.copy()
scen['Turnout_%']          = turnout
scen['Gov_Party_Encoded']  = gov_encoded
scen['Incumbent_Encoded']  = incumbent_encoded
scen['Gov_Aligns_Incumbent'] = (scen['Gov_Party_Encoded'] == incumbent_encoded).astype(int)

# ── Call API for every state ──────────────────────────────────────────────────
with st.spinner("Getting predictions from API..."):
    pred_apc, pred_pdp, pred_lp = [], [], []
    errors = []

    for _, state_row in scen.iterrows():
        try:
            result = call_predict_api(state_row.to_dict())
            pred_apc.append(result['apc_pct'])
            pred_pdp.append(result['pdp_pct'])
            pred_lp.append(result['lp_pct'])
        except Exception as e:
            errors.append(state_row['State'])
            pred_apc.append(np.nan)
            pred_pdp.append(np.nan)
            pred_lp.append(np.nan)

    scen['Pred_APC_%'] = pred_apc
    scen['Pred_PDP_%'] = pred_pdp
    scen['Pred_LP_%']  = pred_lp

if errors:
    st.warning(f"API call failed for: {', '.join(errors)}")

scen = scen.merge(actual_2023[['State', 'Votes']], on='State', how='left')

# ── Selected state ────────────────────────────────────────────────────────────
sel = scen[scen['State'] == selected_state].iloc[0]
act = actual_2023[actual_2023['State'] == selected_state].iloc[0]

st.header(f"📍 {selected_state}")

c1, c2, c3 = st.columns(3)
c1.metric("APC — Predicted", f"{sel['Pred_APC_%']:.1f}%",
          f"Actual: {act['APC_%']:.1f}%" if show_actual else None)
c2.metric("PDP — Predicted", f"{sel['Pred_PDP_%']:.1f}%",
          f"Actual: {act['PDP_%']:.1f}%" if show_actual else None)
c3.metric("LP — Derived",    f"{sel['Pred_LP_%']:.1f}%",
          f"Actual: {act['LP_%']:.1f}%" if show_actual else None)

fig = go.Figure()
if show_actual:
    fig.add_trace(go.Bar(
        x=['APC', 'PDP', 'LP'], y=[act['APC_%'], act['PDP_%'], act['LP_%']],
        name='Actual 2023', marker_color='lightgray'
    ))
fig.add_trace(go.Bar(
    x=['APC', 'PDP', 'LP'],
    y=[sel['Pred_APC_%'], sel['Pred_PDP_%'], sel['Pred_LP_%']],
    name='Predicted', marker_color=['#004C97', '#CC0000', '#008000'],
    text=[f"{sel['Pred_APC_%']:.1f}%", f"{sel['Pred_PDP_%']:.1f}%", f"{sel['Pred_LP_%']:.1f}%"],
    textposition='outside'
))
fig.update_layout(
    title=f"{selected_state} — Predicted vs Actual",
    yaxis=dict(title="Vote Share (%)", range=[0, 100]),
    barmode='group', plot_bgcolor='white', height=400
)
st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── National result ───────────────────────────────────────────────────────────
st.header("🇳🇬 Nationwide Result (under current scenario)")

w = scen['Votes']
nat_pred = {
    'APC': float((scen['Pred_APC_%'] * w).sum() / w.sum()),
    'PDP': float((scen['Pred_PDP_%'] * w).sum() / w.sum()),
    'LP':  float((scen['Pred_LP_%']  * w).sum() / w.sum()),
}
nat_actual = {
    'APC': float((actual_2023['APC_%'] * actual_2023['Votes']).sum() / actual_2023['Votes'].sum()),
    'PDP': float((actual_2023['PDP_%'] * actual_2023['Votes']).sum() / actual_2023['Votes'].sum()),
    'LP':  float((actual_2023['LP_%']  * actual_2023['Votes']).sum() / actual_2023['Votes'].sum()),
}

winner = max(nat_pred, key=nat_pred.get)
n1, n2, n3 = st.columns(3)
n1.metric("Predicted Winner 🏆", winner, f"{nat_pred[winner]:.1f}% nationally")
n2.metric("APC national", f"{nat_pred['APC']:.1f}%", f"Actual: {nat_actual['APC']:.1f}%" if show_actual else None)
n3.metric("PDP national", f"{nat_pred['PDP']:.1f}%", f"Actual: {nat_actual['PDP']:.1f}%" if show_actual else None)

fig2 = go.Figure()
if show_actual:
    fig2.add_trace(go.Bar(
        x=list(nat_actual.keys()), y=list(nat_actual.values()),
        name='Actual 2023', marker_color='lightgray'
    ))
fig2.add_trace(go.Bar(
    x=list(nat_pred.keys()), y=list(nat_pred.values()),
    name='Predicted', marker_color=['#004C97', '#CC0000', '#008000'],
    text=[f"{v:.1f}%" for v in nat_pred.values()], textposition='outside'
))
fig2.update_layout(
    title="National Vote Share — Predicted vs Actual (weighted by votes cast)",
    yaxis=dict(title="Vote Share (%)", range=[0, 60]),
    barmode='group', plot_bgcolor='white', height=400
)
st.plotly_chart(fig2, use_container_width=True)

# ── State table ───────────────────────────────────────────────────────────────
st.subheader("All States — Predicted vs Actual")

table = scen[['State', 'Pred_APC_%', 'Pred_PDP_%', 'Pred_LP_%']].copy()
table[['Pred_APC_%', 'Pred_PDP_%', 'Pred_LP_%']] = table[['Pred_APC_%', 'Pred_PDP_%', 'Pred_LP_%']].round(1)
table.columns = ['State', 'APC %', 'PDP %', 'LP %']

if show_actual:
    table = table.merge(
        actual_2023[['State', 'APC_%', 'PDP_%', 'LP_%']].round(1),
        on='State', suffixes=(' (Pred)', ' (Actual)')
    )

st.dataframe(table.sort_values('State').reset_index(drop=True), use_container_width=True, height=400)

st.caption("⚠️ LP% is derived from remaining vote share — the model was trained on two-party elections (2011–2019) "
           "and cannot independently predict LP's 2023 surge. Expect largest errors in the South-East (Obi) "
           "and North-West (Kwankwaso). National figures weighted by votes cast per state.")
