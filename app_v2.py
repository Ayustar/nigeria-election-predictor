import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
import plotly.express as px
import json

# ── Config ───────────────────────────────────────────────────────────────────
API_BASE_URL = "https://nigeria-election-predictor-api.onrender.com"
API_URL      = f"{API_BASE_URL}/predict_2027"

CANDIDATE_COLORS = {
    'APC': '#004C97',
    'ADC': '#CC0000',
    'NDC': '#008000',
    'APM': '#FF8C00',
}

GOV_ENCODE = {'APC': 0, 'PDP': 1, 'Other': 2}
GOV_DECODE = {0: 'APC', 1: 'PDP', 2: 'Other'}

INCUMBENT_ENCODE = {'APC': 1, 'ADC': 0, 'NDC': 0, 'APM': 0}

st.set_page_config(page_title="Nigeria 2027 Election Predictor", layout="wide")

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    features    = pd.read_csv('data/nigeria_2027_features.csv')
    predictions = pd.read_csv('data/predictions_2027.csv')
    return features, predictions

@st.cache_data
def load_geojson():
    with open('data/nigeria.geojson', 'r') as f:
        return json.load(f)

state_df, pred_df = load_data()
geojson           = load_geojson()

# ── API call — single state only ─────────────────────────────────────────────
def call_predict_2027(row_dict, turnout, gov_encoded,
                      incumbent_encoded, gov_aligns_incumbent):
    payload = {
        "State":                         str(row_dict['State']),
        "Zone_Encoded":                  float(row_dict['Zone_Encoded']),
        "Turnout_pct":                   float(turnout),
        "Gov_Encoded":                   float(gov_encoded),
        "Incumbent_Encoded":             float(incumbent_encoded),
        "Gov_Aligns_Incumbent":          float(gov_aligns_incumbent),
        "National_Security_Incidents":   float(row_dict['National_Security_Incidents']),
        "Prev_APC_pct":                  float(row_dict['Prev_APC_%']),
        "Prev_ADC_pct":                  float(row_dict['Prev_ADC_%']),
        "Prev_NDC_pct":                  float(row_dict['NDC_Base_%']),
        "State_Avg_APC":                 float(row_dict['State_Avg_APC']),
        "State_Avg_ADC":                 float(row_dict['State_Avg_ADC']),
        "State_Avg_NDC":                 float(row_dict['State_Avg_NDC']),
        "Zone_Avg_APC":                  float(row_dict['Zone_Avg_APC']),
        "Zone_Avg_ADC":                  float(row_dict['Zone_Avg_ADC']),
        "Zone_Avg_NDC":                  float(row_dict['Zone_Avg_NDC']),
        "Gov_Is_Presidential_Candidate": float(row_dict['Gov_Is_Presidential_Candidate']),
        "Federal_Admin_Control":         float(row_dict['Federal_Admin_Control']),
        "Geopolitical_Zone":             str(row_dict['Geopolitical_Zone']),
    }
    response = requests.post(API_URL, json=payload, timeout=60)
    response.raise_for_status()
    return response.json()

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🗳️ Nigeria 2027 Presidential Election Predictor")
st.caption("State-level vote share predictions — APC · ADC · NDC · APM")
st.divider()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("State & Scenario Inputs")

    selected_state = st.selectbox(
        "Select State", sorted(state_df['State'].unique()))
    state_row = state_df[state_df['State'] == selected_state].iloc[0]

    st.subheader("Adjust Scenario")

    # Turnout
    baseline_turnout = float(state_row['Turnout_%'])
    turnout = st.slider(
        "Voter Turnout (%)", 10.0, 60.0, baseline_turnout)

    # Governor's party
    current_gov = GOV_DECODE.get(int(state_row['Gov_2026_Encoded']), 'Other')
    gov_options = ['APC', 'PDP', 'Other']
    gov_party   = st.selectbox(
        "Governor's Party",
        gov_options,
        index=gov_options.index(current_gov) if current_gov in gov_options else 2
    )
    gov_encoded = GOV_ENCODE[gov_party]

    # Federal incumbent
    incumbent_options = ['APC', 'ADC', 'NDC', 'APM']
    incumbent = st.selectbox(
        "Federal Incumbent Party",
        incumbent_options,
        index=0  # APC default
    )
    incumbent_encoded      = INCUMBENT_ENCODE[incumbent]
    gov_aligns_incumbent   = 1 if (
        (gov_party == 'APC' and incumbent == 'APC') or
        (gov_party == 'PDP' and incumbent == 'ADC')
    ) else 0

# ── Check if scenario changed from baseline ───────────────────────────────────
scenario_changed = (
    abs(turnout - baseline_turnout) > 0.01 or
    gov_encoded != int(state_row['Gov_2026_Encoded']) or
    incumbent != 'APC'
)

# ── Selected state section ────────────────────────────────────────────────────
st.header(f"📍 {selected_state}")

if scenario_changed:
    with st.spinner(f"Updating prediction for {selected_state}..."):
        try:
            pred = call_predict_2027(
                state_row.to_dict(), turnout,
                gov_encoded, incumbent_encoded, gov_aligns_incumbent
            )
            sel = {
                'APC_%':  pred['apc_pct'],
                'ADC_%':  pred['adc_pct'],
                'NDC_%':  pred['ndc_pct'],
                'APM_%':  pred['apm_pct'],
                'Winner': pred['winner'],
            }
        except Exception as e:
            st.warning(f"API call failed: {e}")
            sel = pred_df[pred_df['State'] == selected_state].iloc[0]
else:
    sel = pred_df[pred_df['State'] == selected_state].iloc[0]

c1, c2, c3, c4 = st.columns(4)
c1.metric("APC (Tinubu)",         f"{sel['APC_%']:.1f}%")
c2.metric("ADC (Atiku)",          f"{sel['ADC_%']:.1f}%")
c3.metric("NDC (Obi-Kwankwaso)", f"{sel['NDC_%']:.1f}%")
c4.metric("APM (Makinde)",        f"{sel['APM_%']:.1f}%")

# State bar chart
fig_state = go.Figure()
parties = ['APC', 'ADC', 'NDC', 'APM']
values  = [sel['APC_%'], sel['ADC_%'], sel['NDC_%'], sel['APM_%']]
colors  = [CANDIDATE_COLORS[p] for p in parties]

fig_state.add_trace(go.Bar(
    x=parties, y=values,
    marker_color=colors,
    text=[f"{v:.1f}%" for v in values],
    textposition='outside'
))

title_suffix = f" (Turnout: {turnout:.1f}%, Gov: {gov_party}, Incumbent: {incumbent})" \
    if scenario_changed else ""

fig_state.update_layout(
    title=f"{selected_state} — 2027 Predicted Vote Share{title_suffix}",
    yaxis=dict(title="Vote Share (%)", range=[0, 100]),
    plot_bgcolor='white', height=400
)
st.plotly_chart(fig_state, use_container_width=True)

st.divider()

# ── National scoreboard — always from precomputed CSV ─────────────────────────
st.header("🇳🇬 National Scoreboard")
st.caption("Based on baseline scenario — 2023 turnout, Sep 2026 governors, APC incumbent")

winner_counts = pred_df['Winner'].value_counts()
s1, s2, s3, s4 = st.columns(4)
s1.metric("APC States",  winner_counts.get('APC', 0), "Tinubu")
s2.metric("ADC States",  winner_counts.get('ADC', 0), "Atiku")
s3.metric("NDC States",  winner_counts.get('NDC', 0), "Obi-Kwankwaso")
s4.metric("APM States",  winner_counts.get('APM', 0), "Makinde")

national_winner = winner_counts.idxmax()
st.success(f"🏆 Projected National Winner: **{national_winner}** "
           f"({winner_counts[national_winner]} states)")

st.divider()

# ── Nigeria Map ───────────────────────────────────────────────────────────────
st.header("🗺️ State-by-State Winner Map")
st.caption("Baseline scenario")

try:
    fig_map = px.choropleth(
        pred_df,
        geojson=geojson,
        locations='State',
        featureidkey='properties.NAME_1',
        color='Winner',
        color_discrete_map=CANDIDATE_COLORS,
        hover_name='State',
        hover_data={
            'APC_%': ':.1f',
            'ADC_%': ':.1f',
            'NDC_%': ':.1f',
            'APM_%': ':.1f',
            'Winner': True
        },
        title='Projected 2027 Winner by State (Baseline)'
    )
    fig_map.update_geos(
        fitbounds="locations",
        visible=False
    )
    fig_map.update_layout(
        height=550,
        margin={"r": 0, "t": 40, "l": 0, "b": 0},
        legend_title="Party"
    )
    st.plotly_chart(fig_map, use_container_width=True)
except Exception as e:
    st.warning(f"Map could not render: {e}")

st.divider()

# ── National vote share chart ─────────────────────────────────────────────────
st.header("📊 National Vote Share (Simple Average)")
st.caption("Baseline scenario")

nat_avg = {
    'APC': pred_df['APC_%'].mean(),
    'ADC': pred_df['ADC_%'].mean(),
    'NDC': pred_df['NDC_%'].mean(),
    'APM': pred_df['APM_%'].mean(),
}

fig_nat = go.Figure()
fig_nat.add_trace(go.Bar(
    x=list(nat_avg.keys()),
    y=list(nat_avg.values()),
    marker_color=[CANDIDATE_COLORS[p] for p in nat_avg.keys()],
    text=[f"{v:.1f}%" for v in nat_avg.values()],
    textposition='outside'
))
fig_nat.update_layout(
    title="National Average Vote Share by Party",
    yaxis=dict(title="Vote Share (%)", range=[0, 60]),
    plot_bgcolor='white', height=400
)
st.plotly_chart(fig_nat, use_container_width=True)

# ── State-by-state table ──────────────────────────────────────────────────────
st.subheader("All States — 2027 Predicted Results (Baseline)")

table = pred_df[['State', 'Geopolitical_Zone', 'APC_%', 'ADC_%',
                  'NDC_%', 'APM_%', 'Winner']].copy()
table[['APC_%', 'ADC_%', 'NDC_%', 'APM_%']] = \
    table[['APC_%', 'ADC_%', 'NDC_%', 'APM_%']].round(1)
table.columns = ['State', 'Zone', 'APC %', 'ADC %',
                  'NDC %', 'APM %', 'Winner']

st.dataframe(
    table.sort_values('State').reset_index(drop=True),
    use_container_width=True, height=400
)

st.caption(
    "⚠️ APM predictions are rule-based (no historical presidential data for APM). "
    "NDC model trained on 2023 data only. "
    "Baseline predictions precomputed at each state's 2023 turnout, Sep 2026 governors, APC incumbent. "
    "Sidebar scenario inputs update the selected state's prediction via live API call. "
    "All predictions are probabilistic estimates, not guarantees."
)
