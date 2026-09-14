import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
import plotly.express as px
import json

# ── Config ───────────────────────────────────────────────────────────────────
API_URL     = "https://nigeria-election-predictor-api.onrender.com/predict_2027"
GEOJSON_URL = "https://raw.githubusercontent.com/wmgeolab/geoBoundaries/main/releaseData/gbOpen/NGA/ADM1/geoBoundaries-NGA-ADM1_simplified.geojson"

CANDIDATE_COLORS = {
    'APC': '#004C97',
    'ADC': '#CC0000',
    'NDC': '#008000',
    'APM': '#FF8C00',
}

st.set_page_config(page_title="Nigeria 2027 Election Predictor", layout="wide")

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv('data/nigeria_2027_features.csv')
    return df

@st.cache_data
def load_geojson():
    resp = requests.get(GEOJSON_URL, timeout=30)
    resp.raise_for_status()
    return resp.json()

state_df = load_data()
geojson  = load_geojson()

# ── API call ──────────────────────────────────────────────────────────────────
def call_predict_2027(row_dict, turnout):
    payload = {
        "State":                         str(row_dict['State']),
        "Zone_Encoded":                  float(row_dict['Zone_Encoded']),
        "Turnout_pct":                   float(turnout),
        "Gov_Encoded":                   float(row_dict['Gov_2026_Encoded']),
        "Incumbent_Encoded":             float(row_dict['Incumbent_Encoded']),
        "Gov_Aligns_Incumbent":          float(row_dict['Gov_Aligns_APC']),
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
    response = requests.post(API_URL, json=payload, timeout=30)
    response.raise_for_status()
    return response.json()

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🗳️ Nigeria 2027 Presidential Election Predictor")
st.caption("State-level vote share predictions — APC · ADC · NDC · APM")
st.divider()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Scenario Inputs")
    selected_state = st.selectbox(
        "Select State", sorted(state_df['State'].unique()))
    st.subheader("Adjust Scenario")
    turnout = st.slider(
        "Voter Turnout (%)", 10.0, 60.0,
        float(state_df[state_df['State'] == selected_state]['Turnout_%'].values[0])
    )

# ── Get predictions for all states ───────────────────────────────────────────
with st.spinner("Getting 2027 predictions from API..."):
    results = []
    errors  = []

    for _, row in state_df.iterrows():
        try:
            pred = call_predict_2027(row.to_dict(), turnout)
            results.append({
                'State':   row['State'],
                'Zone':    row['Geopolitical_Zone'],
                'APC_%':   pred['apc_pct'],
                'ADC_%':   pred['adc_pct'],
                'NDC_%':   pred['ndc_pct'],
                'APM_%':   pred['apm_pct'],
                'Winner':  pred['winner'],
            })
        except Exception as e:
            errors.append(row['State'])
            results.append({
                'State':   row['State'],
                'Zone':    row['Geopolitical_Zone'],
                'APC_%':   np.nan,
                'ADC_%':   np.nan,
                'NDC_%':   np.nan,
                'APM_%':   np.nan,
                'Winner':  'Unknown',
            })

    pred_df = pd.DataFrame(results)

if errors:
    st.warning(f"API call failed for: {', '.join(errors)}")

# ── Selected state ────────────────────────────────────────────────────────────
sel = pred_df[pred_df['State'] == selected_state].iloc[0]

st.header(f"📍 {selected_state}")

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
fig_state.update_layout(
    title=f"{selected_state} — 2027 Predicted Vote Share",
    yaxis=dict(title="Vote Share (%)", range=[0, 100]),
    plot_bgcolor='white', height=400
)
st.plotly_chart(fig_state, use_container_width=True)

st.divider()

# ── National scoreboard ───────────────────────────────────────────────────────
st.header("🇳🇬 National Scoreboard")

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

try:
    fig_map = px.choropleth(
        pred_df,
        geojson=geojson,
        locations='State',
        featureidkey='properties.shapeName',
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
        title='Projected 2027 Winner by State'
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
st.subheader("All States — 2027 Predicted Results")

table = pred_df[['State', 'Zone', 'APC_%', 'ADC_%',
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
    "All predictions are probabilistic estimates, not guarantees. "
    "Turnout scenario adjustable via sidebar slider."
)
