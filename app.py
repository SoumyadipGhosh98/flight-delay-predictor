
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ===============================
# LOAD FILES
# ===============================
model = joblib.load("model/model.pkl")
model_columns = joblib.load("model/model_columns.pkl")
le_tail = joblib.load("model/le_tail.pkl")
route_freq = joblib.load("model/route_frequency.pkl")

airline_stats = joblib.load("model/airline_stats.pkl")
airport_stats = joblib.load("model/airport_stats.pkl")
aircraft_stats = joblib.load("model/aircraft_stats.pkl")

top_airlines = joblib.load("model/top_airlines.pkl")
top_airports = joblib.load("model/top_airports.pkl")

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(page_title="Flight AI Dashboard", layout="wide")

# ===============================
# PREMIUM CSS
# ===============================
st.markdown(''' 
<style>
.main {background-color: #0E1117;}
.big-title {font-size: 42px; font-weight: bold; color: #4CAF50;}
.subtitle {color: #AAAAAA; font-size: 18px;}
.card {
    padding: 20px;
    border-radius: 12px;
    background: #1c1f26;
    margin-bottom: 15px;
}
</style>
''', unsafe_allow_html=True)

# ===============================
# HEADER
# ===============================
st.markdown('<p class="big-title">✈️ Aviation Disruption Intelligence System</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI-powered prediction of delays & cancellations</p>', unsafe_allow_html=True)

st.divider()

# ===============================
# INPUT PANEL
# ===============================
st.subheader("📊 Flight Input Panel")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### ⏱ Schedule Info")
    DEP_HOUR = st.slider("Departure Hour", 0, 23, 10)
    DAY_OF_WEEK = st.selectbox("Day of Week", list(range(1,8)))
    is_weekend = st.toggle("Weekend")
    is_peak_hour = st.toggle("Peak Hour")

with col2:
    st.markdown("### 🌍 Route Info")
    ORIGIN = st.selectbox("Origin Airport", top_airports)
    DEST = st.selectbox("Destination Airport", [a for a in top_airports if a != ORIGIN])
    CARRIER = st.selectbox("Carrier", top_airlines)
    TAIL_NUM = st.text_input("Aircraft Tail Number", "N12345")

st.divider()

# ===============================
# PREDICT BUTTON
# ===============================
if st.button("🚀 Predict Flight Status"):

    # ===============================
    # REAL LOOKUPS
    # ===============================
    airline_data = airline_stats.get(CARRIER, {})
    airport_data = airport_stats.get(ORIGIN, {})
    aircraft_data = aircraft_stats.get(TAIL_NUM, {})

    airline_delay_rate = airline_data.get('airline_delay_rate_7f', 0.2)
    airline_cancel_rate = airline_data.get('airline_cancel_rate_7f', 0.05)

    airport_delay_rate = airport_data.get('airport_delay_rate_7f', 0.2)
    airport_daily_flights = airport_data.get('airport_daily_flights', 200)

    aircraft_delay_rate = aircraft_data.get('aircraft_delay_rate_7f', 0.2)
    aircraft_cancel_rate = aircraft_data.get('aircraft_cancel_rate_7f', 0.05)

    rotation_gap = 2

    # ===============================
    # INPUT DF
    # ===============================
    df = pd.DataFrame([{
        'DEP_HOUR': DEP_HOUR,
        'DAY_OF_WEEK': DAY_OF_WEEK,
        'is_weekend': int(is_weekend),
        'is_peak_hour': int(is_peak_hour),
        'DISTANCE': 500,
        'CRS_ARR_TIME': 1400,
        'airline_delay_rate_7f': airline_delay_rate,
        'airline_cancel_rate_7f': airline_cancel_rate,
        'airport_delay_rate_7f': airport_delay_rate,
        'airport_daily_flights': airport_daily_flights,
        'aircraft_delay_rate_7f': aircraft_delay_rate,
        'aircraft_cancel_rate_7f': aircraft_cancel_rate,
        'aircraft_rotation_gap_hours': rotation_gap
    }])

    # Route feature
    df['route'] = ORIGIN + "_" + DEST
    df['route_freq'] = df['route'].map(route_freq).fillna(0)
    df.drop(columns=['route'], inplace=True)

    # Tail encoding
    try:
        df['TAIL_NUM'] = le_tail.transform([TAIL_NUM])[0]
    except:
        df['TAIL_NUM'] = -1

    # Encoding
    df = pd.get_dummies(df)
    df = df.reindex(columns=model_columns, fill_value=0)

    # ===============================
    # PREDICT
    # ===============================
    probs = model.predict_proba(df)
    pred = np.argmax(probs)

    labels = {
        0: "🟢 On-Time",
        1: "🟡 Delayed",
        2: "🔴 Cancelled"
    }

    # ===============================
    # OUTPUT DASHBOARD
    # ===============================
    st.subheader("📈 Prediction Dashboard")

    colA, colB, colC = st.columns(3)

    colA.metric("On-Time", f"{round(probs[0][0]*100,2)}%")
    colB.metric("Delay", f"{round(probs[0][1]*100,2)}%")
    colC.metric("Cancel", f"{round(probs[0][2]*100,2)}%")

    st.progress(float(np.max(probs)))

    st.markdown(f"### ✈️ Final Prediction: *{labels[pred]}*")

    if pred == 1:
        st.warning("⚠️ Delay risk detected")
    elif pred == 2:
        st.error("🚨 Cancellation risk detected")
    else:
        st.success("✅ Flight likely on time")
