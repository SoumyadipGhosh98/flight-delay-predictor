
# ===============================
# IMPORTS
# ===============================
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from google import genai
from dotenv import load_dotenv

# ===============================
# LOAD ENV (LOCAL)
# ===============================
load_dotenv()

# ===============================
# SAFE API KEY LOADER
# ===============================
def get_api_key():
    return st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")


# ===============================
# AI COPILOT FUNCTION
# ===============================
def generate_ai_insight(input_summary, probs, pred):

    labels = {0: "On-Time", 1: "Delayed", 2: "Cancelled"}

    prompt = f'''
    You are an aviation operations assistant.

    Prediction:
    {labels[pred]}

    Probabilities:
    On-Time: {round(probs[0][0]*100,1)}%
    Delay: {round(probs[0][1]*100,1)}%
    Cancel: {round(probs[0][2]*100,1)}%

    Context:
    {input_summary}

    Explain:
    - Why this happened
    - Key drivers
    - Give recommendation

    Keep it short and practical.
    '''

    client = genai.Client(api_key=get_api_key())

    response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
    return response.text

# ===============================
# LOAD MODEL FILES
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
# PREMIUM OFF-WHITE UI
# ===============================
st.markdown('''
<style>

/* ===== BACKGROUND ===== */
.stApp {
    background: linear-gradient(135deg, #f8fafc, #eef2f7);
    font-family: 'Segoe UI', sans-serif;
}

/* ===== HEADER ===== */
.title {
    font-size: 42px;
    font-weight: 700;
    color: #111827;
}

.subtitle {
    color: #6b7280;
    font-size: 17px;
    margin-bottom: 10px;
}

/* ===== CARD ===== */
.glass {
    background: white;
    border-radius: 16px;
    padding: 22px;
    box-shadow: 0 8px 30px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}

/* ===== BUTTON ===== */
.stButton>button {
    width: 100%;
    height: 52px;
    border-radius: 12px;
    background: linear-gradient(135deg, #2563eb, #3b82f6);
    color: white;
    font-size: 16px;
    font-weight: 600;
    border: none;
    transition: 0.3s;
}

.stButton>button:hover {
    transform: scale(1.02);
}

/* ===== METRIC ===== */
.metric-card {
    background: #f1f5f9;
    padding: 18px;
    border-radius: 12px;
    text-align: center;
}

/* ===== AI BOX ===== */
.ai-box {
    background: #f9fafb;
    padding: 15px;
    border-radius: 10px;
    border-left: 4px solid #3b82f6;
    line-height: 1.6;
}

</style>
''', unsafe_allow_html=True)

# ===============================
# HEADER
# ===============================
st.markdown('<div class="title">✈️ Aviation Disruption Intelligence</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-powered delay & cancellation prediction system</div>', unsafe_allow_html=True)
st.markdown("---")

# ===============================
# INPUT PANEL
# ===============================
st.markdown('<div class="glass">', unsafe_allow_html=True)

st.subheader("📊 Flight Input Panel")

col1, col2 = st.columns(2)

with col1:
    DEP_HOUR = st.slider("Departure Hour", 0, 23, 10)
    DAY_OF_WEEK = st.selectbox("Day of Week", list(range(1,8)))
    is_weekend = st.toggle("Weekend")
    is_peak_hour = st.toggle("Peak Hour")

with col2:
    ORIGIN = st.selectbox("Origin Airport", top_airports)
    DEST = st.selectbox("Destination Airport", [a for a in top_airports if a != ORIGIN])
    CARRIER = st.selectbox("Carrier", top_airlines)
    TAIL_NUM = st.text_input("Aircraft Tail Number", "N12345")

st.markdown('</div>', unsafe_allow_html=True)

# ===============================
# PREDICT BUTTON
# ===============================
if st.button("🚀 Predict Flight Status"):

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

    df['route'] = ORIGIN + "_" + DEST
    df['route_freq'] = df['route'].map(route_freq).fillna(0)
    df.drop(columns=['route'], inplace=True)

    try:
        df['TAIL_NUM'] = le_tail.transform([TAIL_NUM])[0]
    except:
        df['TAIL_NUM'] = -1

    df = pd.get_dummies(df)
    df = df.reindex(columns=model_columns, fill_value=0)

    probs = model.predict_proba(df)
    pred = np.argmax(probs)

    labels = {
        0: "🟢 On-Time",
        1: "🟡 Delayed",
        2: "🔴 Cancelled"
    }

    # OUTPUT
    st.markdown('<div class="glass">', unsafe_allow_html=True)

    st.subheader("📈 Prediction Dashboard")

    colA, colB, colC = st.columns(3)

    with colA:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("On-Time", f"{round(probs[0][0]*100,2)}%")
        st.markdown('</div>', unsafe_allow_html=True)

    with colB:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Delay", f"{round(probs[0][1]*100,2)}%")
        st.markdown('</div>', unsafe_allow_html=True)

    with colC:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Cancel", f"{round(probs[0][2]*100,2)}%")
        st.markdown('</div>', unsafe_allow_html=True)

    st.progress(float(np.max(probs)))
    st.markdown(f"### ✈️ Final Prediction: *{labels[pred]}*")

    if pred == 1:
        st.warning("⚠️ Delay risk detected")
    elif pred == 2:
        st.error("🚨 Cancellation risk detected")
    else:
        st.success("✅ Flight likely on time")

    st.markdown('</div>', unsafe_allow_html=True)

    # ===============================
    # DECISION INTELLIGENCE LAYER
    # ===============================

    st.markdown("### 🧠 Operational Decision Support")

    delay_prob = probs[0][1]
    cancel_prob = probs[0][2]

    # Risk Level Classification
    if cancel_prob > 0.4:
        risk_level = "🔴 High Risk (Cancellation Likely)"
    elif delay_prob > 0.4:
        risk_level = "🟠 Moderate Risk (Delay Likely)"
    else:
        risk_level = "🟢 Low Risk (On-Time Expected)"

    st.markdown(f"*Risk Assessment:* {risk_level}")

    # Actionable Recommendations
    if pred == 2:
        st.error('''
    🚨 *Recommended Actions:*
    - Initiate *pre-emptive rebooking*
    - Reassign aircraft if available
    - Notify passengers early
    - Adjust crew schedules to avoid violations
    ''')

    elif pred == 1:
        st.warning('''
    ⚠️ *Recommended Actions:*
    - Add *buffer time (30–60 mins)*
    - Monitor airport congestion
    - Prepare alternate gate allocation
    - Alert ground operations team
    ''')

    else:
        st.success('''
    ✅ *Recommended Actions:*
    - Proceed as scheduled
    - Maintain standard turnaround operations
    - No intervention required
    ''')

    # Business Impact Layer
    st.markdown("### 💰 Business Impact Insight")

    if pred == 2:
        st.markdown('''
    - High risk of *rebooking cost & compensation*
    - Potential *network disruption propagation*
    - Impact on *customer satisfaction*
    ''')

    elif pred == 1:
        st.markdown('''
    - Moderate *fuel inefficiency risk*
    - Possible *crew overtime cost*
    - Minor impact on *OTP metrics*
    ''')

    else:
        st.markdown('''
    - Optimal *resource utilization*
    - No additional operational cost
    - Positive impact on *on-time performance*
    ''')

    # ===============================
    # AI COPILOT
    # ===============================
    input_summary = {
        "origin": ORIGIN,
        "destination": DEST,
        "carrier": CARRIER,
        "departure_hour": DEP_HOUR
    }

    with st.spinner("🧠 Generating AI Insight..."):
        try:
            insight = generate_ai_insight(input_summary, probs, pred)
        except Exception as e:
            insight = f"Error: {e}"

    st.markdown('<div class="glass">', unsafe_allow_html=True)

    st.subheader("🧠 AI Copilot Insight")

    st.markdown(f'''
    <div class="ai-box">
    {insight}
    </div>
    ''', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
