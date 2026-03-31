# ✈️ Aviation Disruption Intelligence System  
### Multi-Class Flight Delay & Cancellation Prediction (Production-Ready ML System)

---

## 🚀 Project Overview

This project builds a **production-grade, end-to-end machine learning system** to predict flight outcomes **before departure**.

The model classifies flights into:
- 🟢 On-Time  
- 🟡 Delayed (>15 mins)  
- 🔴 Cancelled  

Built on real-world aviation data, the system transforms raw operational data into **decision intelligence for airline operations**.

## 🎯 Business Problem
Flight disruptions create cascading operational and financial challenges:

- Aircraft rotation delays  
- Crew duty violations  
- Airport congestion  
- Passenger missed connections  
- Compensation & rebooking costs  
- Fuel inefficiencies  
- Brand reputation damage  

### 💡 Objective
To build a predictive system that:
- Identifies disruption risk BEFORE departure  
- Enables proactive planning  
- Reduces operational uncertainty  
- Improves airline efficiency

## 📊 Dataset
- Source: U.S. Bureau of Transportation Statistics (BTS)  
- Timeframe: July 2024 – November 2025  
- Filtered to:
  - Top 10 Airlines  
  - Top 10 Airports  
- Size: ~500,000+ flight records  

Includes:
- Flight schedules  
- Carrier data  
- Airport data  
- Operational outcomes

## 🧠 Feature Engineering

### ✈️ Airline Level
- airline_delay_rate_7f  
- airline_cancel_rate_7f  

### 🏢 Airport Level
- airport_delay_rate_7f  
- airport_daily_flights  

### 🔁 Route Level
- route_freq (frequency encoding)

### 🛫 Aircraft Level
- aircraft_delay_rate_7f  
- aircraft_cancel_rate_7f  
- aircraft_rotation_gap_hours  

### ⏱ Time-Based
- DEP_HOUR  
- is_peak_hour (data-driven)  
- is_weekend  

👉 Captures:
- Congestion  
- Delay propagation  
- Operational reliability

## ⚙️ Data Processing Pipeline
- Data merging (multi-month)
- Missing value treatment (domain-based)
- Duplicate validation
- Outlier analysis (retained)
- Leakage removal

### Encoding:
- One-hot → Airlines & Airports  
- Label → Tail Number  
- Frequency → Route  

### Validation:
- Time-based train-test split  

### Multicollinearity:
- VIF-based feature removal

## 🎯 Target Variable

| Class | Meaning |
|------|--------|
| 0 | On-Time |
| 1 | Delayed |
| 2 | Cancelled |

## 🤖 Models Implemented
- Logistic Regression (Baseline + Tuned)
- Decision Tree
- Random Forest
- Gradient Boosting
- XGBoost (Final Model)

## 📈 Evaluation Metrics
- Accuracy  
- Precision  
- Recall  
- F1 Score (Macro) ⭐  
- ROC-AUC  
- Cohen Kappa  

### Why F1 Macro?
- Treats all classes equally  
- Important for delay & cancellation detection  
- Avoids bias toward on-time flights

## 🏆 Model Performance
- ROC-AUC: ~0.80  
- Strong → On-Time  
- Moderate → Delay  
- Weak → Cancellation  

👉 Reflects real-world aviation uncertainty

## ⚠️ Challenges
- Severe class imbalance  
- Missing weather & ATC data  
- Complex non-linear delay propagation

## 🧠 Business Insights
- Airport congestion drives delays  
- Airline reliability matters  
- Aircraft history improves prediction  
- Peak hours increase disruption risk

## 🖥️ Streamlit Application
- Real-time prediction  
- Automated feature generation  
- Interactive UI  
- Business interpretation output

## 📂 Project Structure
flight_delay_app/
│
├── app.py
├── requirements.txt
├── README.md
│
├── model/
│   ├── model.pkl
│   ├── model_columns.pkl
│   ├── le_tail.pkl
│   ├── route_frequency.pkl
│   ├── airline_stats.pkl
│   ├── airport_stats.pkl
│   ├── aircraft_stats.pkl
│
├── notebooks/
│   └── Flight_Delay_Modeling.ipynb

## 🚀 Run Locally
pip install -r requirements.txt  
streamlit run app.py

## 🌍 Deployment
Deployed using Streamlit Cloud:
1. Upload to GitHub  
2. Connect repo  
3. Deploy app.py

## 🔍 Key Technical Decisions
- Leakage-free modeling  
- Time-based validation  
- No scaling for tree models  
- Frequency encoding for route  
- Rolling features for temporal patterns  
- Threshold tuning for recall

## 🔮 Future Improvements
- Weather data integration  
- ATC signals  
- LSTM models  
- Cost-sensitive learning

## 👨‍💻 Author
Soumyadip Ghosh  
MBA (Analytics) | Data Science

## ⭐ Final Note
This is not just a machine learning model —  
it is a real-world aviation intelligence system designed for operational decision-making.