# ✈️ Aviation Disruption Intelligence System  
### Multi-Class Flight Delay & Cancellation Prediction with AI Copilot

---

## 🚀 Live Application  
👉 https://flight-delay-predictor-soumyadipghosh.streamlit.app/

---

## 🧠 Project Overview

This project develops a *production-ready Aviation Disruption Intelligence System* that predicts flight outcomes *before departure* and converts predictions into *actionable operational decisions*.

The system classifies flights into three categories:

- 🟢 *On-Time*
- 🟡 *Delayed (>15 minutes)*
- 🔴 *Cancelled*

Built using *500K+ real-world U.S. Bureau of Transportation Statistics (BTS) flight records*, the project integrates:

- Advanced feature engineering  
- Machine learning modeling  
- Decision intelligence layer  
- AI-powered Copilot (Gemini 2.5 Flash)  
- Interactive Streamlit deployment  

---

## 🎯 Business Problem

Flight disruptions create cascading operational and financial challenges:

### ✈️ Operational Impact
- Aircraft rotation disruptions  
- Crew duty time violations  
- Gate congestion  
- Missed passenger connections  

### 💰 Financial Impact
- Crew overtime costs  
- Passenger compensation & rebooking  
- Fuel inefficiencies  
- Brand erosion and customer churn  

### ⚠️ Current Gap
Airlines operate *reactively*, lacking forward-looking intelligence.

---

## 💡 Business Objective

To build a *predictive + decision-support system* that:

- Predicts delay/cancellation risk before departure  
- Identifies key drivers of disruption  
- Enables proactive operational decisions  
- Reduces cost and improves network efficiency  

---

## 📊 Dataset

### Source:
U.S. Bureau of Transportation Statistics (BTS)

### Time Period:
*July 2024 – November 2025*

### Scope:
- Top 10 airlines  
- Top 10 airports  

### Size:
*~500,000+ flight records*

---

## 🧠 Feature Engineering (CORE STRENGTH)

A major contribution of this project is *multi-level hierarchical feature engineering*:

---

### ✈️ Airline-Level Features
- airline_delay_rate_7f
- airline_cancel_rate_7f

👉 Captures *carrier reliability trends*

---

### 🏢 Airport-Level Features
- airport_delay_rate_7f
- airport_daily_flights

👉 Captures *congestion intensity and operational load*

---

### 🔁 Route-Level Features
- route_freq (frequency encoding)

👉 Captures *route demand and delay propagation patterns*

---

### 🛫 Aircraft-Level Features (Advanced)
- aircraft_delay_rate_7f
- aircraft_cancel_rate_7f
- aircraft_rotation_gap_hours

👉 Captures *fleet stability and rotation risk*

---

### ⏱ Time-Based Features
- DEP_HOUR
- DAY_OF_WEEK
- is_weekend
- is_peak_hour

👉 Captures *temporal demand cycles and congestion windows*

---

## ⚙️ Data Processing Pipeline

### 1️⃣ Data Cleaning
- Missing value handling (domain-based)
- Duplicate removal
- Data type corrections

---

### 2️⃣ Leakage Prevention
Strict exclusion of post-outcome variables such as:
- Actual delays  
- Arrival performance metrics  

👉 Ensures *real-world deployability*

---

### 3️⃣ Encoding Techniques
- One-hot encoding → categorical variables  
- Label encoding → TAIL_NUM  
- Frequency encoding → route  

---

### 4️⃣ Feature Scaling
- Applied only to *linear models (Logistic Regression)*  
- Tree-based models use raw features  

---

### 5️⃣ Multicollinearity Handling
- Variance Inflation Factor (VIF)
- Removed redundant variables:
  - CRS_DEP_TIME
  - CRS_ELAPSED_TIME

---

### 6️⃣ Train-Test Split
- Time-aware split to preserve *operational causality*

---

## 🎯 Target Variable

Multi-class classification:

| Class | Meaning |
|------|--------|
| 0 | On-Time |
| 1 | Delayed |
| 2 | Cancelled |

---

## 🤖 Machine Learning Models

### Baseline Model:
- Logistic Regression

### Advanced Models:
- Decision Tree  
- Random Forest  
- Gradient Boosting  
- *XGBoost (Best Model)*  

---

## 📈 Model Evaluation Metrics

- Accuracy  
- Precision  
- Recall  
- *F1 Score (Macro)* ⭐  
- ROC-AUC  
- Cohen Kappa (Reliability)

---

### 💡 Why F1 Macro?

- Treats all classes equally  
- Critical for imbalanced datasets  
- Ensures delays & cancellations are not ignored  

---

## 🏆 Model Performance Summary

- ROC-AUC: *~0.80*
- Strong performance:
  - On-time predictions  
- Moderate performance:
  - Delay detection  
- Challenging:
  - Cancellation prediction (rare events)

---

## ⚠️ Key Challenges

- Severe class imbalance  
- Lack of weather / ATC data  
- Complex non-linear delay propagation  
- Rare event prediction (cancellations)  

---

## 🧠 Decision Intelligence Layer (KEY DIFFERENTIATOR)

The system goes beyond prediction by providing:

### 🔍 Risk Classification
- Low Risk  
- Moderate Risk  
- High Risk  

---

### 📌 Operational Recommendations

- Delay → buffer time, congestion monitoring  
- Cancellation → rebooking, aircraft reassignment  
- On-time → standard operations  

---

### 💰 Business Impact Insights

- Cost implications  
- Operational efficiency  
- Customer satisfaction impact  

---

## 🤖 AI Copilot (Gemini 2.5 Flash)

Integrated AI layer that:

- Explains prediction results  
- Identifies key disruption drivers  
- Provides human-readable recommendations  

---

## 🔐 Secure API Handling

- Local: .env file  
- Deployment: Streamlit Secrets  

---

## 🖥️ Streamlit Application Features

- Premium UI dashboard  
- Real-time prediction  
- AI-generated insights  
- Decision support system  
- Business impact analysis  

---

## 📂 Project Structure

flight_delay_app/ │ ├── app.py                  # Streamlit application │ ├── model/ │   ├── model.pkl │   ├── model_columns.pkl │   ├── le_tail.pkl │   ├── route_frequency.pkl │   ├── airline_stats.pkl │   ├── airport_stats.pkl │   ├── aircraft_stats.pkl │   ├── top_airlines.pkl │   ├── top_airports.pkl │ ├── notebooks/ │   └── full_project.ipynb │ ├── requirements.txt ├── README.md ├── .gitignore

---

## 🚀 Deployment

- Platform: Streamlit Cloud  
- Auto-deployment via GitHub  
- CI/CD enabled  

---

## 🔐 Security Best Practices

- API keys NOT stored in code  
- .env excluded via .gitignore  
- Secrets managed securely in deployment  

---

## 📊 Business Insights Derived

- Airport congestion is the strongest delay driver  
- Airline reliability significantly impacts outcomes  
- Aircraft-level history improves prediction accuracy  
- Peak hours increase disruption probability  

---

## 🚀 Future Enhancements

- Weather data integration  
- Real-time flight API integration  
- Cost prediction layer  
- SHAP-based explainability  
- Network-level simulation  

---

## 👨‍💻 Author

*Soumyadip Ghosh*  
MBA | Data Science & Analytics  

---

## ⭐ Acknowledgment

Dataset provided by:  
U.S. Bureau of Transportation Statistics (BTS)

---

## ⭐ If you found this useful

Give it a ⭐ on GitHub!