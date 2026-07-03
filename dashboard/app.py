from pathlib import Path

import joblib
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = REPO_ROOT / "models" / "model.pkl"
SCALER_PATH = REPO_ROOT / "models" / "scaler.pkl"

DATA_URL = (
    "https://raw.githubusercontent.com/wessamsw/"
    "Airline_Passenger_Satisfaction/main/airline_passenger_satisfaction.csv"
)

FEATURE_COLS = [
    "Gender_encoded", "Customer Type_encoded", "Type of Travel_encoded",
    "Class_encoded", "Age", "Flight Distance", "Departure Delay",
    "Arrival Delay", "Departure and Arrival Time Convenience",
    "Ease of Online Booking", "Check-in Service", "Online Boarding",
    "Gate Location", "On-board Service", "Seat Comfort", "Leg Room Service",
    "Cleanliness", "Food and Drink", "In-flight Service",
    "In-flight Wifi Service", "In-flight Entertainment", "Baggage Handling",
]
SCALE_COLS = ["Age", "Flight Distance", "Departure Delay", "Arrival Delay"]

GENDER_MAP = {"Female": 0, "Male": 1}
TRAVEL_TYPE_MAP = {"Business": 0, "Personal": 1}
CLASS_MAP = {"Business": 0, "Economy": 1, "Economy Plus": 2}

MODEL_ACCURACY = "96.24%"

st.set_page_config(
    page_title="Airline Satisfaction Dashboard",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.stApp {
  background-image:
    linear-gradient(rgba(235, 245, 255, 0.92), rgba(235, 245, 255, 0.92)),
    url("https://images.unsplash.com/photo-1436491865332-7a61a109cc05?w=1920&q=80");
  background-size: cover;
  background-position: center;
  background-attachment: fixed;
}
[data-testid="stSidebar"] {
  background-color: rgba(10, 50, 120, 0.92) !important;
  border-right: 3px solid #1A73E8;
}
[data-testid="stSidebar"] * { color: white !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label {
  color: #A8D4FF !important;
  font-weight: bold;
}
[data-testid="metric-container"] {
  background-color: rgba(255, 255, 255, 0.85);
  border: 2px solid #1A73E8;
  border-radius: 12px;
  padding: 15px;
  box-shadow: 2px 4px 12px rgba(26, 115, 232, 0.2);
}
h1 { color: #0A3278 !important; text-align: center; }
h2 {
  color: #1A73E8 !important;
  border-left: 4px solid #1A73E8;
  padding-left: 10px;
}
h3 { color: #0A3278 !important; }
.stButton > button {
  background-color: #1A73E8;
  color: white;
  border: none;
  border-radius: 8px;
  padding: 12px 24px;
  font-size: 16px;
  font-weight: bold;
  width: 100%;
}
.stButton > button:hover { background-color: #0A3278; }
.info-box {
  background-color: rgba(26, 115, 232, 0.1);
  border-left: 4px solid #1A73E8;
  border-radius: 5px;
  padding: 12px 16px;
  margin: 10px 0;
  color: #0A3278;
  font-size: 0.95em;
}
p, li { color: #0A3278; }
</style>
""", unsafe_allow_html=True)

try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    plt.style.use("seaborn-whitegrid" if "seaborn-whitegrid" in plt.style.available else "default")

mpl.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#F0F7FF",
    "axes.edgecolor": "#1A73E8",
    "text.color": "#0A3278",
    "axes.labelcolor": "#0A3278",
    "xtick.color": "#0A3278",
    "ytick.color": "#0A3278",
    "grid.color": "#BDD7F5",
    "grid.alpha": 0.5,
})


@st.cache_data
def load_data():
    return pd.read_csv(DATA_URL)


@st.cache_resource
def load_model_and_scaler():
    return joblib.load(MODEL_PATH), joblib.load(SCALER_PATH)


df = load_data()

st.markdown(f"""
<div style='text-align: center; padding: 20px 0 10px 0;'>
  <h1 style='font-size: 2.8em;'>✈️ Airline Passenger Satisfaction Dashboard</h1>
  <p style='color: #1A73E8; font-size: 1.15em; font-weight: bold;'>
    Customer Experience Analytics — Agile Data Science PMA
  </p>
  <p style='color: #555; font-size: 0.95em;'>
    Powered by Random Forest Classification Model | MRTB 2173
  </p>
</div>
""", unsafe_allow_html=True)
st.markdown("---")

st.sidebar.markdown("""
<div style='text-align: center; padding: 15px 0 5px 0;'>
  <h2 style='color: white; font-size: 1.3em;'>🔍 Filter Options</h2>
  <p style='color: #A8D4FF; font-size: 0.85em;'>Adjust filters to explore passenger data</p>
</div>
""", unsafe_allow_html=True)

travel_class = st.sidebar.selectbox("Select Travel Class", ["All"] + sorted(df["Class"].unique()))
travel_type = st.sidebar.selectbox("Select Type of Travel", ["All"] + sorted(df["Type of Travel"].unique()))
age_range = st.sidebar.slider("Select Age Range", int(df["Age"].min()), int(df["Age"].max()), (20, 60))

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='text-align: center;'>
  <p style='color: #A8D4FF; font-size: 0.8em;'>
    MRTB 2173 Agile Data Science PMA<br>Airline Passenger Satisfaction
  </p>
</div>
""", unsafe_allow_html=True)

fdf = df.copy()
if travel_class != "All":
    fdf = fdf[fdf["Class"] == travel_class]
if travel_type != "All":
    fdf = fdf[fdf["Type of Travel"] == travel_type]
fdf = fdf[fdf["Age"].between(*age_range)]

st.markdown("## 📋 Summary Overview")
c1, c2, c3, c4 = st.columns(4)
c1.metric("✈️ Total Passengers", f"{len(fdf):,}")
c2.metric("😊 Satisfaction Rate", f"{(fdf['Satisfaction'] == 'Satisfied').mean() * 100:.1f}%")
c3.metric("⏱️ Avg Departure Delay", f"{fdf['Departure Delay'].mean():.1f} mins")
c4.metric("🗺 Avg Flight Distance", f"{fdf['Flight Distance'].mean():.0f} km")
st.markdown("---")

st.markdown("## 📊 Data Visualizations")
col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Passenger Satisfaction Distribution")
    counts = fdf["Satisfaction"].value_counts()
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(counts.index, counts.values, color=["#E74C3C", "#27AE60"],
                   edgecolor="white", linewidth=1.5, width=0.5)
    ax.set_title("Satisfaction Count", fontsize=13, fontweight="bold")
    ax.set_ylabel("Count")
    for bar, count in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 200,
                f"{count:,}", ha="center", fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

with col2:
    st.subheader("2. Age Distribution of Passengers")
    fig, ax = plt.subplots(figsize=(7, 4))
    fdf["Age"].hist(bins=30, color="#1A73E8", edgecolor="white", linewidth=0.8, ax=ax, alpha=0.85)
    ax.set_title("Passenger Age Distribution", fontsize=13, fontweight="bold")
    ax.set_xlabel("Age")
    ax.set_ylabel("Frequency")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

st.subheader("3. Average In-flight Service Ratings")
service_cols = ["Seat Comfort", "Food and Drink", "In-flight Service",
                 "In-flight Entertainment", "Cleanliness", "Leg Room Service"]
avg = fdf[service_cols].mean().sort_values()
mean_rating = avg.mean()
colors = ["#1A73E8" if v >= mean_rating else "#64B5F6" for v in avg.values]

fig, ax = plt.subplots(figsize=(10, 4))
bars = ax.barh(avg.index, avg.values, color=colors, edgecolor="white", linewidth=0.8, alpha=0.9)
ax.set_title("Average Service Quality Ratings (1-5 scale)", fontsize=13, fontweight="bold")
ax.set_xlabel("Average Rating (1-5)")
ax.set_xlim(0, 5.5)
for bar, val in zip(bars, avg.values):
    ax.text(val + 0.05, bar.get_y() + bar.get_height() / 2, f"{val:.2f}",
            va="center", fontweight="bold")
ax.axvline(mean_rating, color="#E74C3C", linestyle="--", linewidth=2,
           label=f"Average ({mean_rating:.2f})")
ax.legend(facecolor="white", edgecolor="#1A73E8")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
st.pyplot(fig, use_container_width=True)
st.markdown("---")

st.markdown("## 🔮 Predict Passenger Satisfaction")
st.markdown(f"""
<div class='info-box'>
Fill in passenger details below to generate a real-time satisfaction prediction
using the trained Random Forest model ({MODEL_ACCURACY} accuracy).
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    st.markdown("**✈️ Flight & Passenger Details**")
    age = st.number_input("Age", 1, 100, 35)
    distance = st.number_input("Flight Distance (km)", 0, 10000, 1000)
    gender = st.selectbox("Gender", list(GENDER_MAP))
    p_class = st.selectbox("Class", list(CLASS_MAP))
    p_travel = st.selectbox("Type of Travel", list(TRAVEL_TYPE_MAP))
with col2:
    st.markdown("**⭐ Service Quality Ratings (1=Poor, 5=Excellent)**")
    seat = st.slider("Seat Comfort", 1, 5, 3)
    wifi = st.slider("In-flight Wifi", 1, 5, 3)
    boarding = st.slider("Online Boarding", 1, 5, 3)
    entertainment = st.slider("In-flight Entertainment", 1, 5, 3)
    food = st.slider("Food and Drink", 1, 5, 3)

st.markdown("")
if st.button("🚀 Predict Satisfaction"):
    try:
        model, scaler = load_model_and_scaler()

        row = {col: 3 for col in FEATURE_COLS}
        row.update({
            "Gender_encoded": GENDER_MAP[gender],
            "Customer Type_encoded": 1,
            "Type of Travel_encoded": TRAVEL_TYPE_MAP[p_travel],
            "Class_encoded": CLASS_MAP[p_class],
            "Age": age, "Flight Distance": distance,
            "Departure Delay": 0, "Arrival Delay": 0,
            "Seat Comfort": seat, "In-flight Wifi Service": wifi,
            "Online Boarding": boarding,
            "In-flight Entertainment": entertainment,
            "Food and Drink": food,
        })
        X_input = pd.DataFrame([row])[FEATURE_COLS]
        X_input[SCALE_COLS] = scaler.transform(X_input[SCALE_COLS])

        proba = model.predict_proba(X_input)[0]
        st.markdown("")
        if model.predict(X_input)[0] == 1:
            st.success(f"✅ Predicted: SATISFIED | Confidence: {proba[1]*100:.1f}%")
        else:
            st.error(f"❌ Predicted: NEUTRAL OR DISSATISFIED | Confidence: {proba[0]*100:.1f}%")

        m1, m2 = st.columns(2)
        m1.metric("😊 Satisfaction Probability", f"{proba[1]*100:.1f}%")
        m2.metric("😞 Dissatisfaction Probability", f"{proba[0]*100:.1f}%")
    except FileNotFoundError:
        st.warning("⚠️ Model file not found. Ensure models/model.pkl and models/scaler.pkl exist.")

st.markdown("---")

st.markdown("## 📈 Monitoring Metrics & Drift Analysis")

st.subheader("Monitoring Metric 1: Overall Satisfaction Rate")
overall = (df["Satisfaction"] == "Satisfied").mean() * 100
filtered = (fdf["Satisfaction"] == "Satisfied").mean() * 100
c1, c2, c3 = st.columns(3)
c1.metric("Overall Satisfaction Rate", f"{overall:.2f}%")
c2.metric("Filtered Satisfaction Rate", f"{filtered:.2f}%", delta=f"{filtered - overall:.2f}% vs overall")
c3.metric("Passengers in Current View", f"{len(fdf):,}")

st.subheader("Monitoring Metric 2: Data Quality")
missing_total = int(df.isnull().sum().sum())
duplicate_total = int(df.duplicated().sum())
invalid_cleanliness = int((~df["Cleanliness"].between(1, 5)).sum())
c1, c2, c3 = st.columns(3)
c1.metric("Missing Values", missing_total, delta="Arrival Delay column", delta_color="inverse")
c2.metric("Duplicate Rows", duplicate_total)
c3.metric("Invalid Cleanliness Ratings", invalid_cleanliness, delta_color="inverse")

st.subheader("Data Drift Analysis: Old vs New Data")
st.markdown("""
<div class='info-box'>
Simulating drift by comparing first half (old data) vs second half (new data)
of the dataset to detect shifts in passenger behaviour and operational patterns over time.
</div>
""", unsafe_allow_html=True)

mid = len(df) // 2
old, new = df.iloc[:mid], df.iloc[mid:]
old_sat = (old["Satisfaction"] == "Satisfied").mean() * 100
new_sat = (new["Satisfaction"] == "Satisfied").mean() * 100
old_delay = old["Departure Delay"].mean()
new_delay = new["Departure Delay"].mean()

col1, col2 = st.columns(2)
with col1:
    st.markdown("**Satisfaction Rate Drift**")
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(["Old Data\n(First Half)", "New Data\n(Second Half)"], [old_sat, new_sat],
           color=["#1A73E8", "#FF7043"], edgecolor="white", linewidth=1.5, width=0.4)
    ax.set_title("Satisfaction Rate: Old vs New", fontsize=11, fontweight="bold")
    ax.set_ylabel("Satisfaction Rate (%)")
    ax.set_ylim(0, 100)
    for i, v in enumerate([old_sat, new_sat]):
        ax.text(i, v + 1, f"{v:.2f}%", ha="center", fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    st.metric("Satisfaction Drift", f"{abs(new_sat - old_sat):.2f}%", delta=f"{new_sat - old_sat:.2f}%")

with col2:
    st.markdown("**Departure Delay Drift**")
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(["Old Data\n(First Half)", "New Data\n(Second Half)"], [old_delay, new_delay],
           color=["#1A73E8", "#FF7043"], edgecolor="white", linewidth=1.5, width=0.4)
    ax.set_title("Avg Departure Delay: Old vs New", fontsize=11, fontweight="bold")
    ax.set_ylabel("Average Delay (minutes)")
    for i, v in enumerate([old_delay, new_delay]):
        ax.text(i, v + 0.2, f"{v:.2f}", ha="center", fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    st.metric("Delay Drift", f"{abs(new_delay - old_delay):.2f} mins", delta=f"{new_delay - old_delay:.2f} mins")

st.markdown("---")
st.markdown(f"""
<div style='text-align: center; padding: 15px 0; color: #1A73E8; font-size: 0.9em;'>
  ✈️ MRTB 2173 Agile Data Science PMA | Airline Passenger Satisfaction Analysis |
  Random Forest Model ({MODEL_ACCURACY} Accuracy)
</div>
""", unsafe_allow_html=True)
