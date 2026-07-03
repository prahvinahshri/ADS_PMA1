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

st.set_page_config(page_title="Airline Satisfaction Dashboard",
                   page_icon="✈", layout="wide")

st.markdown("""
<style>
.stApp {
  background-image: linear-gradient(rgba(235,245,255,.92), rgba(235,245,255,.92)),
    url("https://images.unsplash.com/photo-1436491865332-7a61a109cc05?w=1920&q=80");
  background-size: cover; background-position: center; background-attachment: fixed;
}
[data-testid="stSidebar"] { background-color: rgba(10,50,120,.92) !important;
  border-right: 3px solid #1A73E8; }
[data-testid="stSidebar"] * { color: white !important; }
h1 { color:#0A3278 !important; text-align:center; }
h2 { color:#1A73E8 !important; border-left:4px solid #1A73E8; padding-left:10px; }
h3, p, li { color:#0A3278; }
.stButton > button { background-color:#1A73E8; color:white; border:none;
  border-radius:8px; padding:12px 24px; font-weight:bold; width:100%; }
.stButton > button:hover { background-color:#0A3278; }
</style>
""", unsafe_allow_html=True)

plt.style.use("seaborn-v0_8-whitegrid")
mpl.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "#F0F7FF",
    "text.color": "#0A3278", "axes.labelcolor": "#0A3278",
    "xtick.color": "#0A3278", "ytick.color": "#0A3278",
})

@st.cache_data
def load_data():
    return pd.read_csv(DATA_URL)

@st.cache_resource
def load_model_and_scaler():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

df = load_data()

st.title("Airline Passenger Satisfaction Dashboard")
st.markdown(
    "<p style='text-align:center;color:#1A73E8;font-weight:bold'>"
    "Customer Experience Analytics — MRTB 2173 Agile Data Science PMA</p>",
    unsafe_allow_html=True)
st.markdown("---")

st.sidebar.header("Filter Options")
travel_class = st.sidebar.selectbox(
    "Travel Class", ["All"] + sorted(df["Class"].unique()))
travel_type = st.sidebar.selectbox(
    "Type of Travel", ["All"] + sorted(df["Type of Travel"].unique()))
age_range = st.sidebar.slider(
    "Age Range", int(df["Age"].min()), int(df["Age"].max()), (20, 60))

fdf = df.copy()
if travel_class != "All":
    fdf = fdf[fdf["Class"] == travel_class]
if travel_type != "All":
    fdf = fdf[fdf["Type of Travel"] == travel_type]
fdf = fdf[fdf["Age"].between(*age_range)]

st.markdown("## Summary Overview")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Passengers", f"{len(fdf):,}")
c2.metric("Satisfaction Rate",
          f"{(fdf['Satisfaction'] == 'Satisfied').mean() * 100:.1f}%")
c3.metric("Avg Departure Delay", f"{fdf['Departure Delay'].mean():.1f} min")
c4.metric("Avg Flight Distance", f"{fdf['Flight Distance'].mean():.0f} km")
st.markdown("---")

st.markdown("## Data Visualizations")
col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Satisfaction Distribution")
    counts = fdf["Satisfaction"].value_counts()
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(counts.index, counts.values,
                  color=["#E74C3C", "#27AE60"], width=0.5)
    ax.bar_label(bars, fmt="{:,.0f}", fontweight="bold", color="#0A3278")
    ax.set_ylabel("Count")
    st.pyplot(fig, use_container_width=True)

with col2:
    st.subheader("2. Age Distribution")
    fig, ax = plt.subplots(figsize=(6, 4))
    fdf["Age"].hist(bins=30, color="#1A73E8", edgecolor="white", ax=ax)
    ax.set_xlabel("Age")
    ax.set_ylabel("Frequency")
    st.pyplot(fig, use_container_width=True)

st.subheader("3. Average In-flight Service Ratings")
service_cols = ["Seat Comfort", "Food and Drink", "In-flight Service",
                "In-flight Entertainment", "Cleanliness", "Leg Room Service"]
avg = fdf[service_cols].mean().sort_values()
fig, ax = plt.subplots(figsize=(10, 4))
bars = ax.barh(avg.index, avg.values, color="#1A73E8")
ax.bar_label(bars, fmt="%.2f", fontweight="bold", color="#0A3278")
ax.axvline(avg.mean(), color="#E74C3C", ls="--",
           label=f"Average ({avg.mean():.2f})")
ax.set_xlim(0, 5.5)
ax.set_xlabel("Average Rating (1-5)")
ax.legend()
st.pyplot(fig, use_container_width=True)
st.markdown("---")

st.markdown("## Predict Passenger Satisfaction")
st.info("Uses the Sprint 3 Random Forest model and the StandardScaler "
        "fitted during training.")

col1, col2 = st.columns(2)
with col1:
    st.markdown("**Flight and Passenger Details**")
    age = st.number_input("Age", 1, 100, 35)
    distance = st.number_input("Flight Distance (km)", 0, 10000, 1000)
    gender = st.selectbox("Gender", list(GENDER_MAP))
    p_class = st.selectbox("Class", list(CLASS_MAP))
    p_travel = st.selectbox("Type of Travel", list(TRAVEL_TYPE_MAP))
with col2:
    st.markdown("**Service Ratings (1 = Poor, 5 = Excellent)**")
    seat = st.slider("Seat Comfort", 1, 5, 3)
    wifi = st.slider("In-flight Wifi", 1, 5, 3)
    boarding = st.slider("Online Boarding", 1, 5, 3)
    entertainment = st.slider("In-flight Entertainment", 1, 5, 3)
    food = st.slider("Food and Drink", 1, 5, 3)

if st.button("Predict Satisfaction"):
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
        if model.predict(X_input)[0] == 1:
            st.success(f"Predicted: SATISFIED (confidence {proba[1]*100:.1f}%)")
        else:
            st.error("Predicted: NEUTRAL OR DISSATISFIED "
                     f"(confidence {proba[0]*100:.1f}%)")
        m1, m2 = st.columns(2)
        m1.metric("Satisfaction Probability", f"{proba[1]*100:.1f}%")
        m2.metric("Dissatisfaction Probability", f"{proba[0]*100:.1f}%")
    except FileNotFoundError:
        st.warning("Model files not found — check models/model.pkl and "
                   "models/scaler.pkl are in the repository.")
st.markdown("---")

st.markdown("## Monitoring Metrics and Drift Analysis")

st.subheader("Overall Satisfaction Rate")
overall = (df["Satisfaction"] == "Satisfied").mean() * 100
filtered = (fdf["Satisfaction"] == "Satisfied").mean() * 100
c1, c2, c3 = st.columns(3)
c1.metric("Overall Satisfaction Rate", f"{overall:.2f}%")
c2.metric("Filtered Satisfaction Rate", f"{filtered:.2f}%",
          delta=f"{filtered - overall:.2f}% vs overall")
c3.metric("Passengers in Current View", f"{len(fdf):,}")

st.subheader("Data Quality")
c1, c2, c3 = st.columns(3)
c1.metric("Missing Values", int(df.isnull().sum().sum()))
c2.metric("Duplicate Rows", int(df.duplicated().sum()))
c3.metric("Invalid Cleanliness Ratings",
          int((~df["Cleanliness"].between(1, 5)).sum()))

st.subheader("Data Drift: Old vs New Data")
st.caption("Old data = first half of the dataset, new data = second half.")

mid = len(df) // 2
old, new = df.iloc[:mid], df.iloc[mid:]
old_sat = (old["Satisfaction"] == "Satisfied").mean() * 100
new_sat = (new["Satisfaction"] == "Satisfied").mean() * 100
old_delay = old["Departure Delay"].mean()
new_delay = new["Departure Delay"].mean()

col1, col2 = st.columns(2)
with col1:
    fig, ax = plt.subplots(figsize=(5, 3))
    bars = ax.bar(["Old", "New"], [old_sat, new_sat],
                  color=["#1A73E8", "#FF7043"], width=0.4)
    ax.bar_label(bars, fmt="%.2f%%", fontweight="bold", color="#0A3278")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Satisfaction Rate (%)")
    ax.set_title("Satisfaction Rate: Old vs New", color="#0A3278")
    st.pyplot(fig, use_container_width=True)
    st.metric("Satisfaction Drift", f"{abs(new_sat - old_sat):.2f}%")
with col2:
    fig, ax = plt.subplots(figsize=(5, 3))
    bars = ax.bar(["Old", "New"], [old_delay, new_delay],
                  color=["#1A73E8", "#FF7043"], width=0.4)
    ax.bar_label(bars, fmt="%.2f", fontweight="bold", color="#0A3278")
    ax.set_ylabel("Average Delay (minutes)")
    ax.set_title("Avg Departure Delay: Old vs New", color="#0A3278")
    st.pyplot(fig, use_container_width=True)
    st.metric("Delay Drift", f"{abs(new_delay - old_delay):.2f} mins")
