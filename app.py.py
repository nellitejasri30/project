import streamlit as st
import numpy as np
import joblib

# Load saved model and scaler
model = joblib.load("best_rf_model.pkl")
scaler = joblib.load("standard_scaler.pkl")

st.set_page_config(page_title="Bike Rental Predictor", page_icon="🚴")
st.title("🚴 Bike Rental Demand Prediction")
st.markdown("Provide input values to estimate the number of bike rentals (`cnt`).")

# 🔹 Non-scaled categorical inputs
season = st.selectbox("Season", [1, 2, 3, 4], format_func=lambda x: {1:"Spring", 2:"Summer", 3:"Fall", 4:"Winter"}[x])
yr = st.selectbox("Year", [0, 1, 2, 3], format_func=lambda x: {0: "2021", 1: "2022", 2: "2023", 3: "2024"}[x])
mnth = st.selectbox("Month", list(range(1, 13)))
holiday = st.radio("Holiday?", [0, 1], format_func=lambda x: "Yes" if x else "No")
weekday = st.selectbox("Weekday", list(range(0, 7)), format_func=lambda x: ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"][x])
workingday = st.radio("Working Day?", [0, 1], format_func=lambda x: "Yes" if x else "No")
weathersit = st.selectbox("Weather Situation", [1, 2, 3, 4], format_func=lambda x: {
    1: "Clear",
    2: "Mist + Cloudy",
    3: "Light Snow/Rain",
    4: "Heavy Rain/Snow"
}[x])

# 🔹 Continuous inputs (to be scaled)
temp = st.number_input("Temperature (Normalized 0-1)", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
atemp = st.number_input("Feels Like Temperature (Normalized 0-1)", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
hum = st.number_input("Humidity (Normalized 0-1)", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
windspeed = st.number_input("Windspeed (Normalized 0-1)", min_value=0.0, max_value=1.0, value=0.3, step=0.01)

# 1️⃣ Prepare input arrays
categorical_features = [season, yr, mnth, holiday, weekday, workingday, weathersit]
numerical_features = np.array([[temp, atemp, hum, windspeed]])
scaled_numerical = scaler.transform(numerical_features)[0]

# 2️⃣ Final combined input for prediction
final_features = np.array(categorical_features + list(scaled_numerical)).reshape(1, -1)

# 3️⃣ Predict on button click
if st.button("Predict 🚀"):
    prediction = model.predict(final_features)[0]
    st.success(f"📈 Estimated Bike Rentals: **{int(prediction)}**")
