import streamlit as st
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load("ipl_knn_model.joblib")
scaler = joblib.load("scaler.joblib")

# Title
st.title("🏏 IPL Final Score Predictor using KNN")

# Input form
bat_teams = [
    "Mumbai Indians", "Royal Challengers", "Delhi Capitals", "Chennai Super Kings",
    "Rajasthan Royals", "Sunrisers Hyderabad", "Kings XI Punjab", "Kolkata Knight Riders"
]
bowl_teams = bat_teams

bat_team = st.selectbox("Select Batting Team", bat_teams)
bowl_team = st.selectbox("Select Bowling Team", bowl_teams)
overs = st.slider("Overs Completed", 0.1, 20.0, 5.0, step=0.1)
runs = st.number_input("Current Runs", 0, 300, 50)
wickets = st.number_input("Wickets Lost", 0, 10, 2)
runs_last_5 = st.number_input("Runs in Last 5 Overs", 0, 100, 30)

# Create input features
input_data = {
    'overs': overs,
    'runs': runs,
    'wickets': wickets,
    'runs_last_5': runs_last_5
}

# One-hot encoding: match training order
for team in bat_teams[1:]:  # drop_first=True during training
    input_data[f'bat_team_{team}'] = 1 if bat_team == team else 0
for team in bowl_teams[1:]:
    input_data[f'bowl_team_{team}'] = 1 if bowl_team == team else 0

input_df = pd.DataFrame([input_data])

# Apply scaler and predict
scaled_input = scaler.transform(input_df)
if st.button("Predict Score"):
    prediction = model.predict(scaled_input)[0]
    st.success(f"🎯 Predicted Final Score: {round(prediction)} runs")
