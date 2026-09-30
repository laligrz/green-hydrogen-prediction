import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import streamlit as st

# Page Settings
st.set_page_config(
    page_title="Hydrogen Production Predictor", page_icon="🔋", layout="centered"
)

st.title(" Green Hydrogen Production Forecasting Modelر")


# Data loading function and model training with caching to accelerate performance
@st.cache_data
def load_and_train_model():
  try:
    df = pd.read_csv("realistic_hydrogen_data.csv")
  except FileNotFoundError:
    return None, None, None

  X = df[["Wind_Speed_m/s", "Electrolyzer_Efficiency_%", "Wind_Power_kW"]]
  y = df["Hydrogen_Production_kg/day"]

  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.2, random_state=42
  )

  # Building the AdaBoost model with optimal parameters
  model = AdaBoostRegressor(
      n_estimators=120, learning_rate=0.08, random_state=42
  )
  model.fit(X_train, y_train)

  # Accuracy calculation
  y_pred = model.predict(X_test)
  r2 = r2_score(y_test, y_pred)

  return model, r2, df


model, r2_score_val, df = load_and_train_model()

if model is None:
  st.error(
      " Warning: Data file not found 'realistic_hydrogen_data.csv'."
      " Please run the data generation code first!"
  )
else:
  st.success(
      f" Data upload and model training were successful!✅ | Model accuracy ($R^2$):"
      f" {r2_score_val:.4f}"
  )

  # Side panel for entering variables
  st.sidebar.header(" Renewable station inputs")

  min_ws = float(df["Wind_Speed_m/s"].min())
  max_ws = float(df["Wind_Speed_m/s"].max())
  input_wind_speed = st.sidebar.slider(
      "Wind speed (m/s)", min_ws, max_ws, 10.0
  )

  min_eff = float(df["Electrolyzer_Efficiency_%"].min())
  max_eff = float(df["Electrolyzer_Efficiency_%"].max())
  input_efficiency = st.sidebar.slider(
      "Electrolyte analyzer efficiency (%)", min_eff, max_eff, 70.0
  )

  # Calculating instantaneous wind energy based on the cube of the wind speed
  air_density = 1.225
  rotor_area = 5000
  calculated_wind_power = (
      0.5 * air_density * rotor_area * (input_wind_speed**3)
  ) / 1000

  st.sidebar.info(
      f" Physically calculated wind energy **{calculated_wind_power:.2f} kW**"
  )

  # Prediction button
  if st.button(" Calculate the expected hydrogen production"):
    user_input = np.array(
        [[input_wind_speed, input_efficiency, calculated_wind_power]]
    )
    predicted_hydrogen = model.predict(user_input)[0]

    st.metric(
        label="Total expected hydrogen production
",
        value=f"{predicted_hydrogen:.2f} kg/day",
    )
