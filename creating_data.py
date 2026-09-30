
import numpy as np
import pandas as pd

# Define the number of samples (e.g., 1000 operating hours)
n_samples = 1000

# Fix the random seed to ensure reproducible results
np.random.seed(42)

# 1. Generate wind speed inputs (meters/second) within a realistic range for wind turbines
wind_speed = np.random.uniform(3.0, 25.0, n_samples)

# 2. Generate electrolyzer efficiency (%) within the typical industrial range (60% to 82%)
electrolyzer_efficiency = np.random.uniform(0.60, 0.82, n_samples)

# 3. Calculate wind power based on physics (proportional to the cube of wind speed, v^3)
air_density = 1.225  # Air density at sea level (kg/m^3)
rotor_area = 5000     # Turbine blade swept area in square meters

# Wind power equation in kilowatts
wind_power_kw = (0.5 * air_density * rotor_area * (wind_speed ** 3)) / 1000

# 4. Calculate hydrogen production dynamically using thermodynamic principles
# Higher Heating Value of hydrogen (HHV ~ 39.4 kWh/kg)
hhv_hydrogen = 39.4
daily_hours = 24

# Production amount = (electrical energy * efficiency * operating hours) / heating value
hydrogen_production = (
    wind_power_kw * electrolyzer_efficiency * daily_hours
) / hhv_hydrogen

# Add very small Gaussian noise to represent natural measurement errors
# that can occur in real-world industrial plants
noise = np.random.normal(0, 1.5, n_samples)

# Ensure that there are no negative production values
hydrogen_production = np.clip(hydrogen_production + noise, 0, None)

# 5. Organize the data into a structured DataFrame
df_physics = pd.DataFrame({
    'Wind_Speed_m/s': np.round(wind_speed, 2),
    'Electrolyzer_Efficiency_%': np.round(electrolyzer_efficiency * 100, 2),
    'Wind_Power_kW': np.round(wind_power_kw, 2),
    'Hydrogen_Production_kg/day': np.round(hydrogen_production, 2)
})

# Save the dataset as a CSV file for later use in the AI model
df_physics.to_csv('realistic_hydrogen_data.csv', index=False)

print("The physics-based database was successfully generated and saved as 'realistic_hydrogen_data.csv'!")
```
