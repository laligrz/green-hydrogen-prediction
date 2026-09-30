import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

print("--- Starting the Hybrid AdaBoost Hydrogen Production Model ---")

# 1. Load the thermodynamics-based physical database
try:
    df = pd.read_csv("realistic_hydrogen_data.csv")
    print("The physics-based data file ('realistic_hydrogen_data.csv') was loaded successfully!")
except FileNotFoundError:
    print(
        "Error: The data file was not found. Please make sure to run "
        "the data generation code first."
    )
    exit()

# 2. Define the input matrix (X) and target output (y)
# Inputs include: wind speed, electrolyzer efficiency,
# and the physically calculated wind power
X = df[["Wind_Speed_m/s", "Electrolyzer_Efficiency_%", "Wind_Power_kW"]]
y = df["Hydrogen_Production_kg/day"]

# 3. Split the data: 80% for training and 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(
    f"Data successfully split: {len(X_train)} samples for training, "
    f"{len(X_test)} samples for testing."
)

# 4. Configure the AdaBoost model using the optimized parameters
# obtained from BWOA optimization
optimized_adaboost = AdaBoostRegressor(
    n_estimators=120,      # Number of stages/trees
    learning_rate=0.08,   # Optimal learning rate
    random_state=42       # Fix the random seed for reproducibility
)

# 5. Train the actual model on the physics-based training data
print("Training the AdaBoost model...")
optimized_adaboost.fit(X_train, y_train)
print("Training completed successfully!")

# 6. Make predictions and evaluate the performance on the test data
y_pred = optimized_adaboost.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# 7. Print the final evaluation results
print("\n" + "=" * 40)
print("       Hybrid Model Performance Evaluation Results       ")
print("=" * 40)
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R² Score (Explained Variance): {r2:.4f}")
print("=" * 40)
