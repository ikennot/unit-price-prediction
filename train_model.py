import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib
import os

# Load dataset
df = pd.read_csv('data.csv')

# Features and target
X = df[['bathroom', 'bedroom', 'floor_area', 'lot_size', 'year']]
y = df['unit_price']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split for evaluation
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train Random Forest model
model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print("✅ Model trained successfully!")
print("R² Score:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))

# Save model and scaler
os.makedirs('model', exist_ok=True)
joblib.dump(model, 'model/unit_price_model.pkl')
joblib.dump(scaler, 'model/scaler.pkl')

print("\n📦 Model and Scaler saved in 'model/' folder.")
