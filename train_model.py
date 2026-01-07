import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib
import os

# =============================
# 1️⃣ Load dataset
# =============================
df = pd.read_csv('data.csv')

# Basic check
print("Data shape:", df.shape)
print("Missing values:\n", df.isnull().sum())

# =============================
# 2️⃣ Define features and target
# =============================
X = df[['bathroom', 'bedroom', 'floor_area', 'lot_size', 'year']]
y = df['unit_price']

# =============================
# 3️⃣ Scale features
# =============================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =============================
# 4️⃣ Train-test split
# =============================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# =============================
# 5️⃣ Hyperparameter tuning
# =============================
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(
    estimator=RandomForestRegressor(random_state=42),
    param_grid=param_grid,
    cv=3,
    n_jobs=-1,
    scoring='r2'
)

grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

print("🔍 Best Parameters Found:")
print(grid_search.best_params_)

# =============================
# 6️⃣ Evaluate model
# =============================
y_pred = best_model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("\n✅ Model trained successfully!")
print(f"R² Score: {r2:.4f}")
print(f"MSE: {mse:.2f}")
print(f"MAE: {mae:.2f}")

# =============================
# 7️⃣ Save model and scaler
# =============================
os.makedirs('model', exist_ok=True)
joblib.dump(best_model, 'model/unit_price_model.pkl')
joblib.dump(scaler, 'model/scaler.pkl')

print("\n📦 Model and Scaler saved in 'model/' folder.")
