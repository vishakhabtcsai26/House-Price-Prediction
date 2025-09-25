# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.datasets import fetch_california_housing

# California housing dataset load karo
house_price = fetch_california_housing(as_frame=True)

# Features aur target alag karo
X = house_price.data
y = house_price.target

print("Dataset shape:", X.shape)
print(X.head())


# %%
# DataFrame bana lo
df = pd.DataFrame(X, columns=house_price.feature_names)

# Target column add karna
df["Price"] = y

# Top 5 rows dekhna
print(df.head())


# %%
# Features aur target
X = df.drop("Price", axis=1)
y = df["Price"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)


# %%
# Model training
from sklearn.linear_model import LinearRegression

# Linear Regression model
model = LinearRegression()

# Train the model on training data
model.fit(X_train, y_train)

# Predict on training data
training_data_prediction = model.predict(X_train)

print("Training Prediction:", training_data_prediction[:10])  # first 10 predictions


# %%
from sklearn.linear_model import LinearRegression

# Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)


# %%
# Predict on training data
training_data_prediction = model.predict(X_train)

print("First 10 Training Predictions:", training_data_prediction[:10])


# %%
from xgboost import XGBRegressor


# %%
# XGBoost Regressor
xgb_model = XGBRegressor()

# Train the model
xgb_model.fit(X_train, y_train)


# %%
# Predict on training data
xgb_train_pred = xgb_model.predict(X_train)


# %%
# Predict on test data
xgb_test_pred = xgb_model.predict(X_test)


# %%
from sklearn import metrics
import numpy as np

# Training Performance
r2_train_xgb = metrics.r2_score(y_train, xgb_train_pred)
mae_train_xgb = metrics.mean_absolute_error(y_train, xgb_train_pred)
rmse_train_xgb = np.sqrt(metrics.mean_squared_error(y_train, xgb_train_pred))

print("XGBoost Training R2:", r2_train_xgb)
print("XGBoost Training MAE:", mae_train_xgb)
print("XGBoost Training RMSE:", rmse_train_xgb)

# Testing Performance
r2_test_xgb = metrics.r2_score(y_test, xgb_test_pred)
mae_test_xgb = metrics.mean_absolute_error(y_test, xgb_test_pred)
rmse_test_xgb = np.sqrt(metrics.mean_squared_error(y_test, xgb_test_pred))

print("XGBoost Testing R2:", r2_test_xgb)
print("XGBoost Testing MAE:", mae_test_xgb)
print("XGBoost Testing RMSE:", rmse_test_xgb)


# %%
# Prediction on training data
y_train_pred = model.predict(X_train)

# Prediction on testing data
y_test_pred = model.predict(X_test)

import matplotlib.pyplot as plt

# Training data
plt.scatter(y_train, y_train_pred, color='blue')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Training Data: Actual vs Predicted')
plt.show()

# Testing data
plt.scatter(y_test, y_test_pred, color='green')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Testing Data: Actual vs Predicted')
plt.show()


# %%
# Residuals for testing data
residuals = y_test - y_test_pred

plt.scatter(y_test_pred, residuals, color='red')
plt.axhline(y=0, color='black', linestyle='--')
plt.xlabel('Predicted Prices')
plt.ylabel('Residuals')
plt.title('Residual Plot (Testing Data)')
plt.show()


# %%
import xgboost as xgb
import pandas as pd
import matplotlib.pyplot as plt

# Get feature importance for XGBoost model
xgb.plot_importance(xgb_model)
plt.title('Feature Importance')
plt.show()


# %%
from sklearn.model_selection import GridSearchCV
import xgboost as xgb

xgb_model = xgb.XGBRegressor()

params = {
    'n_estimators': [100, 300, 500],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.7, 0.8, 1.0]
}

grid = GridSearchCV(estimator=xgb_model, param_grid=params, scoring='r2', cv=3, n_jobs=-1)
grid.fit(X_train, y_train)

print("Best parameters:", grid.best_params_)
print("Best R2 score:", grid.best_score_)


# %%
print("Training R2:", metrics.r2_score(y_train, y_train_pred))
print("Training MAE:", metrics.mean_absolute_error(y_train, y_train_pred))
print("Training RMSE:", np.sqrt(metrics.mean_squared_error(y_train, y_train_pred)))

print("Testing R2:", metrics.r2_score(y_test, y_test_pred))
print("Testing MAE:", metrics.mean_absolute_error(y_test, y_test_pred))
print("Testing RMSE:", np.sqrt(metrics.mean_squared_error(y_test, y_test_pred)))


# %%
# =============================
# 1. Predictions
# =============================
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# =============================
# 2. Scatter Plots: Actual vs Predicted
# =============================
import matplotlib.pyplot as plt

# Training data
plt.figure(figsize=(8,6))
plt.scatter(y_train, y_train_pred, color='blue')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Training Data: Actual vs Predicted')
plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], 'r--')  # perfect fit line
plt.show()

# Testing data
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_test_pred, color='green')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Testing Data: Actual vs Predicted')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')  # perfect fit line
plt.show()

# =============================
# 3. Residual Plots
# =============================
# Training residuals
residuals_train = y_train - y_train_pred
plt.figure(figsize=(8,6))
plt.scatter(y_train_pred, residuals_train, color='purple')
plt.axhline(y=0, color='black', linestyle='--')
plt.xlabel('Predicted Prices')
plt.ylabel('Residuals')
plt.title('Residual Plot (Training Data)')
plt.show()

# Testing residuals
residuals_test = y_test - y_test_pred
plt.figure(figsize=(8,6))
plt.scatter(y_test_pred, residuals_test, color='red')
plt.axhline(y=0, color='black', linestyle='--')
plt.xlabel('Predicted Prices')
plt.ylabel('Residuals')
plt.title('Residual Plot (Testing Data)')
plt.show()

# =============================
# 4. Feature Importance
# =============================
# Use the best fitted XGBoost model from GridSearchCV for feature importance
import xgboost as xgb

plt.figure(figsize=(10,8))
xgb.plot_importance(grid.best_estimator_, importance_type='weight', max_num_features=10, title='Top 10 Feature Importances')
plt.show()

# =============================
# 5. Evaluation Metrics
# =============================
from sklearn import metrics
import numpy as np

print("---- Training Data ----")
print("R2 Score:", metrics.r2_score(y_train, y_train_pred))
print("MAE:", metrics.mean_absolute_error(y_train, y_train_pred))
print("RMSE:", np.sqrt(metrics.mean_squared_error(y_train, y_train_pred)))

print("\n---- Testing Data ----")
print("R2 Score:", metrics.r2_score(y_test, y_test_pred))
print("MAE:", metrics.mean_absolute_error(y_test, y_test_pred))
print("RMSE:", np.sqrt(metrics.mean_squared_error(y_test, y_test_pred)))


# %%
from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb

# =============================
# 1. Hyperparameter Grid
# =============================
param_grid = {
    'n_estimators': [100, 300, 500, 700],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 4, 5, 6, 7],
    'min_child_weight': [1, 3, 5, 7],
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
    'gamma': [0, 0.1, 0.2, 0.3, 0.4]
}

# =============================
# 2. XGBoost Regressor
# =============================
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

# =============================
# 3. RandomizedSearchCV
# =============================
random_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_grid,
    n_iter=50,           # 50 random combinations
    scoring='r2',
    cv=3,                # 3-fold cross validation
    verbose=2,
    n_jobs=-1,
    random_state=42
)

# =============================
# 4. Fit on training data
# =============================
random_search.fit(X_train, y_train)

# =============================
# 5. Best Parameters & Score
# =============================
print("Best Parameters:", random_search.best_params_)
print("Best CV R2 Score:", random_search.best_score_)

# =============================
# 6. Evaluate on Test Data
# =============================
best_model = random_search.best_estimator_
y_test_pred = best_model.predict(X_test)

from sklearn import metrics
import numpy as np

print("Test R2 Score:", metrics.r2_score(y_test, y_test_pred))
print("Test MAE:", metrics.mean_absolute_error(y_test, y_test_pred))
print("Test RMSE:", np.sqrt(metrics.mean_squared_error(y_test, y_test_pred)))


# %%
import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))
plt.scatter(y_test, y_test_pred, color='blue')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linewidth=2)  # perfect prediction line
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices (XGBoost)")
plt.show()


# %%
residuals = y_test - y_test_pred

plt.figure(figsize=(8,6))
plt.scatter(y_test_pred, residuals, color='green')
plt.hlines(y=0, xmin=min(y_test_pred), xmax=max(y_test_pred), colors='red', linewidth=2)
plt.xlabel("Predicted Prices")
plt.ylabel("Residuals")
plt.title("Residual Plot (XGBoost)")
plt.show()


# %%
import xgboost as xgb

xgb.plot_importance(best_model, max_num_features=10, importance_type='weight', height=0.8)
plt.title("Top 10 Feature Importances (XGBoost)")
plt.show()


# %%
import pandas as pd

# Metrics
results = pd.DataFrame({
    "Model": ["Linear Regression", "XGBoost"],
    "Train R2": [metrics.r2_score(y_train, model.predict(X_train)), metrics.r2_score(y_train, best_model.predict(X_train))],
    "Test R2": [metrics.r2_score(y_test, model.predict(X_test)), metrics.r2_score(y_test, best_model.predict(X_test))],
    "Train MAE": [metrics.mean_absolute_error(y_train, model.predict(X_train)), metrics.mean_absolute_error(y_train, best_model.predict(X_train))],
    "Test MAE": [metrics.mean_absolute_error(y_test, model.predict(X_test)), metrics.mean_absolute_error(y_test, best_model.predict(X_test))],
    "Train RMSE": [np.sqrt(metrics.mean_squared_error(y_train, model.predict(X_train))), np.sqrt(metrics.mean_squared_error(y_train, best_model.predict(X_train)))],
    "Test RMSE": [np.sqrt(metrics.mean_squared_error(y_test, model.predict(X_test))), np.sqrt(metrics.mean_squared_error(y_test, best_model.predict(X_test)))]
})

results


# %%
import joblib

# Save XGBoost model
joblib.dump(best_model, "xgboost_house_price_model.pkl")

# Load later
# loaded_model = joblib.load("xgboost_house_price_model.pkl")
# predictions = loaded_model.predict(X_test)



