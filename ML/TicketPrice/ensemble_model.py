import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import StackingRegressor
import kagglehub


# Load dataset (example: historical ticket pricing data)
data = kagglehub.dataset_download("ibrahimelsayed182/plane-ticket-price")
# data = pd.read_csv('ticket_pricing_data.csv')
X = data.drop(columns=['ticket_price'])  # Features
y = data['ticket_price']  # Target variable

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Base models
linear_model = LinearRegression()
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
nn_model = MLPRegressor(hidden_layer_sizes=(50, 50), max_iter=1000, random_state=42)

# Train base models
linear_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)
gb_model.fit(X_train, y_train)
nn_model.fit(X_train, y_train)

# Time series model (assuming time-based data)
time_series_data = data[['date', 'ticket_price']].set_index('date')
arima_model = ARIMA(time_series_data, order=(5, 1, 0)).fit()

# Meta-model (stacking)
base_models = [
    ('linear', linear_model),
    ('rf', rf_model),
    ('gb', gb_model),
    ('nn', nn_model)
]
stacking_model = StackingRegressor(estimators=base_models, final_estimator=LinearRegression())

# Train stacking model
stacking_model.fit(X_train, y_train)

# Evaluate ensemble model
y_pred = stacking_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Ensemble Model MSE: {mse}')

# Predict optimal ticket price
optimal_price = stacking_model.predict(new_data)  # new_data: features for the new event
print(f'Optimal Ticket Price: {optimal_price}')