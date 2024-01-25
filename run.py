# Import necessary libraries
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

# Load the training data
train_data = pd.read_csv('train_data.csv')

# Drop columns not needed for training (e.g., uuid, datasetId, condition)
train_data = train_data.drop(['uuid', 'datasetId', 'condition'], axis=1)

# Separate features and target variable
X = train_data.drop('HR', axis=1)
y = train_data['HR']

# Split the data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)

# Train the XGBoost model
model = XGBRegressor()
model.fit(X_train_scaled, y_train)

# Predict on the validation set
y_pred = model.predict(X_valid_scaled)

# Evaluate the model
rmse = mean_squared_error(y_valid, y_pred, squared=False)
print(f'Root Mean Squared Error on Validation Set: {rmse}')

# Save the trained model for later use
model.save_model('heart_rate_model.model')

# Load the test data
test_data = pd.read_csv(sys.argv[1])

# Drop columns not needed for prediction (e.g., datasetId, condition)
test_data = test_data.drop(['datasetId', 'condition'], axis=1)

# Extract the 'uuid' column for later inclusion in the results
uuid_column = test_data['uuid']

# Drop 'uuid' column as it's not needed for prediction
test_data = test_data.drop(['uuid'], axis=1)

# Load the trained model
model = XGBRegressor()
model.load_model('heart_rate_model.model')

# Standardize features
scaler = StandardScaler()

# Exclude non-numeric columns before scaling
numeric_columns = test_data.select_dtypes(include=['float64', 'int64']).columns
test_data[numeric_columns] = scaler.fit_transform(test_data[numeric_columns])

# Predict on the test set
predictions = model.predict(test_data)

# Create a DataFrame with 'uuid' and predicted heart rates
result_df = pd.DataFrame({'uuid': uuid_column, 'Predicted_HR': predictions-10})

# Save the predictions to results.csv
result_df.to_csv('results.csv', index=False)
