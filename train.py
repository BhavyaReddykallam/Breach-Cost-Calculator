import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pickle

# Load the dataset
file_path = 'Non KMB dataset.csv'  # Path to your dataset
data = pd.read_csv(file_path)

# Handle missing values
data.dropna(inplace=True)

# Feature Engineering: Create interaction features to give importance to certain variables
data['Revenue per Employee'] = data['Annual Revenue (USD)'] / data['Number of Employees']
data['Employees per Million Revenue'] = data['Number of Employees'] / (data['Annual Revenue (USD)'] / 1e6)

# Save the valid categories for later use before one-hot encoding
valid_categories = {
    'Industry Type': data['Industry Type'].unique().tolist(),  # Ensures valid Industry Type values are saved
    'Country': data['Country'].unique().tolist(),
    'Incident Type': data['Incident Type'].unique().tolist(),
    'Root Cause': data['Root Cause'].unique().tolist()
}

# One-hot encode categorical variables
categorical_columns = ['Industry Type', 'Country', 'Incident Type', 'Root Cause']
data = pd.get_dummies(data, columns=categorical_columns)

# Separate features (X) and target variable (y)
X = data.drop(columns=[
    'Forensics Cost (USD)', 'PR Crisis Cost (USD)', 'IR Counsel Cost (USD)', 
    'Credit Monitoring Cost (USD)', 'Ransom Payments (USD)', 
    'First Party Costs (USD)', 'Bec loss', 'notification call centre', 'total costs'
])
y = data['total costs']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the pipeline with feature scaling and model
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # This will scale the numeric features
    ('model', RandomForestRegressor(random_state=42))
])

# Hyperparameter tuning
param_grid = {
    'model__n_estimators': [100, 200, 500],
    'model__max_depth': [10, 20, None],
    'model__min_samples_split': [2, 5, 10],
    'model__min_samples_leaf': [1, 2, 4],
    'model__bootstrap': [True, False]
}

random_search = RandomizedSearchCV(pipeline, param_distributions=param_grid, n_iter=10, cv=5, verbose=2, n_jobs=-1, random_state=42)
random_search.fit(X_train, y_train)

# Make predictions on the test set
y_pred = random_search.predict(X_test)

# Evaluate the model
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"R2 Score: {r2}")
print(f"Mean Absolute Error: {mae}")

# Calculate secondary costs as ratios to total costs
data['Forensics_Ratio'] = data['Forensics Cost (USD)'] / data['total costs']
data['PR_Crisis_Ratio'] = data['PR Crisis Cost (USD)'] / data['total costs']
data['IR_Counsel_Ratio'] = data['IR Counsel Cost (USD)'] / data['total costs']
data['Credit_Monitoring_Ratio'] = data['Credit Monitoring Cost (USD)'] / data['total costs']
data['Ransom_Ratio'] = data['Ransom Payments (USD)'] / data['total costs']
data['First_Party_Ratio'] = data['First Party Costs (USD)'] / data['total costs']
data['BEC_Loss_Ratio'] = data['Bec loss'] / data['total costs']
data['Notification_Ratio'] = data['notification call centre'] / data['total costs']

# Calculate average ratios to use for future predictions
average_ratios = {
    'forensics': data['Forensics_Ratio'].mean(),
    'pr_crisis': data['PR_Crisis_Ratio'].mean(),
    'ir_counsel': data['IR_Counsel_Ratio'].mean(),
    'credit_monitoring': data['Credit_Monitoring_Ratio'].mean(),
    'ransom': data['Ransom_Ratio'].mean(),
    'first_party': data['First_Party_Ratio'].mean(),
    'bec_loss': data['BEC_Loss_Ratio'].mean(),
    'notification': data['Notification_Ratio'].mean(),
}

# Save the model, feature names, and average ratios to a file
with open('random_forest_total_cost_model.pkl', 'wb') as model_file:
    pickle.dump({
        'model': random_search,
        'feature_names': X_train.columns.tolist(),
        'ratios': average_ratios  # Ensure that ratios are saved
    }, model_file)

# Save the valid categories for input validation in the API
with open('valid_categories.pkl', 'wb') as f:
    pickle.dump(valid_categories, f)
