from flask import Flask, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# Load the trained model and feature names
with open('random_forest_total_cost_model.pkl', 'rb') as model_file:
    model_data = pickle.load(model_file)
    total_cost_model = model_data['model']
    feature_names = model_data['feature_names']
    average_ratios = model_data['ratios']  # Load the ratios here

# Load valid categories (this should now include Brazil and others)
with open('valid_categories.pkl', 'rb') as f:
    valid_categories = pickle.load(f)

# Define the categorical columns that were one-hot encoded in the training data
categorical_columns = ['Industry Type', 'Country', 'Incident Type', 'Root Cause']

# Function to validate input
def validate_input(data):
    # Check for each categorical column if the input value exists in the predefined categories
    for col, valid_values in valid_categories.items():
        if col in data and data[col] not in valid_values:
            return False, f"Invalid {col}: {data[col]}. Valid options are: {valid_values}"
    return True, ""

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse the incoming JSON request
        data = request.json
        
        # Validate input data
        is_valid, error_message = validate_input(data)
        if not is_valid:
            return jsonify({"error": error_message}), 400

        # Convert the input into a DataFrame
        input_df = pd.DataFrame([data])
        
        # One-hot encode the categorical variables
        input_df = pd.get_dummies(input_df)
        
        # Ensure that all the necessary columns are present, even if they are not in the input
        for col in categorical_columns:
            for category in valid_categories[col]:
                encoded_col = f"{col}_{category}"
                if encoded_col not in input_df.columns:
                    input_df[encoded_col] = 0
        
        # Align the input DataFrame with the model's expected feature set
        input_df = input_df.reindex(columns=feature_names, fill_value=0)
        
        # Make prediction for total cost
        predicted_total_cost = total_cost_model.predict(input_df)[0]
        
        # Calculate secondary costs based on the predicted total cost and the average ratios
        secondary_costs = {
            'Forensics Cost (USD)': predicted_total_cost * average_ratios['forensics'],
            'PR Crisis Cost (USD)': predicted_total_cost * average_ratios['pr_crisis'],
            'IR Counsel Cost (USD)': predicted_total_cost * average_ratios['ir_counsel'],
            'Credit Monitoring Cost (USD)': predicted_total_cost * average_ratios['credit_monitoring'],
            'Ransom Payments (USD)': predicted_total_cost * average_ratios['ransom'],
            'First Party Costs (USD)': predicted_total_cost * average_ratios['first_party'],
            'BEC Loss (USD)': predicted_total_cost * average_ratios['bec_loss'],
            'Notification Call Centre (USD)': predicted_total_cost * average_ratios['notification']
        }
        
        # Return the total cost and the secondary costs in a clean format
        response = {
            'Total Costs (USD)': predicted_total_cost,
            'Secondary Costs': secondary_costs
        }

        return jsonify(response)
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
