import joblib
import pandas as pd
# 1. Load the fitted pipeline
pipeline_path = "full_credit_pipeline.joblib"

loaded_pipeline = joblib.load(pipeline_path)
print("\n--- Loaded Fitted Pipeline for Inference ---")

# 2. Load the test data
test_data_path = "data/test_data.csv"
test_data = pd.read_csv(test_data_path)

# 3. Make predictions
predictions = loaded_pipeline.predict(test_data) # Scoring new customer data
probabilities = loaded_pipeline.predict_proba(test_data)[:, 1] # Prob of positive class

# Add predictions back to the raw data for inspection
test_data.loc[:, 'Prediction'] = predictions
test_data.loc[:, 'Probability_Default'] = probabilities

# 4. Save the predictions to a CSV file
output_path = "data/test_data_predictions.csv"