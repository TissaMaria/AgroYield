import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import re

crop_data = pd.read_csv(r"C:\Users\Tissa Maria\OneDrive\Desktop\Crop-Yield-Prediction-in-India-using-ML\crop_production.csv")

crop_data['Crop'] = crop_data['Crop'].apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', str(x).lower()))  # Ensure 'x' is a string
crop_data['Crop'] = crop_data['Crop'].apply(lambda x: x.split('/')[0].strip())  # Remove anything after '/' and strip whitespace

crop_recommendation_data = pd.read_csv(r"C:\Users\Tissa Maria\OneDrive\Desktop\Crop-Yield-Prediction-in-India-using-ML\Crop_recommendation.csv")

crop_recommendation_data['label'] = crop_recommendation_data['label'].apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', str(x).lower()))  # Ensure 'x' is a string
crop_recommendation_data['label'] = crop_recommendation_data['label'].apply(lambda x: x.split('/')[0].strip())  # Remove anything after '/' and strip whitespace

merged_data = pd.merge(
    crop_data, 
    crop_recommendation_data, 
    left_on='Crop', 
    right_on='label', 
    how='inner'  
)

merged_data = merged_data.drop(columns=['label'])

merged_data.dropna(inplace=True)
merged_data.drop_duplicates(inplace=True)

merged_data['Yield'] = merged_data['Production'] / merged_data['Area']

final_data = merged_data[['District_Name', 'Season', 'Crop', 
                          'Area', 'Yield', 'N', 'P', 'K', 
                          'temperature', 'humidity', 'ph', 'rainfall']].copy()  # Use .copy() to avoid warning

final_data['District_Name'] = final_data['District_Name'].str.lower()
final_data['Season'] = final_data['Season'].str.lower()
final_data['Crop'] = final_data['Crop'].str.lower()

final_data.dropna(inplace=True) 

# Adjust
sample_size = 20000 
if final_data.shape[0] > sample_size:
    reduced_data = final_data.sample(n=sample_size, random_state=42)
else:
    reduced_data = final_data

reduced_data.to_csv('reduced_crop_data.csv', index=False)

dummy_data = pd.get_dummies(reduced_data, drop_first=True)

X = dummy_data.drop(['Yield'], axis=1)
y = dummy_data['Yield']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Random Forest Regressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the trained model to a file
with open('model/crop_yield_rf_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("Random Forest model trained and saved successfully!")

# Performance
y_pred = model.predict(X_test)

# Calculate metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Model Performance on Test Data:")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R²): {r2}")

# Prediction function
def predict_yield(input_data):
   
    with open('model/crop_yield_rf_model.pkl', 'rb') as f:
        model = pickle.load(f)

    input_df = pd.DataFrame([input_data])

    input_df['District_Name'] = input_df['District_Name'].str.lower()
    input_df['Season'] = input_df['Season'].str.lower()
    input_df['Crop'] = input_df['Crop'].str.lower()

    input_df = pd.get_dummies(input_df)

  
    input_df = input_df.reindex(columns=X.columns, fill_value=0)

    prediction = model.predict(input_df)[0]
    return round(prediction, 2)

test_input = {
    'District_Name': 'nicobars',
    'Season': 'kharif',
    'Crop': 'rice',
    'Area': 102.0,
    'N': 90,
    'P': 42,
    'K': 43,
    'temperature': 20.87974371,
    'humidity': 82.00274423,
    'ph': 6.502985292000001,
    'rainfall': 202.9355362,
}

predicted_yield = predict_yield(test_input)
print(f"Predicted Yield: {predicted_yield}")
