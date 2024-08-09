import pandas as pd

# Load the dataset with engineered features
data = pd.read_csv('features_data.csv')

# Selecting relevant features based on correlation or EDA results
selected_features = data[['AT', 'V', 'AP', 'RH', 'AT_V', 'AT_AP', 'V_RH', 'AP_RH', 'PE_Class']]
selected_features.to_csv('selected_features_data.csv', index=False)
print("Feature selection completed and saved to 'selected_features_data.csv'.")
