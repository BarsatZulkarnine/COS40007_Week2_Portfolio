import pandas as pd

data = pd.read_csv('features_data.csv')

# selected features for correlation 
selected_features = ['AT', 'V', 'AP', 'RH', 'AT_V', 'AT_AP', 'V_RH', 'AP_RH', 'PE_Class']

selected_data = data[selected_features]

# Save the new dataset 
selected_data.to_csv('selected_features_data.csv', index=False)
print("Feature selection completed and saved to 'selected_features_data.csv'.")
print(selected_data.head())
