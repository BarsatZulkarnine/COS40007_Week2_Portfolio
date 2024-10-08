import pandas as pd

data = pd.read_csv('./normalized_data.csv')


data['AT_V'] = data['AT'] * data['V']
data['AT_AP'] = data['AT'] * data['AP']
data['V_RH'] = data['V'] * data['RH']
data['AP_RH'] = data['AP'] * data['RH']

# Save 
data.to_csv('features_data.csv', index=False)
print(data.head())
print("Feature engineering completed and saved to 'features_data.csv'.")
