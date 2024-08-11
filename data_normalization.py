import pandas as pd
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv('./cleaned_data.csv')

bins = [420, 435, 450, 465, 480, 496]
labels = [1, 2, 3, 4, 5]
data['PE_Class'] = pd.cut(data['PE'], bins=bins, labels=labels, right=False)

# Normalization
scaler = MinMaxScaler()
data[['AT', 'V', 'AP', 'RH']] = scaler.fit_transform(data[['AT', 'V', 'AP', 'RH']])

# Save 
print(data.head())
data.to_csv('normalized_data.csv', index=False)
print("Class labeling and normalization completed and saved to 'normalized_data.csv'.")
