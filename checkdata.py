import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# read data
data = pd.read_excel('./Folds5x2_pp.xlsx')
# Check Data frame info
print(data.info())
# Remove duplicates
data = data.drop_duplicates()

# Check for missing values
print("Missing values in each column:")
print(data.isnull().sum())
# Check for duplicates
print(data.duplicated().sum())
# plot
plt.figure(figsize=(10, 8))
sns.boxplot(data=data)
plt.show()

# Save 
data.to_csv('cleaned_data.csv', index=False)
print("Data cleaning completed and saved to 'cleaned_data.csv'.")
