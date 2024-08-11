import pandas as pd
from sklearn.tree import DecisionTreeClassifier  
from sklearn.model_selection import train_test_split  
from sklearn import metrics 

data = pd.read_csv('cleaned_data.csv')
normalized_data = pd.read_csv('normalized_data.csv')
features_data = pd.read_csv('features_data.csv')
selected_features_data = pd.read_csv('selected_features_data.csv')

feature_sets = {
    "Model 1": data[['AT', 'V', 'AP', 'RH']],  # All features without normalization and without composite features
    "Model 2": normalized_data[['AT', 'V', 'AP', 'RH', 'PE_Class']],  # All features with normalization and without composite features
    "Model 3": features_data[['AT', 'V', 'AP', 'RH', 'AT_V', 'AT_AP', 'V_RH', 'AP_RH']],  # All features with normalization and containing composite features
    "Model 4": selected_features_data[['AT', 'V', 'AP', 'RH', 'AT_V', 'AT_AP', 'V_RH', 'AP_RH']],  # Selected features with normalization
    "Model 5": selected_features_data[['AT', 'V', 'AP', 'RH']]  # Selected features without normalization
}

results = {}

# Train
for model_name, X in feature_sets.items():
    y = selected_features_data['PE_Class'] 

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) 
    
    clf = DecisionTreeClassifier()
    
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    
    accuracy = metrics.accuracy_score(y_test, y_pred)
    results[model_name] = accuracy

for model_name, accuracy in results.items():
    print(f"{model_name} Accuracy: {accuracy}")

# Save 
comparison_df = pd.DataFrame(list(results.items()), columns=['Model', 'Accuracy'])
comparison_df.to_csv('model_comparison.csv', index=False)
print("Model development completed and accuracies saved to 'model_comparison.csv'.")
