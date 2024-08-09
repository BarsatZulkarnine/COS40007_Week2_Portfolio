import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load the datasets
data = pd.read_csv('normalized_data.csv')
features_data = pd.read_csv('features_data.csv')
selected_features_data = pd.read_csv('selected_features_data.csv')

# Define feature sets
feature_sets = {
    "Model 1": data[['AT', 'V', 'AP', 'RH']],  # all features without normalization and without composite features
    "Model 2": data[['AT', 'V', 'AP', 'RH', 'PE_Class']],  # all features with normalization and without composite features
    "Model 3": features_data[['AT', 'V', 'AP', 'RH', 'AT_V', 'AT_AP', 'V_RH', 'AP_RH']],  # all features with normalization and containing composite features
    "Model 4": selected_features_data[['AT', 'V', 'AP', 'RH', 'AT_V', 'AT_AP', 'V_RH', 'AP_RH']],  # selected features with normalization
    "Model 5": selected_features_data[['AT', 'V', 'AP', 'RH']]  # selected feature without normalization
}

# Initialize results dictionary
results = {}

# Train and evaluate models
for model_name, features in feature_sets.items():
    X = features
    y = selected_features_data['PE_Class']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
    
    # Initialize the model
    clf = DecisionTreeClassifier()
    
    # Train the model
    clf.fit(X_train, y_train)
    
    # Predict
    y_pred = clf.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    results[model_name] = accuracy

# Print results
for model_name, accuracy in results.items():
    print(f"{model_name} accuracy: {accuracy}")

# Save results for comparison
comparison_df = pd.DataFrame(list(results.items()), columns=['Model', 'Accuracy'])
comparison_df.to_csv('model_comparison.csv', index=False)
print("Model development completed and accuracies saved to 'model_comparison.csv'.")
