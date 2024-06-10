import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load dataset
url = "https://www.kaggle.com/code/snorpiii/heart-disease-predic-machine-learning-naive-bayes"
columns = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
]
data = pd.read_csv(url, names=columns)

# Replace missing values marked as '?' with NaN
data.replace('?', np.nan, inplace=True)

# Convert necessary columns to numeric type
data = data.apply(pd.to_numeric)

# Fill missing values
data.fillna(data.mean(), inplace=True)

# Encode categorical variables
data['sex'] = data['sex'].map({0: 'female', 1: 'male'})
data = pd.get_dummies(data, columns=['sex'], drop_first=True)

# Define features and target
X = data.drop('target', axis=1)
y = data['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize models
log_reg = LogisticRegression(max_iter=1000)
tree_clf = DecisionTreeClassifier()
rf_clf = RandomForestClassifier(n_estimators=100)

# Train models
log_reg.fit(X_train, y_train)
tree_clf.fit(X_train, y_train)
rf_clf.fit(X_train, y_train)

# Predictions
y_pred_log_reg = log_reg.predict(X_test)
y_pred_tree_clf = tree_clf.predict(X_test)
y_pred_rf_clf = rf_clf.predict(X_test)

# Evaluate models
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_log_reg))
print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_tree_clf))
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf_clf))

# Confusion Matrix and Classification Report for the best model
best_model = rf_clf
y_pred_best = y_pred_rf_clf
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_best))
print("Classification Report:\n", classification_report(y_test, y_pred_best))

# Confusion matrix heatmap
conf_matrix = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()

# Save the best model
joblib.dump(best_model, 'heart_disease_model.pkl')

# Save the scaler
joblib.dump(scaler, 'scaler.pkl')

# Load the saved model and scaler
loaded_model = joblib.load('heart_disease_model.pkl')
loaded_scaler = joblib.load('scaler.pkl')

# Example new data
new_data = np.array([[63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]])
new_data_scaled = loaded_scaler.transform(new_data)

# Predict
prediction = loaded_model.predict(new_data_scaled)
print("Prediction (0: No Heart Disease, 1: Heart Disease):", prediction[0])
