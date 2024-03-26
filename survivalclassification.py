import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(0)

# Generate synthetic data
n_samples = 1000

# Age: randomly generated between 18 and 80
age = np.random.randint(18, 81, size=n_samples)

# Gender: randomly generated with equal probability
gender = np.random.choice(['male', 'female'], size=n_samples)

# Swimming Ability: randomly generated with a higher probability for 'yes'
swimming_ability = np.random.choice(['yes', 'no'], size=n_samples, p=[0.7, 0.3])

# Will Power: randomly generated on a scale from 0 to 10
will_power = np.random.randint(0, 11, size=n_samples)

# Health Condition: randomly generated on a scale from 0 to 10
health_condition = np.random.randint(0, 11, size=n_samples)

# Location in Building: randomly generated floor number
location_in_building = np.random.randint(1, 11, size=n_samples)

# Time to Rescue: randomly generated time in minutes
time_to_rescue = np.random.randint(5, 301, size=n_samples)

# Injury Severity: randomly generated on a scale from 0 to 10
injury_severity = np.random.randint(0, 11, size=n_samples)

# Create DataFrame
data = pd.DataFrame({
    'Age': age,
    'Gender': gender,
    'Swimming_Ability': swimming_ability,
    'Will_Power': will_power,
    'Health_Condition': health_condition,
    'Location_in_Building': location_in_building,
    'Time_to_Rescue': time_to_rescue,
    'Injury_Severity': injury_severity
})

# Display the first few rows of the dataset
print(data.head())

import seaborn as sns
import matplotlib.pyplot as plt

# EDA
# Visualize distribution of features
sns.histplot(data['Age'], bins=20, kde=True)
plt.title('Distribution of Age')
plt.show()

sns.countplot(data['Gender'])
plt.title('Distribution of Gender')
plt.show()

sns.countplot(data['Swimming_Ability'])
plt.title('Distribution of Swimming Ability')
plt.show()

# Explore relationship between features and survival
sns.boxplot(x='Survived', y='Age', data=data)
plt.title('Age Distribution by Survival')
plt.show()

sns.boxplot(x='Survived', y='Will_Power', data=data)
plt.title('Will Power Distribution by Survival')
plt.show()

# Feature Engineering
# Convert categorical features into numerical representations
data['Gender'] = data['Gender'].map({'male': 0, 'female': 1})
data['Swimming_Ability'] = data['Swimming_Ability'].map({'no': 0, 'yes': 1})

# Model Building
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Define features and target variable
X = data.drop('Survived', axis=1)
y = data['Survived']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train Random Forest Classifier
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Predict on test set
y_pred = rf_model.predict(X_test)

# Model Evaluation
# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Visualize feature importance
feature_importance = pd.Series(rf_model.feature_importances_, index=X.columns)
feature_importance.nlargest(5).plot(kind='barh')
plt.title('Top 5 Important Features')
plt.xlabel('Feature Importance')
plt.show()

from pdpbox import pdp, info_plots

# PDP for Age feature
pdp_age = pdp.pdp_isolate(model=rf_model, dataset=X_test, model_features=X.columns, feature='Age')
pdp.pdp_plot(pdp_age, 'Age')
plt.xlabel('Age')
plt.ylabel('Predicted Probability of Survival')
plt.title('Partial Dependence Plot for Age')
plt.show()

from pdpbox import pdp, info_plots

import shap

# Initialize the SHAP explainer
explainer = shap.TreeExplainer(rf_model)

# Calculate SHAP values for a sample of data
sample_idx = 0  # Choose a sample index for explanation
shap_values = explainer.shap_values(X_test.iloc[sample_idx])

# Visualize the SHAP values
shap.force_plot(explainer.expected_value[1], shap_values[1], X_test.iloc[sample_idx])
plt.title('SHAP Values for Survival Prediction')
plt.show()

from sklearn.calibration import calibration_curve

# Generate calibration plot
prob_true, prob_pred = calibration_curve(y_test, rf_model.predict_proba(X_test)[:, 1], n_bins=10)
plt.plot(prob_pred, prob_true, marker='o', linestyle='-')
plt.xlabel('Mean Predicted Probability')
plt.ylabel('Fraction of Positives')
plt.title('Calibration Plot')
plt.show()

from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Initialize base estimators
estimators = [
    ('rf', RandomForestClassifier()),
    ('svm', SVC(probability=True))
]

# Initialize stacking classifier
stacking_model = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())

# Train stacking model
stacking_model.fit(X_train, y_train)

# Predict on test set
stacking_y_pred = stacking_model.predict(X_test)

# Evaluate stacking model
stacking_accuracy = accuracy_score(y_test, stacking_y_pred)
print("Accuracy of Stacking Classifier:", stacking_accuracy)