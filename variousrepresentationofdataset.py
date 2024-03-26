import pandas as pd

# Load the dataset
titanic_data = pd.read_csv("titanic.csv")

# Display the first few rows of the dataset to understand its structure
print(titanic_data.head())

#Data Cleaning

# Fill missing values in 'Age' column with median age
titanic_data['Age'].fillna(titanic_data['Age'].median(), inplace=True)

# Fill missing values in 'Embarked' column with mode
titanic_data['Embarked'].fillna(titanic_data['Embarked'].mode()[0], inplace=True)

# Create 'FamilySize' feature
titanic_data['FamilySize'] = titanic_data['SibSp'] + titanic_data['Parch'] + 1

# Perform one-hot encoding for 'Sex' and 'Embarked' columns
titanic_data = pd.get_dummies(titanic_data, columns=['Sex', 'Embarked'], drop_first=True)

import seaborn as sns
import matplotlib.pyplot as plt

# Visualize survival rate by gender
sns.barplot(x='Sex_male', y='Survived', data=titanic_data)
plt.title('Survival Rate by Gender')
plt.xlabel('Gender (Male)')
plt.ylabel('Survival Rate')
plt.show()


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Split the data into features and target variable
X = titanic_data.drop(['Survived', 'PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
y = titanic_data['Survived']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict the labels for the test set
y_pred = model.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy of Logistic Regression Model:", accuracy)

from sklearn.preprocessing import MinMaxScaler

# Initialize MinMaxScaler
scaler = MinMaxScaler()

# Scale numerical features
X[['Age', 'Fare', 'FamilySize']] = scaler.fit_transform(X[['Age', 'Fare', 'FamilySize']])

from sklearn.feature_selection import RFE

# Initialize logistic regression model for feature selection
logreg = LogisticRegression()

# Initialize RFE
rfe = RFE(logreg, n_features_to_select=5)

# Fit RFE
rfe.fit(X, y)

# Get selected features
selected_features = X.columns[rfe.support_]
print("Selected Features:", selected_features)

from sklearn.model_selection import cross_val_score

# Perform cross-validation
cv_scores = cross_val_score(model, X, y, cv=5)

# Calculate mean cross-validation score
mean_cv_score = cv_scores.mean()
print("Mean Cross-Validation Score:", mean_cv_score)

from sklearn.model_selection import GridSearchCV

# Define hyperparameters grid
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}

# Initialize GridSearchCV
grid_search = GridSearchCV(model, param_grid, cv=5)

# Fit GridSearchCV
grid_search.fit(X, y)

# Get best hyperparameters
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

from sklearn.ensemble import RandomForestClassifier

# Initialize Random Forest Classifier
rf_model = RandomForestClassifier()

# Train the model
rf_model.fit(X_train, y_train)

# Predict the labels for the test set
rf_y_pred = rf_model.predict(X_test)

# Calculate the accuracy of the model
rf_accuracy = accuracy_score(y_test, rf_y_pred)
print("Accuracy of Random Forest Classifier:", rf_accuracy)
