import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score, confusion_matrix, roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# Display the first 5 rows of the datasets
print("First 5 rows of the training dataset:")
print(train.head())
print("\nFirst 5 rows of the test dataset:")
print(test.head())

# Preprocessing missing values
# Fill missing 'Embarked' and 'Fare' with the most common value (mode) or median
test['Embarked'] = test['Embarked'].fillna(test['Embarked'].mode()[0])
test['Fare'] = test['Fare'].fillna(test['Fare'].median())
train['Age'] = train['Age'].fillna(train['Age'].median())
test['Age'] = test['Age'].fillna(test['Age'].median())

# Convert 'Sex' and 'Embarked' into numeric values
train['Sex'] = train['Sex'].map({'male': 0, 'female': 1})
test['Sex'] = test['Sex'].map({'male': 0, 'female': 1})

train['Embarked'] = train['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})
test['Embarked'] = test['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

# Selecting features (X) and target (y)
X = train[['Pclass', 'Sex', 'Age', 'Fare']]
y = train['Survived']

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression Model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred_linear = linear_model.predict(X_val)

# Convert linear regression predictions to binary values (0 or 1)
y_pred_linear_binary = (y_pred_linear > 0.5).astype(int)

# Calculate RMSE (Root Mean Squared Error) for Linear Regression
rmse_linear = np.sqrt(mean_squared_error(y_val, y_pred_linear))
print(f"Linear Regression RMSE: {rmse_linear}")

# Logistic Regression Model
logistic_model = LogisticRegression(max_iter=200)
logistic_model.fit(X_train, y_train)
y_pred_logistic = logistic_model.predict(X_val)

# Calculate Accuracy and F1 Score for Logistic Regression
accuracy_logistic = accuracy_score(y_val, y_pred_logistic)
f1_logistic = f1_score(y_val, y_pred_logistic)
print(f"Logistic Regression Accuracy: {accuracy_logistic * 100:.2f}%")
print(f"Logistic Regression F1 Score: {f1_logistic * 100:.2f}%")

# Calculate Accuracy for Linear Regression
accuracy_linear = accuracy_score(y_val, y_pred_linear_binary)
print(f"Linear Regression Accuracy: {accuracy_linear * 100:.2f}%")

# Show comparison of accuracy between both models
print("\nComparison of Accuracy:")
print(f"Logistic Regression Accuracy: {accuracy_logistic * 100:.2f}%")
print(f"Linear Regression Accuracy: {accuracy_linear * 100:.2f}%")

# Confusion Matrix for Logistic Regression
conf_matrix = confusion_matrix(y_val, y_pred_logistic)
plt.figure(figsize=(6, 6))
plt.imshow(conf_matrix, cmap="Blues")
plt.title("Confusion Matrix for Logistic Regression")
plt.colorbar()
plt.xticks([0, 1], labels=["Not Survived", "Survived"])
plt.yticks([0, 1], labels=["Not Survived", "Survived"])
plt.xlabel("Predicted")
plt.ylabel("Actual")

# Show values in the confusion matrix
for i in range(2):
    for j in range(2):
        plt.text(j, i, conf_matrix[i, j], ha='center', va='center', color='red')

plt.show()

# ROC Curve for Logistic Regression
fpr, tpr, thresholds = roc_curve(y_val, y_pred_logistic)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend(loc="lower right")
plt.show()

# Scatter plot for Linear Regression Predictions
plt.scatter(y_val, y_pred_linear_binary, alpha=0.5)
plt.title("Linear Regression Predictions vs Actual")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.show()
