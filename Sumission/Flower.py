# Step 1: Load and Explore the Dataset
from sklearn.datasets import load_iris
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib

# Load the dataset
iris = load_iris()
data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
data['species'] = iris.target

# Display the first few rows
print("First few rows of the dataset:")
print(data.head())

# Generate basic statistics
print("\nBasic statistics:")
print(data.describe())

# Check for missing values
print("\nMissing values per column:")
print(data.isnull().sum())

# Step 2: Data Preprocessing
X = data.iloc[:, :-1]  # Features
y = data['species']    # Target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("\nTraining and testing split completed.")

# Step 3: Build the Machine Learning Model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.2f}")

# Step 4: Visualize Results
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Step 5: Save and Load the Model
joblib.dump(model, "iris_model.pkl")
print("\nModel saved to 'iris_model.pkl'.")

# Load the model
loaded_model = joblib.load("iris_model.pkl")
print("\nModel loaded successfully.")

# Perform predictions with the loaded model
new_predictions = loaded_model.predict(X_test)
print("\nPredictions on test data using loaded model:")
print(new_predictions)
