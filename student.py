import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Step 1: Define the dataset
data = {
    'MathScore': [45, 78, 55, 90, 40],
    'Attendance': [60, 85, 70, 95, 50],
    'StudyHours': [4, 10, 5, 12, 2],
    'AtRisk': [1, 0, 1, 0, 1]
}

df = pd.DataFrame(data)

# Step 2: Features and Target
X = df[['MathScore', 'Attendance', 'StudyHours']]
y = df['AtRisk']

# Step 3: Train/test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train the KNN model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Step 5: Predict on test data
y_pred = knn.predict(X_test)

# Step 6: Results
print("Test Data:\n", X_test)
print("\nPredicted Risk Status:", y_pred.tolist())
print("Actual Risk Status   :", y_test.tolist())

# Step 7: Accuracy
acc = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {acc * 100:.2f}%")
