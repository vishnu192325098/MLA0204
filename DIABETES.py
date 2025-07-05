import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Step 1: Create the dataset
data = {
    'GLUCOSE': [130, 85, 155, 99, 140],
    'BMI': [33.6, 26.1, 35.2, 28.4, 32.7],
    'AGE': [45, 31, 50, 29, 41],
    'DIABETES': [1, 0, 1, 0, 1]
}

df = pd.DataFrame(data)

# Step 2: Split into features (X) and target (y)
X = df[['GLUCOSE', 'BMI', 'AGE']]
y = df['DIABETES']

# Step 3: Train/Test split (since data is small, we'll use all for training)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Create and train the KNN model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Step 5: Make a prediction
y_pred = knn.predict(X_test)

# Step 6: Show results
print("Test Data:")
print(X_test)
print("\nPredicted Diabetes Status:", y_pred.tolist())
print("Actual Diabetes Status   :", y_test.tolist())

# Step 7: Accuracy (optional)
acc = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {acc * 100:.2f}%")
