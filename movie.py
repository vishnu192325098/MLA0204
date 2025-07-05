import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Step 1: Sample dataset
data = {
    'Age': [18, 25, 30, 22, 35, 40],
    'Gender': ['Male', 'Female', 'Female', 'Male', 'Female', 'Male'],
    'FavoriteGenre': ['Action', 'Romance', 'Drama', 'Action', 'Comedy', 'Romance'],
    'WatchHours': [10, 5, 6, 12, 4, 3],
    'LikedMovie': [1, 0, 1, 1, 0, 0]
}

df = pd.DataFrame(data)

# Step 2: Encode categorical features
le_gender = LabelEncoder()
le_genre = LabelEncoder()

df['Gender'] = le_gender.fit_transform(df['Gender'])        # Male = 1, Female = 0
df['FavoriteGenre'] = le_genre.fit_transform(df['FavoriteGenre'])  # Genre encoded numerically

# Step 3: Split into features and target
X = df[['Age', 'Gender', 'FavoriteGenre', 'WatchHours']]
y = df['LikedMovie']

# Step 4: Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train KNN model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Step 6: Predict
y_pred = knn.predict(X_test)

# Step 7: Results
print("Test Data:\n", X_test)
print("\nPredicted Movie Preferences:", y_pred.tolist())
print("Actual Movie Preferences   :", y_test.tolist())
print(f"\nAccuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
