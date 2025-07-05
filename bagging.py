import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from collections import Counter
data = load_iris()
X = data.data
y = data.target
n_estimators = 10
n_samples = len(X)
models = []
for _ in range(n_estimators):
    indices = np.random.choice(n_samples, n_samples, replace=True)
    X_sample = X[indices]
    y_sample = y[indices]
    model = DecisionTreeClassifier()
    model.fit(X_sample, y_sample)
    models.append(model)
def bagging_predict(models, X_test):
    predictions = []
    for x in X_test:
        preds = [model.predict([x])[0] for model in models]
        final_pred = Counter(preds).most_common(1)[0][0]
        predictions.append(final_pred)
    return np.array(predictions)
y_pred = bagging_predict(models, X)
print("Accuracy (manual bagging):", accuracy_score(y, y_pred))
