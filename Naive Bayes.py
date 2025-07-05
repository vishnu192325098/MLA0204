import pandas as pd
import numpy as np
data = pd.DataFrame([
    ['Sunny', 'Hot', 'High', 'Weak', 'No'],
    ['Sunny', 'Hot', 'High', 'Strong', 'No'],
    ['Overcast', 'Hot', 'High', 'Weak', 'Yes'],
    ['Rain', 'Mild', 'High', 'Weak', 'Yes'],
    ['Rain', 'Cool', 'Normal', 'Weak', 'Yes'],
    ['Rain', 'Cool', 'Normal', 'Strong', 'No'],
    ['Overcast', 'Cool', 'Normal', 'Strong', 'Yes'],
    ['Sunny', 'Mild', 'High', 'Weak', 'No'],
    ['Sunny', 'Cool', 'Normal', 'Weak', 'Yes'],
    ['Rain', 'Mild', 'Normal', 'Weak', 'Yes'],
], columns=['Outlook', 'Temperature', 'Humidity', 'Wind', 'Play'])
features = data.iloc[:, :-1]
labels = data.iloc[:, -1]
classes = labels.unique()
priors = {}
for c in classes:
    priors[c] = sum(labels == c) / len(labels)
def calc_likelihoods(features, labels):
    likelihoods = {}
    for c in classes:
        class_data = features[labels == c]
        likelihoods[c] = {}
        for col in features.columns:
            likelihoods[c][col] = {}
            for val in features[col].unique():
                count = sum(class_data[col] == val)
                likelihoods[c][col][val] = (count + 1) / (len(class_data) + len(features[col].unique()))  # Laplace smoothing
    return likelihoods
likelihoods = calc_likelihoods(features, labels)
def predict(instance):
    posteriors = {}
    for c in classes:
        prob = priors[c]
        for col in features.columns:
            val = instance[col]
            prob *= likelihoods[c][col].get(val, 1e-6)
        posteriors[c] = prob
    return max(posteriors, key=posteriors.get)
test = {'Outlook': 'Sunny', 'Temperature': 'Cool', 'Humidity': 'High', 'Wind': 'Strong'}
result = predict(test)
print("Prediction (manual):", result)
