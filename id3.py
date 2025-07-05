import pandas as pd
import numpy as np
from math import log2
data = {
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast',
                'Sunny', 'Sunny', 'Rain', 'Sunny', 'Overcast', 'Overcast', 'Rain'],
    'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool',
                    'Mild', 'Cool', 'Mild', 'Mild', 'Mild', 'Hot', 'Mild'],
    'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal',
                 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'High'],
    'Wind': ['Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong',
             'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Strong'],
    'PlayTennis': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes',
                   'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
}
df = pd.DataFrame(data)
def entropy(column):
    values, counts = np.unique(column, return_counts=True)
    total = np.sum(counts)
    ent = -np.sum([(count / total) * log2(count / total) for count in counts])
    return ent
def information_gain(data, attribute, target='PlayTennis'):
    total_entropy = entropy(data[target])
    values, counts = np.unique(data[attribute], return_counts=True)
    weighted_entropy = np.sum([(counts[i] / np.sum(counts)) *
        entropy(data[data[attribute] == values[i]][target]) for i in range(len(values))])
    return total_entropy - weighted_entropy
features = df.columns[:-1]
gains = {feature: information_gain(df, feature) for feature in features}
best_attribute = max(gains, key=gains.get)
print("Information Gain for each attribute:")
for feature, gain in gains.items():
    print(f"{feature}: {gain:.4f}")
print(f"\n✅ Best Attribute to split on (Root Node): {best_attribute}")
