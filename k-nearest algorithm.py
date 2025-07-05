import numpy as np
from collections import Counter
X = [
    [170, 65],
    [160, 55],
    [175, 70],
    [165, 60],
    [155, 50]
]
y = ['A', 'B', 'A', 'A', 'B']
query_point = [162, 62]
def euclidean_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
distances = []
for i in range(len(X)):
    dist = euclidean_distance(query_point, X[i])
    distances.append((dist, y[i], X[i]))
distances.sort(key=lambda x: x[0])
print("Three Nearest Neighbors:")
for i in range(3):
    print(f"{i+1}. Distance = {distances[i][0]:.2f}, Point = {distances[i][2]}, Class = {distances[i][1]}")
k_nearest_classes = [distances[i][1] for i in range(3)]
prediction = Counter(k_nearest_classes).most_common(1)[0][0]
print(f"\nPredicted Class for query point {query_point}: {prediction}")
