import numpy as np
from collections import Counter

# euclidean_distance(p1, p2):
# Calculates the distance between two points using the Euclidean formula:
# distance=(x1−x2)2+(y1−y2)2
def euclidean_distance(p1, p2):
    return np.sqrt(np.sum(np.array(p1) - np.array(p2))**2)

def knn_predict(X_train, y_train, x_test, k=3):
    distances = []

    for i in range(len(X_train)):
        dist = euclidean_distance(X_train[i], x_test)
        distances.append((dist, y_train[i]))

    distances.sort(key=lambda x: x[0])

    k_labels = [label for (_, label) in distances[:k]]

    prediction = Counter(k_labels).most_common(1)[0][0]
    return prediction

# sample
X_train = [[1, 2], [2, 3], [3, 1], [6, 5], [7, 7]]
y_train = ['A', 'A', 'A', 'B', 'B']
x_test = [5, 5]

print(f"Predict class", knn_predict(X_train, y_train, x_test, k=3))