from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target

# The columns:
#     sepal length, sepal width, petal length, and petal width: numerical features measured in centimeters.
#     species: a label — it's 0, 1, or 2, representing three iris flower types:
#         0 = Iris setosa
#         1 = Iris versicolor
#         2 = Iris virginica
# The rows: are individual flower samples. So row 0 is one iris flower with its own measurements.

print(df.head())

X = df.drop(columns=['species'])
y = df['species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🌲 RandomForestClassifier
# 🔧 How it works:
#     It builds many decision trees (hence “forest”).
#     Each tree makes a prediction.
#     The forest votes (majority wins) to decide the final class.
#
# 📚 Concept:
#     It’s an ensemble method: combining multiple weak models (trees) into a strong one.
#     Trees are trained on random subsets of the data and features (bagging).
#
# ✅ Pros:
#     Works well on most datasets.
#     Handles both linear and nonlinear patterns.
#     Good at ignoring noise (due to averaging).
#     Can measure feature importance.
#
# ❌ Cons:
#     Slower to train (many trees).
#     Less interpretable than simple models.

# model = RandomForestClassifier()

# 📍 KNeighborsClassifier (KNN)
# 🔧 How it works:
#     It stores all the training data.
#     To predict a new sample:
#         It looks at the k closest points (neighbors).
#         It uses majority vote to classify.
#
# 📚 Concept:
#     It’s a lazy learner: it doesn’t train a model, it just memorizes.
#     It uses distance (e.g. Euclidean) to compare new data points to the old ones.
#
# ✅ Pros:
#     Very simple and intuitive.
#     No model training needed.
#     Works well with small datasets.
#
# ❌ Cons:
#     Slow for large datasets (must compare every new point to all others).
#     Sensitive to irrelevant features or scaling.
#     Needs a good choice of k.

model = KNeighborsClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

# Features (X)	Sepal & petal lengths/widths
# Labels (y)	Iris species (0, 1, 2)
# Train/test split	Keep some data hidden from the model until testing
# Model training	The model “learns” from the patterns
# Prediction	The model guesses the species of new flowers
# Accuracy	% of correct guesses in the test set

print(f"Accuracy: {acc * 100:.2f}%")