from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
data = load_iris()
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=49
)
weak_learner = DecisionTreeClassifier(max_depth=1)
boost_model = AdaBoostClassifier(
    estimator=weak_learner, 
    n_estimators=50,
    learning_rate=1
)
boost_model.fit(X_train, y_train)
y_pred = boost_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Predicted Labels:", y_pred)
print("Actual Labels   :", y_test)
print("Boosting Accuracy:", accuracy)
