from sklearn.model_selection import GridSearchCV
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from pprint import pprint


iris = datasets.load_iris()

x = iris.data
y = iris.target

param_grid = {
    'n_estimators': [9,10, 11,15, 20,25, 40, 50],
    'max_features': ['auto', 'sqrt', 'log2'],
    'min_samples_leaf': [1, 5, 10, 15],
}

rf_model = RandomForestClassifier(random_state=1)

grid_search = GridSearchCV(rf_model, param_grid, cv=5)

model = grid_search.fit(x, y)

# pprint(model.best_estimator_.get_params())

predictions = model.predict(x)

# print(predictions)

print("hyperparameter tuning:")
print(model.score(x, y))
