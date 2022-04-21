from sklearn.model_selection import RandomizedSearchCV
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from pprint import pprint


iris = datasets.load_iris()

x = iris.data
y = iris.target

param_grid = {
    'n_estimators': list(range(1, 100)),
    'max_features': ['auto', 'sqrt', 'log2'],
    'min_samples_leaf': list(range(1, 20)),
}


rf_model = RandomForestClassifier(random_state=1)

grid_search = RandomizedSearchCV(rf_model, param_grid, cv=5, n_iter = 100)

model = grid_search.fit(x, y)

pprint(model.best_estimator_.get_params())

predictions = model.predict(x)

print(predictions)
print(model.score(x, y))