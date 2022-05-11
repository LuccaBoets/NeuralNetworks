import random
from sklearn.model_selection import RandomizedSearchCV
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from pprint import pprint

iris = datasets.load_iris()

x = iris.data
y = iris.target

param_grid = {
    'n_estimators': [random.randint(1, 30) for i in range(10)],
    'max_features': ['auto', 'sqrt', 'log2'],
    'min_samples_leaf': [random.randint(1, 20) for i in range(10)],
}


rf_model = RandomForestClassifier(random_state=1)

random_search = RandomizedSearchCV(rf_model, param_grid, cv=5, n_iter = 30)

model = random_search.fit(x, y)

pprint(model.best_estimator_.get_params())

predictions = model.predict(x)

pprint(predictions)
pprint(model.score(x, y))