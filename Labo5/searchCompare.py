from sklearn.model_selection import GridSearchCV
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from pprint import pprint


iris = datasets.load_iris()

x = iris.data
y = iris.target

param_grid = {
    'max_iter': [x for x in range(0,50,1)],
    'solver': ['newton-cg', 'lbfgs', 'liblinear'],
    'C': [x for x in range(0,30,1)]
}

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression( solver='lbfgs', multi_class='auto', random_state=0 )

# print(log_reg.get_params().keys())

grid_search = GridSearchCV(estimator=log_reg, param_grid=param_grid, n_jobs=-1, cv=5, scoring='accuracy',error_score=0)

grid_result = grid_search.fit(X_train, y_train)

pprint(grid_result.best_estimator_.get_params())

train_score = grid_result.score( X_test, y_test )
print("H Score: {}".format(train_score))

log_result = log_reg.fit(X_train, y_train)

train_score = log_result.score( X_test, y_test )
print("N Score: {}".format(train_score))