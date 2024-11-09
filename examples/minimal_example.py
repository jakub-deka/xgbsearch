from xgbsearch import XgbGridSearch, XgbRandomSearch
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import roc_auc_score

X, y = make_classification(random_state=42)
X = pd.DataFrame(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# These parameters will be passed to xgb.fit as is.
fit_params = {
    "device": "cuda",
    "objective": "binary:logistic",
    "eval_metric": ["auc"],
}

# The parameters here will be tuned. If the parameter is a single value, it will be passed as is.
# If the parameter is a list, all possible combinations will be searched using grid search.
tune_params_grid = {
    "eta": [0.01, 0.001],
    "max_depth": [5, 11],
    "min_child_weight": 3,
}

grid_search = XgbGridSearch(tune_params_grid, fit_params)
eval_set = [(X_train, y_train, "train"), (X_test, y_test, "test")]
grid_search.fit(X_train, y_train, eval_set, 10000, 100, verbose_eval=25)

# The parameters here will be tuned. If the parameter is a single value, it will be passed as is.
# If the parameter is a list, during each iteration a single value will be picked from that list.
# If the parameter is a tuple of two floats, a random value between the two ends will be picked.
# If the parameter is a tuple of two ints, a random int value between the two ends will be picked.
tune_params_random = {
    "eta": (0.1, 0.005),
    "max_depth": (5, 11),
    "min_child_weight": [1, 2, 3],
}

random_search = XgbRandomSearch(tune_params_random, fit_params, max_iter_count=3)
eval_set = [(X_train, y_train, "train"), (X_test, y_test, "test")]
random_search.fit(X_train, y_train, eval_set, 10000, 100, verbose_eval=25)

# You can access the results like this.
print(random_search.get_best_model())  # returns the best model object
print(
    random_search.get_best_model_results()
)  # returns best model results dict with complete results
print(random_search.predict(X_test))  # generates predictions for the BEST model
print(
    random_search.score(X_test, y_test, roc_auc_score)
)  # calculates the score using given function; note the function needs to accept X, y as input
