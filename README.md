# xgbsearch

A pacakge implementing grid search and random search for XGBoost.

## Description

An implementation of grid search and random search for hyper parameter tunning for `xgboost` that allows for specification of `eval_sets` and usage of early stopping.

## Installation

Install it via PyPI using `pip` command.

```bash
# Install or Upgrade to newest available version
$ pip install -U xgbsearch
```

## Motivation

Current implementation of grid search and random search in scikit-learn (using GridSearchCV/RandomSearchCV) does not allow for specifying evaluation sets that can be easily passed to `XGboost.fit()` and used in early stopping.

This package significantly simplifies implementation of grid search and random search.

Additionally, it makes implementation of your own, new search algorithms very simple and provides a consistent interface for running various hyper parameter tunning algorightms while retaining early stopping and eval sets functionality provided by `xgboost`.

## Example usage

The package provides 3 classes:

* `XgbSearch` - the base super class that contains most of the implementation details.
* `XgbGridSearch` - implementation of grid search (exchaustive search) algorithm for hyper parameter tunning.
* `XgbRandomSearch` - implementation of random search algorithm for hyper parameter tunning.

```python
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

```

## Output

After fitting, `XgbGridSearch` and `XgbRandomSearch` will populate a `result` list. This will contain one dict per iteration with complete result of the model run.

```python
{'model': <xgboost.core.Booster at 0x7f3b00bcd070>, # actual model object, if early stopping is enabled this will be the LAST model fitted, not the best one
 'parameters': {'eta': 0.05,'colsample_bytree': 0.8,'max_depth': 11,'min_child_weight': 3,'device': 'cuda','objective': 'binary:logistic','eval_metric': ['auc']}, # Parameters passed to xgb.fit
 'additional_settings': {'num_boost_round': 10000,'verbose_eval': 10,'early_stopping_rounds': 100}, # additional settings passed to xgb.fit
 # Results of the model fitting process by boosting round
 'model_training_results': {'TRAIN': OrderedDict([('auc',
                [0.8943848819579814,
                 ...
                 0.9805122373835824,
                 0.9805555555555555,
                 0.9806530214424951])]),
  'train': OrderedDict([('auc',
                [0.8943848819579814,
                 ...
                 0.9805122373835824,
                 0.9805555555555555,
                 0.9806530214424951])]),
  'test': OrderedDict([('auc',
                [0.8711075249536788,
                 ...
                 0.8837786145478453,
                 0.8839579224194609,
                 0.8836590759667683])])},
 'best_iteration': 204, # index of the best iteration, this is significant when using early_stopping
 'best_score': 0.8865877712031558, # value of last eval_metric on last eval_set for the best model
 'best_model': <xgboost.core.Booster at 0x7f3b01a75a30>} # actual best model object
 ```

 ## Implementing your own search

 It is very easy to implement your own search using XgbSearch class.

```python
from xgbsearch import XgbSearch

class MyOwnSearch(XgbSearch):

    def __init__(self, tune_params, fit_params, add_value, maximise_score=True):
        super().__init__(tune_params, fit_params, maximise_score)
        self.add_value = add_value

    def _generate_params(self):
        # Toy example. Will just take the first parameter and add to it a value specified in the constructor.
        # this method needs to return a list dicts of parameters that will be passed into xgb.fit()
        result = []
        for i in range(3):
            first_key = list(self.tune_params.keys())[0]
            loop_result = self.tune_params | self.fit_params
            loop_result[first_key] = loop_result[first_key] + i * self.add_value
            result.append(loop_result)

        return result


# Run it!
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
    "eta": 0.01,
    "max_depth": 5,
    "min_child_weight": 3,
}

my_search = MyOwnSearch(tune_params_grid, fit_params, 0.01)
eval_set = [(X_train, y_train, "train"), (X_test, y_test, "test")]
my_search.fit(X_train, y_train, eval_set, 10000, 100, verbose_eval=25)
```