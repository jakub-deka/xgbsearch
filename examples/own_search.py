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
