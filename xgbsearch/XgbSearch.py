from typing import Dict, Any
import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.data import DMatrix
from colorama import Fore, Back, Style
import itertools
import random


class XgbSearch:
    def __init__(
        self,
        tune_params: Dict[
            str, float | int | str | tuple[float, float] | list[float | int | str]
        ],
        fit_params: Dict[str, str | int | float],
        maximise_score: bool = True,
    ):
        """Set up a XgbSearch object.

        Args:
            tune_params: Dictionary of parameters to be searched.
            fit_params: Dictionary of parameters that will be passed in additiona to the searched params into `xgb.fit` method.
            maximise_score: By default when getting the best model one with the highest score will be returned.

        Example:
        >>> # These parameters will be ALWAYS passed to `xgb.fit` method.
        >>> fit_params = {
        >>>     "device": "cuda",
        >>>     "objective": "binary:logistic",
        >>>     "eval_metric": ["aucpr", "auc"],
        >>> }
        >>> # These parameters will be tuned.
        >>> tune_params = {
        >>>     "eta": 0.01,
        >>>     "colsample_bytree": 0.8,
        >>>     "max_depth": 11,
        >>>     "min_child_weight": 3,
        >>> }
        >>>
        >>> simple_search = XgbSearch(tune_params, fit_params)
        >>> eval_set = [(x_train, y_train, "train"), (x_test, y_test, "test")]
        >>> # Use this syntax for early stopping
        >>> simple_search.fit(x_train, y_train, eval_set, 10000, 100, verbose_eval=100)
        >>> # No early stopping
        >>> simple_search.fit(x_train, y_train, eval_set, 5, None, verbose_eval=100)
        """
        self.tune_params = tune_params
        self.fit_params = fit_params
        self.search_type = "single model fit"
        self.maximise_score = maximise_score

        if "eval_metric" in fit_params:
            if isinstance(fit_params["eval_metric"], list):
                self.eval_metric = fit_params["eval_metric"][-1]
            else:
                self.eval_metric = fit_params["eval_metric"]
        else:
            self.eval_metric = "auc"

    def fit(
        self,
        X: pd.DataFrame,
        y: list[float | int],
        eval_set: list[tuple[pd.DataFrame, list[float | int], str]],
        num_boost_round: int,
        early_stopping_rounds: int | None = None,
        verbose_eval: bool | int = True,
    ):
        """Fits the model according to the search scheme.

        Args:
            X (pd.DataFrame): Input data frame with features to be used.
            y (list[float  |  int]): A list of targets
            eval_set (list[tuple[pd.DataFrame, list[float  |  int], str]]): List of validation sets for which metrics will evaluated during training. Validation metrics will help us track the performance of the model. For example [(x_train, y_train, "this is a training")].
            num_boost_round (int): Number of rounds to run.
            early_stopping_rounds (int | None, optional): Activates early stopping. Validation metric needs to improve at least once in every early_stopping_rounds round(s) to continue training. See `xgb.fit` help for more information. Defaults to None.
            verbose_eval (bool | int, optional): Requires at least one item in evals. If verbose_eval is True then the evaluation metric on the validation set is printed at each boosting stage. If verbose_eval is an integer then the evaluation metric on the validation set is printed at every given verbose_eval boosting stage. Defaults to True.

        Returns:
            Nothing, but populates the results of the fitting in variable called `results`.

            This variable contains a list of dictionaries with complete run results. Each dict will have following keys.

                * model: xgb.Booster object
                * parameters: complete parameters dict passed to xgb.fit()
                * additional_settings: a dict of additional settings passed to xgb.fit()
                * model_training_results: a complete record of evaluation metrics for each eval set
                * best_iteration (int): best model iteration; when using early stopping this will be < num_boost_rounds
                * best_score (float): value of the last evaluation metric on the last eval_set
                * best_model (xgb.Booster): Model object for the model with best score according to the last eval metric on the last eval set

        """
        self._training_dmatrix = self._convert_to_DMatrix(X, y)
        self._eval_dmatrix = [(self._training_dmatrix, "TRAIN")] + [
            (self._convert_to_DMatrix(X, y), name) for (X, y, name) in eval_set
        ]

        param_list = self._generate_params()
        print(
            f"{Fore.BLUE}Running {self.search_type} over {len(param_list)} iterations.{Style.RESET_ALL}"
        )

        self.results = []

        for index, p in enumerate(param_list):
            print(
                f"🟨🟨🟨🟨🟨🟨🟨      🏃‍♂️‍➡️{Fore.BLACK}Running iteration {index}.{Style.RESET_ALL}       🟨🟨🟨🟨🟨🟨🟨"
            )
            print(p)

            individual_result = {}

            booster = xgb.train(
                dtrain=self._training_dmatrix,
                params=p,
                evals=self._eval_dmatrix,
                num_boost_round=num_boost_round,
                early_stopping_rounds=early_stopping_rounds,
                verbose_eval=verbose_eval,
                evals_result=individual_result,
            )

            loop_res = {
                "model": booster,
                "parameters": p,
                "additional_settings": {
                    "num_boost_round": num_boost_round,
                    "verbose_eval": verbose_eval,
                    "early_stopping_rounds": early_stopping_rounds,
                },
                "model_training_results": individual_result,
            }

            if early_stopping_rounds is not None:
                loop_res["best_iteration"] = booster.best_iteration
                loop_res["best_score"] = booster.best_score
                loop_res["best_model"] = booster[: loop_res["best_iteration"] + 1]
            else:
                loop_res["best_iteration"] = num_boost_round
                loop_res["best_score"] = self._get_best_score(individual_result)
                loop_res["best_model"] = booster

            self.results.append(loop_res)

            print(f"✅ {Fore.BLACK + Style.BRIGHT}Done!{Style.RESET_ALL}")

    def get_best_model_results(self) -> dict[str, Any]:
        """Returns the dictionary for the best model fitted.

        Returns:
            dict[str, Any]: Dict with information about the best model including the model itself.
        """
        self._check_if_fitted()
        best_model_results = self.results[0]

        for r in self.results:
            if (
                self.maximise_score
                and r["best_score"] > best_model_results["best_score"]
            ):
                best_model_results = r
            elif (
                not self.maximise_score
                and r["best_score"] < best_model_results["best_score"]
            ):
                best_model_results = r

        return best_model_results

    def get_best_model(self) -> xgb.Booster:
        """Retrieves the best xbg.Booster model fitted.

        Returns:
            xgb.Booster: of the best model
        """
        return self.get_best_model_results()["best_model"]

    def _check_if_fitted(self):
        if not hasattr(self, "results"):
            raise Exception(
                "You need to run fit() first to be able to get the best model."
            )

    def _get_best_score(self, training_results):
        last_key = list(training_results.keys())[-1]
        metric = list(training_results[last_key].keys())[-1]
        return training_results[last_key][metric][-1]

    def _generate_params(self) -> list[dict]:
        """Generates default parameters for the fitting of the model.

        The base class will simply use the provided parameters once.

        Use `XgbGridSearch` or `XgbRandomSearch` for an implementation of GridSearch and RandomSearch algorithms.

        Returns a list of dicts that will be then used by `fit` function to perform either a grid search or random search.

        As this is the base class, we are going to assume that tune params is a dict of values to be passed directly to XGBoost.
        """
        return [self.fit_params | self.tune_params]

    def _convert_to_DMatrix(self, X: pd.DataFrame, y: list[float | int]) -> DMatrix:
        """Helper function that converts X `pd.DataFrame` and y `list` to XGBoost native DMatrix data type.

        Args:
            X (pd.DataFrame): A data frame with feature values.
            y (list[float | int]): A list of targets to be predicted.

        Returns:
            DMatrix
        """
        return xgb.DMatrix(data=X, label=y, enable_categorical=True)

    def predict(self, X: pd.DataFrame) -> np.array:
        """Generate predictions from the best model fitted. This will select the best combination of hyper parameters tested and, if using early stopping, the best model instance.

        Args:
            X (pd.DataFrame): Input data frame to be scored.

        Returns:
            np.array: Array of probabilities.
        """
        self._check_if_fitted()
        model = self.get_best_model()
        return model.predict(xgb.DMatrix(X))

    def append_prediction_to_df(
        self, X: pd.DataFrame, pred_col: str = "prediction"
    ) -> pd.DataFrame:
        """Calculate predictions and append it to the data frame being scored.

        Args:
            X (pd.DataFrame): Data frame to be scored.
            pred_col (str, optional): Name of the new column to hold prediction. Defaults to "prediction".

        Returns:
            pd.DataFrame: _description_
        """
        X[pred_col] = self.predict(X)
        return X

    def score(
        self,
        X: pd.DataFrame,
        y: list[float | int],
        score_func,
        **kwargs,
    ):
        """Calculate a score using a scoring function with signature score_func(y_true, y_pred, **kwargs).

        Args:
            X (pd.DataFrame): _description_
            y (list[float  |  int]): _description_
            score_func (_type_): _description_

        Returns:
            _type_: _description_

        Example:
        >>> from sklearn.metrics import roc_auc_score
        >>> grid_search.score(x_test, y_test, roc_auc_score)
        """
        predictions = self.predict(X)
        score = score_func(y, predictions, **kwargs)
        return score


class XgbGridSearch(XgbSearch):
    def __init__(
        self,
        tune_params: Dict[str, float | int | str | list[float | int | str]],
        fit_params: Dict[str, str | int | float],
        maximise_score: bool = True,
    ):
        """Set up a XgbGridSearch object.

        Args:
            tune_params: Dictionary of parameters to be searched. At least one of the parameters has to be a list.
            fit_params: Dictionary of parameters that will be passed in addition to the searched params into `xgb.fit` method.
            maximise_score: By default when getting the best model one with the highest score will be returned.

        Example:
        >>> # These parameters will be ALWAYS passed to `xgb.fit` method.
        >>> fit_params = {
        >>>     "device": "cuda",
        >>>     "objective": "binary:logistic",
        >>>     "eval_metric": ["aucpr", "auc"],
        >>> }
        >>> # These parameters will be tuned.
        >>> tune_params = {
        >>>     "eta": [0.01, 0.5],
        >>>     "colsample_bytree": 0.8,
        >>>     "max_depth": 11,
        >>>     "min_child_weight": 3,
        >>> }
        >>>
        >>> grid_search = XgbSearch(tune_params, fit_params)
        >>> eval_set = [(x_train, y_train, "train"), (x_test, y_test, "test")]
        >>> # Use this syntax for early stopping
        >>> grid_search.fit(x_train, y_train, eval_set, 10000, 100, verbose_eval=100)
        >>> # No early stopping
        >>> grid_search.fit(x_train, y_train, eval_set, 5, None, verbose_eval=100)
        """
        list_check = any([isinstance(x, list) for x in tune_params.values()])
        if not list_check:
            raise Exception(
                "At least one of the `tune_params` elements has to be a list."
            )

        # Convert all instances of not list into one
        tune_params = {
            k: v if isinstance(v, list) else [v] for k, v in tune_params.items()
        }

        super().__init__(tune_params, fit_params, maximise_score)
        self.search_type = "grid search"

    def _generate_params(self):
        keys = self.tune_params.keys()
        combinations = itertools.product(*self.tune_params.values())
        return [dict(zip(keys, combo)) | self.fit_params for combo in combinations]


class XgbRandomSearch(XgbSearch):
    def __init__(
        self,
        tune_params: Dict[str, float | int | str | list[float | int | str]],
        fit_params: Dict[str, str | int | float],
        maximise_score: bool = True,
        max_iter_count: int = 100,
    ):
        """Set up a XgbGridSearch object.

        Args:
            tune_params: Dictionary of parameters to be searched. At least one of the parameters has to a tuple. For the parameters that are a tupel of floats/ints, we will generate random numbers between the 2 values of the same data type. For parameters that are a list, one random entity will be selected.
            fit_params: Dictionary of parameters that will be passed in addition to the searched params into `xgb.fit` method.
            maximise_score: By default when getting the best model one with the highest score will be returned.

        Example:
        >>> # These parameters will be ALWAYS passed to `xgb.fit` method.
        >>> fit_params = {
        >>>     "device": "cuda",
        >>>     "objective": "binary:logistic",
        >>>     "eval_metric": ["aucpr", "auc"],
        >>> }
        >>> # These parameters will be tuned.
        >>> tune_params = {
        >>>     "eta": [0.01, 0.5], # select randomly one of the two values here
        >>>     "colsample_bytree": (0.8, 1.0), # generate a random number between 0.8 and 1.0 inclusive
        >>>     "max_depth": (1, 11), # generate a random int between 1 and 11 inclusive
        >>>     "min_child_weight": 3,
        >>> }
        >>>
        >>> grid_search = XgbSearch(tune_params, fit_params)
        >>> eval_set = [(x_train, y_train, "train"), (x_test, y_test, "test")]
        >>> # Use this syntax for early stopping
        >>> grid_search.fit(x_train, y_train, eval_set, 10000, 100, verbose_eval=100)
        >>> # No early stopping
        >>> grid_search.fit(x_train, y_train, eval_set, 5, None, verbose_eval=100)
        """
        tuple_check = any([isinstance(x, tuple) for x in tune_params.values()])
        if not tuple_check:
            raise Exception(
                "At least one of the `tune_params` elements has to be a tuple."
            )

        super().__init__(tune_params, fit_params, maximise_score)
        self.search_type = "random search"
        self.max_iter_count = max_iter_count

    def _generate_params(self):
        res = []
        for i in range(self.max_iter_count):
            loop_res = {}
            for k, v in self.tune_params.items():
                if isinstance(v, list):
                    loop_res[k] = random.choice(v)
                elif isinstance(v, tuple):
                    if isinstance(v[0], float):
                        loop_res[k] = random.uniform(*v)
                    elif isinstance(v[0], int):
                        loop_res[k] = random.randint(*v)
                    else:
                        raise ValueError(
                            f"Only tuples with ranges expressed as int or float are supported. Check {k}:{v}."
                        )
                else:
                    loop_res[k] = v
            res.append(loop_res)
        return res
