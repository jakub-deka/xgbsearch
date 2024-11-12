import stat
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Any

from xgbsearch import XgbSearch


class XgbResultDisplay:
    """
    A class providing utility functions for easily displaying the results of model runs using xgbsearch package.

    Methods:
        model_results_as_df_norm: Returns pandas data frame with fitting results in a normalised form given model_results.
        model_results_as_df_flat: Returns pandas data frame with one row per step with each column representing one metric on one dataset.
        model_summary_as_df: Returns a large pandas data frame with all modellign steps evaluation results for all metrics/datasets combinations.
        plot_model_training_performance: Generates a chart for the model training performance for the specified model results.
    """

    def __init__(self):
        pass

    @staticmethod
    def model_results_as_df_norm(
        model_results: dict[Any],
        eval_sets: list[str] | None = None,
        metrics: list[str] | None = None,
    ) -> pd.DataFrame:
        """Convert fitting metrics from list of ordered dicts to pandas DataFrame.

        Args:
            model_results (dict[Any]): model_result to be converted to data frame. Note that you want to pass into this the model_result dictionary generated when you run `fit`.
            eval_sets (list[str] | None, optional): List of datasets to be kept. Defaults to None which will return all datasets.
            metrics (list[str] | None, optional): List of metrics to be kept. Defaults to None which will return all metrics.

        Returns:
            pd.DataFrame: A normalised data frame with one row per step, per dataset, per metric.

        Example:
            >>> XgbResultDisplay.model_results_as_df_norm(grid_search.get_best_model_results())
        """
        best_step = model_results["best_iteration"]
        model_training_results = model_results["model_training_results"]
        dfs = []
        for dataset_name, results in model_training_results.items():
            for metric, values in results.items():

                if eval_sets is None or dataset_name in eval_sets:
                    if metrics is None or metric in metrics:
                        inner_df = pd.DataFrame(
                            {
                                "step": range(len(values)),
                                "dataset_name": dataset_name,
                                "metric_name": metric,
                                "metric_value": values,
                            }
                        )
                        dfs.append(inner_df)

        res = pd.concat(dfs).assign(
            is_best=lambda x: np.where(x.step == best_step, 1, 0)
        )

        return res

    @staticmethod
    def model_results_as_df_flat(
        model_results: dict[Any],
        eval_sets: list[str] | None = None,
        metrics: list[str] | None = None,
    ) -> pd.DataFrame:
        """Convert fitting metrics from list of ordered dicts to a flat pandas DataFrame.

        Args:
            model_results (dict[Any]): model_result to be converted to data frame. Note that you want to pass into this the model_result dictionary generated when you run `fit`.
            eval_sets (list[str] | None, optional): List of datasets to be kept. Defaults to None which will return all datasets.
            metrics (list[str] | None, optional): List of metrics to be kept. Defaults to None which will return all metrics.

        Returns:
            pd.DataFrame: A normalised data frame with one row per step, per dataset, per metric.

        Example:
            >>> XgbResultDisplay.model_results_as_df_flat(grid_search.get_best_model_results())
        """
        df = XgbResultDisplay.model_results_as_df_norm(
            model_results, eval_sets, metrics
        ).pivot_table(
            values="metric_value",
            columns=["dataset_name", "metric_name"],
            index=["step", "is_best"],
        )
        df.columns = ["_".join(col) for col in df.columns]

        return df.reset_index()

    @staticmethod
    def model_summary_as_df(search_obj: XgbSearch) -> pd.DataFrame:
        """Collects results of fitting all the models into a single data frame

        Args:
            search_obj (XgbSearch): fitted XgbSearch object

        Returns:
            pd.DataFrame

        Example:
            >>> grid_search = XgbGridSearch(tune_params_grid, fit_params)
            >>> eval_set = [(x_train, y_train, "train"), (x_test, y_test, "test")]
            >>> grid_search.fit(x_train, y_train, eval_set, 10000, 100, verbose_eval=100)
            >>> XgbResultDisplay.model_summary_as_df(grid_search)
        """
        model_results = [x for x in search_obj.results]
        res = []
        for i, r in enumerate(model_results):
            df = (
                XgbResultDisplay.model_results_as_df_flat(r)
                .assign(model_id=i)
                .assign(parameters=str(r["parameters"]))
            )
            df = XgbResultDisplay.pull_cols_to_front(df, ["model_id", "parameters"])
            res.append(df)
        return pd.concat(res)

    @staticmethod
    def pull_cols_to_front(df: pd.DataFrame, front_cols: list[str]):
        columns = front_cols + [col for col in df.columns if col not in front_cols]
        return df[columns]

    @staticmethod
    def plot_model_training_performance(
        model_results,
        eval_sets: list[str] | None = None,
        metrics: list[str] | None = None,
    ):
        """Displays the results of fitting the model with step/model generation on x axis.

        Args:
            model_results: Results of fitting the model.
            eval_sets (list[str] | None, optional): List of datasets to be kept. Defaults to None which will return all datasets.
            metrics (list[str] | None, optional): List of metrics to be kept. Defaults to None which will return all metrics.

        Examples:
            >>> grid_search = XgbGridSearch(tune_params_grid, fit_params)
            >>> eval_set = [(x_train, y_train, "train"), (x_test, y_test, "test")]
            >>> grid_search.fit(x_train, y_train, eval_set, 10000, 100, verbose_eval=100)
            >>> XgbResultDisplay.plot_model_training_performance(grid_search.results[1])
        """
        df_norm = XgbResultDisplay.model_results_as_df_norm(
            model_results, eval_sets, metrics
        )
        metrics = df_norm.metric_name.unique()
        fig, ax = plt.subplots(
            figsize=(8 * len(metrics), 6), nrows=1, ncols=len(metrics)
        )

        if len(metrics) == 1:
            ax = [ax]
        for i, metric in enumerate(metrics):
            local_df = df_norm.query(f"metric_name == '{metric}'")
            sns.lineplot(
                data=local_df, x="step", y="metric_value", hue="dataset_name", ax=ax[i]
            )
            ax[i].axvline(
                x=local_df.query("is_best == 1").step.min(),
                color="black",
                linestyle=":",
                label="best iteration",
                alpha=0.3,
            )
            ax[i].legend()

            sns.scatterplot(
                data=local_df.query("is_best == 1"),
                x="step",
                y="metric_value",
                hue="dataset_name",
                ax=ax[i],
                s=250,
                marker="*",
                legend=False,
            )

            ax[i].set(
                title=f"{metric}",
                xlabel="Model iteration",
                ylabel=metric,
            )

        plt.show()
