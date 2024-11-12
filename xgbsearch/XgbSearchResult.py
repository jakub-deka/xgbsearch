from xgboost.core import Booster
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class XgbSearchModel:

    def __init__(
        self,
        model: Booster,
        parameters: dict[str, str | float | int],
        evals_result,
        num_boost_round: int,
        early_stopping_rounds: int | None,
    ):
        self.model = model
        self.parameters = parameters
        self.evals_result = evals_result
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds

        if early_stopping_rounds is not None:
            self.best_iteration = model.best_iteration
            self.best_score = model.best_score
            self.best_model = model[: model.best_iteration + 1]
        else:
            self.best_iteration = num_boost_round
            self.best_score = self._get_last_score(evals_result)
            self.best_model = model[: model.best_iteration + 1]

    def _get_last_score(self, training_results):
        last_key = list(training_results.keys())[-1]
        metric = list(training_results[last_key].keys())[-1]
        return training_results[last_key][metric][-1]

    def model_training_results_as_df(
        self, eval_sets: list[str] | None = None, metrics: list[str] | None = None
    ) -> pd.DataFrame:
        best_step = self.best_iteration
        dfs = []
        for dataset_name, results in self.evals_result.items():
            for metric, values in results.items():

                if eval_sets is None or dataset_name in eval_sets:
                    if metrics is None or metric in metrics:
                        inner_df = pd.DataFrame(
                            {
                                "dataset_name": dataset_name,
                                "metric_name": metric,
                                "step": range(len(values)),
                                "metric_value": values,
                            }
                        )
                        dfs.append(inner_df)

        res = pd.concat(dfs).assign(
            is_best=lambda x: np.where(x.step == best_step, 1, 0)
        )

        return res

    def plot_model_training(
        self, eval_sets: list[str] | None = None, metrics: list[str] | None = None
    ):
        df_norm = self.model_training_results_as_df(eval_sets, metrics)
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

    def __repr__(self):
        return f"XgbSearchModel(model={self.model}, parameters={self.parameters}, num_boost_round={self.num_boost_round}, early_stopping_rounds={self.early_stopping_rounds}, model_training_results=list[OrderedDict], best_iteration={self.best_iteration}, best_score={self.best_score}, best_model={self.best_model})"


class XgbSearchResults:
    def __init__(self, maximise_score: bool = True):
        self._results = []
        self._maximise_score = maximise_score

    def add_model(self, model: XgbSearchModel | list[XgbSearchModel]):
        if isinstance(model, XgbSearchModel):
            self._results.append(model)
        elif isinstance(model, list):
            self._results += model

    def get_models(self) -> list[XgbSearchModel]:
        return self._results

    def get_best_model(self) -> XgbSearchModel:
        best_model = self._results[0]

        for r in self._results:
            if self._maximise_score and r.best_score > best_model.best_score:
                best_model = r
            elif not self._maximise_score and r.best_score < best_model.best_score:
                best_model = r

        return best_model

    def fitted(self):
        return len(self._results) > 0

    def __repr__(self):
        r = f"XgbSearchResults(maximise_score={self._maximise_score}, result_count={len(self._results)})"
        for result in self._results:
            r += "\n" + str(result)

        return r
