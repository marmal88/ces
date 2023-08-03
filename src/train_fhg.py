import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import Normalizer, OrdinalEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, make_scorer
import mlflow
from dotenv import load_dotenv


class Training_Pipeline:
    """A class for training pipeline including data preprocessing,
    hyperparameter tuning, and logging with MLflow.
    """

    def __init__(self, df_location: str):
        """
        Initializes the Training_Pipeline instance.

        Args:
            df_location (str): Location of the input CSV file.
        """
        self.setup_mlflow()

        self.df = pd.read_csv(df_location)
        # Pre selects the features to drop from the dataframe
        self.X = self.df.drop(
            columns=[
                "median_ces",
                "longitude",
                "latitude",
                "total_bedrooms",
            ]
        )
        # Identify the y-hat
        self.Y = self.df["median_ces"]

        # Create scoring matrix
        self.scoring = {
            "MAE": make_scorer(mean_absolute_error),
            "RMSE": make_scorer(mean_squared_error, squared=False),
        }

    def setup_mlflow(self):
        """Sets up MLflow experiment and logging."""

        # load necessary access keys
        load_dotenv(dotenv_path="docker/.env")

        # Create nested runs
        mlflow.set_tracking_uri(uri="http://127.0.0.1:5005")
        mlflow.set_experiment(experiment_name="random_forest")
        mlflow.sklearn.autolog()

    def hyperparameter_tuning(self, cv_num: str):
        """
        Performs hyperparameter tuning for Linear Regression using GridSearchCV
        and logs the results in MLflow.

        Args:
            cv_num (str): Number of cross-validation folds.
        """
        X_train, _, y_train, _ = self._split()

        # Name the categories in ocean proximity
        categories = [x for x in self.df.ocean_proximity.unique()]

        # Create and fit column transformer
        self.ct = ColumnTransformer(
            [
                ("normalizer", Normalizer(), ["total_rooms", "households"]),
                (
                    "standard_scaler",
                    StandardScaler(),
                    ["housing_median_age", "median_income"],
                ),
                (
                    "ordinal_encoder",
                    OrdinalEncoder(categories=[categories]),
                    ["ocean_proximity"],
                ),
            ],
            remainder="drop",
        )
        self.ct.fit(X_train, y_train)

        # Hyperparameter tuning for HistGradientBoostingRegressor
        hgb_reg_pipe = Pipeline(
            steps=[
                ("column_transfomer", self.ct),
                ("hgb_reg", HistGradientBoostingRegressor()),
            ]
        )

        # Define the parameters
        param_dist = {
            "hgb_reg__warm_start": [True, False],
            "hgb_reg__learning_rate": [0.1, 0.01],
            "hgb_reg__max_depth": [3, 10, 20],
            "hgb_reg__min_samples_leaf": [20, 40, 60],
        }

        # Create randomized search as there are more parameters
        random_search = RandomizedSearchCV(
            hgb_reg_pipe,
            param_distributions=param_dist,
            scoring=self.scoring,
            n_iter=10,
            refit="RMSE",
            cv=5,
        )

        # fitting on training data
        with mlflow.start_run():
            random_search.fit(X_train, y_train)

            # joblib.dump(random_search, 'random_search.pkl')

            # mlflow.log_artifact()
            # print(mlflow.get_run(run_id=run.info.run_id))
            mlflow.sklearn.log_model(
                sk_model=random_search, artifact_path=f"run.info.run_id"
            )  # path=f"run.info.run_id"

        for metric in self.scoring:
            best_index = np.argmax(random_search.cv_results_[f"mean_test_{metric}"])
            best_params = random_search.cv_results_[f"params"][best_index]
            best_score = random_search.cv_results_[f"mean_test_{metric}"][best_index]

            print(f"Best Parameters ({metric}):", best_params)
            print(f"Best Score ({metric}): {best_score:.2f}")

        mlflow.end_run()

    def _split(self):
        """
        Splits the data into training and testing sets.

        Returns:
            X_train, X_test, y_train, y_test: Split datasets.
        """
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.Y, test_size=0.2, random_state=24
        )

        return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    Training_Pipeline(df_location="data/cali_ces_households.csv").hyperparameter_tuning(
        cv_num=5
    )
