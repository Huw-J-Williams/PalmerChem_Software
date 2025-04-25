import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split

import random as rand
import logging as log
import inspect
from typing import Callable, Union
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr
import joblib
from joblib import Parallel, delayed
import json
from pathlib import Path




class RFRegressor():
    def __init__(
                self,
                 cv_function: Callable,
                 hp_search_function: Callable,
                 cv_kwargs: dict,
                 hp_search_kwargs: dict,
                 log_level: int=log.INFO,
                 random_seed: int=None
                 ):
        
        """
        Description
        -----------
        Initialize the RFRegressor class for training and evaluating RandomForestRegressor models 
        with custom cross-validation and hyperparameter search strategies.

        Parameters
        ----------
        cv_function : Callable
            Cross-validation constructor (e.g., sklearn.model_selection.KFold or StratifiedKFold).
        hp_search_function : Callable
            Hyperparameter search class (e.g., sklearn.model_selection.GridSearchCV or RandomizedSearchCV).
        cv_kwargs : dict
            Keyword arguments to configure the cross-validation object.
        hp_search_kwargs : dict
            Keyword arguments to configure the hyperparameter search object.
        log_level : int, optional
            Logging level for the internal logger (default: logging.INFO).
        random_seed: int, optional
            Seed set by the user for testing and evaluation, setting same seed for all random processes.
        """
        
        # Setting up logger for the class
        self.logger = log.getLogger(self.__class__.__name__)
        self.logger.setLevel(log_level)

        # Making sure logger only has one handler
        if not self.logger.hasHandlers():
            handler = log.StreamHandler()
            formatter = log.Formatter('%(name)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        self.logger.info("RFRegressor initialised.\n")

        # Setting class variables
        self.cv_function = cv_function
        self.cv_kwargs = cv_kwargs
        self.hp_search_function = hp_search_function
        self.hp_search_kwargs = hp_search_kwargs

        # Initialising random seed
        self.random_seed = random_seed

    def _set_inner_cv(
        self, cv_kwargs, cv_seed=None
                     ):
        """
        Description
        -----------
        Initialize the cross-validation (CV) object for use during hyperparameter tuning.

        Parameters
        ----------
        cv_kwargs : dict
            Keyword arguments to pass to the cross-validation constructor (e.g., number of splits).
        cv_seed : int, optional
            Random seed for reproducibility. If None, a seed will be generated.

        Returns
        -------
        inner_cv : object
            The initialized cross-validation object (e.g., KFold, StratifiedKFold).
        cv_seed : int
            The random seed used to configure the CV object.
        """

        # Setting random seed value
        if self.random_seed:
            cv_seed = self.random_seed

        # Setting random seed for CV function
        init_sig = inspect.signature(self.cv_function)
        if "random_state" in init_sig.parameters:
            cv_kwargs["random_state"] = cv_seed
            self.logger.debug(f"Set random_state={cv_seed} in cv_kwargs for {self.cv_function.__name__}\n")

        self.inner_cv = self.cv_function(**cv_kwargs)

        return self.inner_cv, cv_seed
    
    def _set_hyperparameter_search(
        self, search_kwargs, hyperparameters, estimator, search_seed=None
    ):
        """
        Description
        -----------
        Initialize a hyperparameter search object (e.g., GridSearchCV) using the provided 
        estimator, search space, and configuration.

        Parameters
        ----------
        search_kwargs : dict
            Dictionary of keyword arguments for the hyperparameter search class 
            (e.g., scoring, cv, n_iter for RandomizedSearchCV, etc.).
        hyperparameters : dict
            Hyperparameter grid or distribution for the search (used with param_grid or param_distributions).
        estimator : object
            The machine learning estimator to optimize (e.g., RandomForestRegressor).
        search_seed : int, optional
            Random seed for reproducibility of the search (auto-generated if not provided).

        Returns
        -------
        hp_search_object : object
            Instantiated hyperparameter search object (e.g., GridSearchCV or RandomizedSearchCV).
        search_seed : int
            The random seed used for the search process.
        """

        if self.random_seed:
            search_seed = self.random_seed

        # Setting random seed for hyperparameter search function
        init_sig = inspect.signature(self.hp_search_function)
        if "random_state" in init_sig.parameters:
            search_kwargs["random_state"] = search_seed
            self.logger.debug(f"Set random_state={search_seed} in search_kwargs for {self.hp_search_function.__name__}\n")

        search_kwargs['param_grid'] = hyperparameters
        search_kwargs['estimator'] = estimator
        hp_search_object = self.hp_search_function(**search_kwargs)

        return hp_search_object, search_seed
        
    def _calculate_performance(
        self, feature_test: pd.DataFrame, target_test: pd.DataFrame, best_rf: object
    ):
        """
        Returns
        -------
        Dictionary of performance metrics in this order-
        1. Bias
        2. Standard Error of Potential
        3. Mean Squared Error
        4. Root Mean Squared Error (computed from SDEP and Bias)
        5. Pearson R coefficient
        6. Spearman R coefficient
        7. r2 score
        """
        # TO BE REPLACED BY EXTERNAL CODE

        # Calculate Errors
        true = target_test.astype(float)
        pred =  best_rf.predict(feature_test)
        pred = np.asarray(pred)
    
        errors = true - pred

        # Calculate performance metrics
        bias = np.mean(errors)
        sdep = (np.mean((true - pred - (np.mean(true - pred))) ** 2)) ** 0.5
        mse = mean_squared_error(true, pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(true, pred)

        # Pearson & Spearman correlations
        try:
            r_pearson, p_pearson = pearsonr(true, pred)
        except Exception as e:
            self.logger.warning(f"Pearson correlation failed: {e}\n")
            r_pearson, p_pearson = None, None

        try:
            r_spearman, _ = spearmanr(true, pred)
        except Exception as e:
            self.logger.warning(f"Spearman correlation failed: {e}\n")
            r_spearman = None

        return ({
                "bias": bias,
                "sdep": sdep,
                "mse": mse,
                "rmse": rmse,
                "r2": r2,
                "r_pearson": r_pearson,
                "p_pearson": p_pearson,
                "r_spearman": r_spearman,
                },
                true,
                pred)

    def _fit_model_and_evaluate(
            self,
            resample_n: int,
            features: pd.DataFrame,
            targets: pd.DataFrame,
            test_size: float,
            save_interval_models: bool,
            save_path: str,
            hyperparameters: dict,
            cv_seed:int=None,
            search_seed:int=None
    ):
            
        """
        Description
        -----------
        Perform one resample iteration of model training, hyperparameter tuning, 
        evaluation, and feature importance extraction.

        Parameters
        ----------
        resample_n : int
            Resample iteration number (used for saving models and logs).
        features : pd.DataFrame
            Feature matrix (input variables) for training and testing.
        targets : pd.DataFrame
            Target values (output variable) corresponding to the feature matrix.
        test_size : float
            Fraction of data to use for testing (must be < 1).
        save_interval_models : bool
            If True, saves the model from this resample iteration to disk.
        save_path : str
            Path to save the model if `save_interval_models` is True.
        hyperparameters : dict
            Hyperparameter grid for tuning the RandomForestRegressor.
        cv_seed : int, optional
            Random seed used specifically for cross-validation reproducibility
        search_seed : int, optional
            Random seed used specifically for hyperparameter search reproducibility

        Returns
        -------
        best_params : dict
            Best hyperparameter values found by the hyperparameter search.
        performance : dict
            Dictionary of performance metrics (bias, sdep, mse, rmse, r2, Pearson r, etc.).
        feature_importances : np.ndarray
            Array of feature importances from the best RandomForestRegressor.
        resample_n : int
            The current resample iteration number (incremented from input).
        true : np.ndarray
            True target values from the test set.
        pred : np.ndarray
            Predicted target values from the test set.
        """

        resample_n = resample_n + 1     # Starting from 1, not 0

        self.logger.debug(f"Running _fit_model_and_evaluate\n")

        # Setting random seed value     
        rf_seed = rand.randint(0, 2**31)        

        # Doing the train/test split
        feat_tr, feat_te, tar_tr, tar_te = train_test_split(
            features, targets, test_size=test_size, random_state=rf_seed
        )

        self.logger.debug(
            f"""Length of test set for batch {resample_n}:
        Features = {len(feat_te)}
        Targets = {len(tar_te)}\n"""
        )
        
        self.logger.debug(
            f"""Length of training set for batch {resample_n}:
        Features = {len(feat_tr)}
        Targets = {len(tar_tr)}\n"""
        )       

        # Converting DataFrames to Numpy arrays if necessary
        tar_tr = tar_tr.values.ravel() if isinstance(tar_tr, pd.DataFrame) else tar_tr
        tar_te = tar_te.values.ravel() if isinstance(tar_te, pd.DataFrame) else tar_te

        # Initialising the RandomForestRegressor model
        rfr = RandomForestRegressor()

        self.inner_cv, kfold_rng = self._set_inner_cv(cv_kwargs=self.cv_kwargs, cv_seed=cv_seed)

        model, _ = self._set_hyperparameter_search(
            search_kwargs=self.hp_search_kwargs, 
            hyperparameters=hyperparameters,
            estimator=rfr,
            search_seed=search_seed
            )

        self.logger.debug(f"Checkpoint 1: Inner CV and HP search initialised\n")
        model.fit(feat_tr, tar_tr)

        self.logger.debug(f"Checkpoint 2: RFR model trained\n")

        # Obtaining the best model
        best_rf = model.best_estimator_

        performance, true, pred = self._calculate_performance(
            target_test=tar_te, feature_test=feat_te, best_rf=best_rf
        )
    
        self.logger.debug(f"Checkpoint 3: Performance metrics calculated\n")

        if save_interval_models:
            # Save model
            joblib.dump(best_rf, f"{save_path}/model_resample_{resample_n}.pkl")

            # Save predictions
            pd.Series(pred, index=feat_te.index).to_csv(
                f"{save_path}/preds_resample_{resample_n}.csv.gz",
                compression="gzip",
                index_label="ID"
            )

            # Save performance
            with open(f"{save_path}/performance_stats_resample_{resample_n}.json", "w") as file:
                json.dump(performance, file, indent=4)

            # Save feature importances
            feat_importance_df = pd.DataFrame({
                "Feature": features.columns,
                "Importance": best_rf.feature_importances_
            }).sort_values(by="Importance", ascending=False)

            feat_importance_df.to_csv(
                f"{save_path}/feature_importance_resample_{resample_n}.csv", index=False
            )

        return (
            model.best_params_,
            performance,
            best_rf.feature_importances_,
            resample_n,
            true,
            pred,
        )
    
    def trainRFRegressor(
            self,
            n_resamples: int,
            data: Union[pd.DataFrame, str],
            target_column: str,
            hyperparameters: dict,
            test_size:float,
            metadata_columns: list=None,
            save_interval_models:bool=False,
            save_path:str=None,
            save_final_model:bool=False,
            plot_feat_importance:bool=False,
            batch_size:int=2,
            n_jobs:int=1,
            cv_seeds:list=None,
            search_seeds:list=None,
            final_rf_seed: int=None,
    ):
        """
        Description
        -----------
        Train a RandomForestRegressor model with repeated resampling, 
        hyperparameter tuning, and performance evaluation.

        Parameters
        ----------
        n_resamples : int
            Number of resampling iterations (outer cross-validation loop).
        data : pd.DataFrame or str
            Full training dataset as a DataFrame or path to a CSV file.
        target_column : str
            Name of the target column in the dataset.
        hyperparameters : dict
            Hyperparameter grid for RandomForestRegressor.
        test_size : float
            Proportion of the dataset to use as a test set in each resample.
        save_interval_models : bool, optional
            If True, saves the model from each resample iteration.
        save_path : str, optional
            Directory path to save all outputs (models, metrics, data).
        save_final_model : bool, optional
            If True, saves the final trained model and related files.
        plot_feat_importance : bool, optional
            If True, saves a plot and data of feature importances.
        batch_size : int, optional
            Number of resample iterations per job (used with multiprocessing).
        n_jobs : int, optional
            Number of parallel jobs for resampling (used with joblib.Parallel).
        cv_seeds : list of int, optional
            List of random seeds for each resample’s cross-validation step.
            If None, generated randomly.
        search_seeds : list of int, optional
            List of random seeds for each resample’s hyperparameter search step.
            If None, generated randomly.
        final_rf_seed : int, optional
            Random seed to apply to the final RandomForestRegressor for reproducibility.
        metadata_columns: list, optional
            List of column names which contain information which is not either feature or 
            target values

        Returns
        -------
        final_rf : RandomForestRegressor
            The final trained RandomForestRegressor model using best parameters.
        best_params : dict
            Hyperparameters selected as most frequent across all resamples.
        performance_dict : dict
            Averaged performance metrics across all resample iterations.
        feat_importance_df : pd.DataFrame
            Feature importance scores averaged across all resamples.
        """

        # Setting seeds for each process
        if cv_seeds is None:
            cv_seeds = [rand.randint(0, 2**31) for _ in range(n_resamples)]
        if search_seeds is None:
            search_seeds = [rand.randint(0, 2**31) for _ in range(n_resamples)]

        assert len(cv_seeds) == n_resamples, "cv_seeds must match n_resamples"
        assert len(search_seeds) == n_resamples, "search_seeds must match n_resamples"
    
        if self.random_seed is not None:
            final_rf_seed = self.random_seed
        elif final_rf_seed is not None:
            pass 
        else:
            final_rf_seed = rand.randint(0, 2**31)


        if isinstance(data, str):
            data = pd.read_csv(data, index_col="ID")

        if metadata_columns:
            metadata = data[[metadata_columns]]
            columns_to_drop = metadata_columns + target_column
        else:
            columns_to_drop = [target_column]
        features = data.drop(columns=columns_to_drop)
        targets = data[[target_column]]

        if save_interval_models:
            self.interval_path = Path(f"{save_path}/all_resample_data/")
            self.interval_path.mkdir(exist_ok=True)
        else:
            self.interval_path = save_path
        
        def _process_batch(batch_indices:list):
            """
            Wrapped function to allow for easy multiprocessing. Fits
            models for each resample and calculates performance metrics.
            """
            results_batch = []
            for n in batch_indices:
                result = self._fit_model_and_evaluate(
                    resample_n=n,
                    features=features,
                    targets=targets,
                    test_size=test_size,
                    save_interval_models=save_interval_models,
                    save_path=self.interval_path,
                    hyperparameters=hyperparameters,
                    cv_seed=cv_seeds[n],
                    search_seed=search_seeds[n]
                    )
                results_batch.append(result)
            return results_batch
        
        n_batches = (n_resamples + batch_size - 1) // batch_size
        batches=[
            range(i * batch_size, min((i + 1) * batch_size, n_resamples))
            for i in range(n_batches)
        ]

        # Multiprocessing each resample
        results_batches = Parallel(n_jobs=n_jobs)(
            delayed(_process_batch)(batch) for batch in batches
        )

        # Flattening results and unpacking into lists
        results = [result for batch in results_batches for result in batch]

        (
            best_params_ls,
            self.performance_ls,
            feat_importance_ls,
            self.resample_number_ls,
            self.true_vals_ls,
            self.pred_vals_ls,
        ) = zip(*results)

        self.best_params_df = pd.DataFrame(best_params_ls)
        best_params = self.best_params_df.mode().iloc[0].to_dict()
        best_params['random_state'] = final_rf_seed

        self.logger.debug(f"Best Params DataFrame:\n{self.best_params_df}\n")
        self.logger.debug(f"Mode of Params:\n{best_params}\n")

        # Forcing integer type on all parameters except max_features
        for key, value in best_params.items():
            if key != "max_features":
                best_params[key] = int(value)

        # Getting average performance across all resamples
        self.performance_dict = {
            "Bias": round(
                float(np.mean([perf['bias'] for perf in self.performance_ls])), 4
            ),
            "SDEP": round(
                float(np.mean([perf['sdep'] for perf in self.performance_ls])), 4
            ),
            "MSE": round(
                float(np.mean([perf['mse'] for perf in self.performance_ls])), 4
            ),
            "RMSE": round(
                float(np.mean([perf['rmse'] for perf in self.performance_ls])), 4
            ),
            "r2": round(float(np.mean([perf['r2'] for perf in self.performance_ls])), 4),
            "Pearson_r": round(
                float(np.mean([perf['r_pearson'] for perf in self.performance_ls])), 4
            ),
            "Pearson_p": round(
                float(np.mean([perf['p_pearson'] for perf in self.performance_ls])), 4
            ),
        }

        # Getting average feature importance across all resamples
        avg_feat_importance = np.mean(feat_importance_ls, axis=0)
        feat_importance_df = pd.DataFrame(
            {"Feature": features.columns.tolist(),
             "Importance": avg_feat_importance}
        ).sort_values(by="Importance", ascending=False)

        if plot_feat_importance:
            # INSERT PLOT FUNCTION
            pass

        # Training final model using best parameters
        self.final_rf = RandomForestRegressor(**best_params)
        self.final_rf.fit(features, targets.values.ravel())
        self.logger.info(f"Final RandomForestRegressor model trained.\n")

        # Saving all data
        if save_final_model:
            Path(f"{save_path}/training_data/").mkdir(parents=True, exist_ok=True)

            self.logger.info(f"Saving final model to:\n{save_path}/final_model.pkl\n")
            joblib.dump(self.final_rf, f"{save_path}/training_data/final_model.pkl")

            with open(f"{save_path}/training_data/performance_stats.json", "w") as file:
                json.dump(self.performance_dict, file, indent=4)

            with open(f"{save_path}/training_data/best_params.json", "w") as file:
                json.dump(best_params, file, indent=4)

            features.to_csv(
                f"{save_path}/training_data/training_features.csv.gz",
                index_label="ID",
                compression="gzip",
            )
            targets.to_csv(
                f"{save_path}/training_data/training_targets.csv.gz",
                index_label="ID",
                compression="gzip",
            )

        self.logger.info(f"Performance of final RandomForestRegressor model:\n{self.performance_dict}\n")

        return (
            self.final_rf,
            best_params,
            self.performance_dict,
            feat_importance_df
        )
    
    def predictRFRegressor(
            self,
            feature_data: Union[pd.DataFrame, str],
            prediction_col: str,
            final_rf: Union[RandomForestRegressor, str],
            save_preds: bool=False,
            save_path: str=None,
            filename: str=None,
    ):
        
        """
        Description
        -----------
        Generate predictions from a trained RandomForestRegressor model,
        including uncertainty estimation based on the standard deviation
        across all individual tree predictions.

        Parameters
        ----------
        feature_data : pd.DataFrame or str
            The input features as a DataFrame or a path to a CSV file (indexed by "ID").
        prediction_col : str
            The name of the column to store the predicted values in the output DataFrame.
        final_rf : RandomForestRegressor or str
            The trained RandomForestRegressor model or a path to a saved model (.pkl file).
        save_preds : bool, optional
            Whether to save the predictions and uncertainties to a compressed CSV file.
        save_path : str, optional
            Directory path to save the predictions (if save_preds is True).
        filename : str, optional
            File name (without extension) to save predictions under (if save_preds is True).

        Returns
        -------
        pd.DataFrame
            DataFrame containing predictions and corresponding uncertainty estimates
            from the RandomForestRegressor model.
        """

        # Loading data if csv path given, if necessary
        if isinstance(feature_data, str):
            feature_data = pd.read_csv(feature_data, index_col="ID")
        
        # Loading final RFR model, if necessary
        if isinstance(final_rf, str):
            rf_model = joblib.load(final_rf)

        elif isinstance(final_rf, RandomForestRegressor):
            rf_model = final_rf
        
        else:
            rf_model = self.final_rf

        preds_df = pd.DataFrame()
        preds_df[prediction_col] = rf_model.predict(feature_data)
        preds_df.index = feature_data.index

        all_tree_preds = np.stack(
            [tree.predict(feature_data.to_numpy()) for tree in rf_model.estimators_]
        )

        preds_df["Uncertainty"] = np.std(all_tree_preds, axis=0)

        if save_preds:
            if save_path is None or filename is None:
                raise ValueError("Both save_path and filename must be provided to save predictions")
            
            preds_df.to_csv(
                f"{save_path}/{filename}.csv.gz",
                index_label="ID",
                compression="gzip",
            )

        return preds_df
