# %%
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from RFRegressor import RFRegressor
import logging as log

# Loading dataset
housing = fetch_california_housing(as_frame=True)
data = housing.frame.iloc[:100].copy()
data["ID"] = data.index
data.set_index("ID", inplace=True)

target_column = "MedHouseVal"

# Defining hyperparameter grid
hyperparams = {
    "n_estimators": [100],
    "max_depth": [1, 10],
    "min_samples_split": [2, 5]
}

# Instantiating RFRegressor
rf_model = RFRegressor(
    cv_function=KFold,
    hp_search_function=GridSearchCV,
    cv_kwargs={"n_splits": 3, "shuffle": True},
    hp_search_kwargs={"cv": 3, "scoring": "neg_mean_squared_error"},
    log_level=log.DEBUG,
    random_seed=1
)

# %%
# Training model
final_model, best_params, performance_dict, feat_importance_df = rf_model.trainRFRegressor(
    n_resamples=3,
    data=data,
    target_column=target_column,
    hyperparameters=hyperparams,
    test_size=0.2,
    save_interval_models=True,
    save_path="./",
    save_final_model=True,
    plot_feat_importance=False,
    batch_size=1,
    n_jobs=1,
    final_rf_seed=1
)
# %%

# Making predictions
preds_df = rf_model.predictRFRegressor(
    feature_data=data.drop(columns=[target_column]),
    prediction_col="Prediction",
    final_rf=final_model,
    save_preds=True,
    save_path="./",
    filename="preds"

)

print(preds_df.head())
# %%
