# %%
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import KFold, GridSearchCV
from RFRegressor import RFRegressor
import logging as log

# Loading dataset
housing = fetch_california_housing(as_frame=True)
data = housing.frame.iloc[:100].copy()
data["ID"] = data.index
data.set_index("ID", inplace=True)

data.to_csv("/users/yhb18174/PalmerChem_Software/src/models/testing/california_housing_test_data.csv")

target_column = "MedHouseVal"

# Defining hyperparameter grid
hyperparams = {
    "n_estimators": [100],
    "max_depth": [1, 10],
    "n_jobs": [1],

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
    save_interval_models=False,
    save_path="./",
    save_final_model=False,
    plot_feat_importance=False,
    batch_size=1,
    n_jobs=1,
    final_rf_seed=1
)
# %%

print(feat_importance_df)


# Making predictions
preds_df = rf_model.predictRFRegressor(
    feature_data=data.drop(columns=[target_column]),
    prediction_col="Prediction",
    final_rf=final_model,
    save_preds=True,
    save_path="./",
    filename="preds"

)

# print(preds_df.head())
# %%
