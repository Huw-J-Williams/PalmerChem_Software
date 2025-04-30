# %%
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from pathlib import Path
import logging as log

import sys
sys.path.insert(0, "/users/yhb18174/PalmerChem_Software/src/models/")

from RFRegressor import RFRegressor

# Load and prepare data
data = pd.read_csv("california_housing_test_data.csv", index_col='ID')
target_column = "MedHouseVal"
hyperparams = {
    "n_estimators": [100],
    "max_depth": [1, 10],
    "n_jobs": [1],
    "min_samples_split": [2, 5]
}

# Instantiate RFRegressor
model = RFRegressor(
    cv_function=KFold,
    hp_search_function=GridSearchCV,
    cv_kwargs={"n_splits": 3, "shuffle": True},
    hp_search_kwargs={"cv": 3, "scoring": "neg_mean_squared_error"},
    log_level=log.DEBUG,
    random_seed=1
)

def test_rfr_training(rfr_model=model):
    # Train model
    final_model, best_params, performance_dict, feat_importance_df = model.trainRFRegressor(
        n_resamples=1,
        data=data,
        target_column=target_column,
        hyperparameters=hyperparams,
        test_size=0.2,
        save_interval_models=False,
        save_final_model=False,
        plot_feat_importance=False,
        batch_size=1,
        n_jobs=1,
        final_rf_seed=1
    )

    assert isinstance(final_model, RandomForestRegressor)
    assert isinstance(performance_dict, dict)

    assert best_params == {'max_depth': 10, 'min_samples_split': 2, 'n_estimators': 100, 'n_jobs': 1, 'random_state': 1}
    assert round(feat_importance_df.iloc[0]['Importance'], 2) == 0.62
    
# %%

def test_rfr_prediction(rfr_model=model):
    # Making predictions
    preds_df = model.predictRFRegressor(
        feature_data=data.drop(columns=[target_column]),
        prediction_col="Prediction",
        final_rf=None,
        save_preds=True,
        save_path="./",
        filename="preds"
    )

    assert round(preds_df['Prediction'].iloc[0], 2) == 4.11
    assert round(preds_df['Uncertainty'].iloc[0], 2) == 0.59

    assert round(preds_df['Prediction'].iloc[50], 2) == 1.65
    assert round(preds_df['Uncertainty'].iloc[50], 2) == 0.18

    assert round(preds_df['Prediction'].iloc[-1], 2) == 1.85
    assert round(preds_df['Uncertainty'].iloc[-1], 2) == 0.23



    