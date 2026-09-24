import numpy as np
import pandas as pd
import xgboost as xgb
from schemas import ModelMetrics
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def _get_transformer(
    X: pd.DataFrame, *, exclude: list[str] | None = None
) -> ColumnTransformer:
    if exclude is None:
        exclude = []

    included_cols = set(X.columns) - set(exclude)
    categorical_cols = [col for col in included_cols if X[col].dtype == "object"]
    numerical_cols = [col for col in included_cols if col not in categorical_cols]

    transformer = ColumnTransformer(
        [
            ("num", StandardScaler(), numerical_cols),
            (
                "cat",
                OneHotEncoder(
                    drop="first", sparse_output=False, handle_unknown="ignore"
                ),
                categorical_cols,
            ),
        ],
        remainder="drop",
    )

    return transformer


def _evaluate_model(model, X: pd.DataFrame, y: pd.Series) -> ModelMetrics:
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)[:, 1]
    return ModelMetrics(
        accuracy=float(accuracy_score(y, y_pred)),
        precision=float(precision_score(y, y_pred)),
        recall=float(recall_score(y, y_pred)),
        f1=float(f1_score(y, y_pred)),
        roc_auc=float(roc_auc_score(y, y_pred_proba)),
    )


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = 5,
    test_size=0.2,
    random_state=42,
) -> tuple[Pipeline, ModelMetrics]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Balancing of positive and negative weights.
    # https://xgboost.readthedocs.io/en/stable/tutorials/param_tuning.html#handle-imbalanced-dataset
    scale_pos_weight = np.sum(y_train == 0) / np.sum(y_train == 1)

    pipeline = Pipeline(
        steps=[
            ("transformer", _get_transformer(X, exclude=["user_id"])),
            (
                "clf",
                xgb.XGBClassifier(
                    random_state=random_state,
                    scale_pos_weight=scale_pos_weight,
                ),
            ),
        ]
    )

    param_grid = {
        "clf__n_estimators": list(range(150, 301, 50)),
        "clf__max_depth": list(range(5, 7, 1)),
        "clf__learning_rate": list(np.arange(0.01, 0.11, 0.05)),
        "clf__subsample": list(np.arange(0.2, 0.7, 0.1)),
    }

    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    grid_search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_grid,
        scoring="roc_auc",  # f1 would assume a threshold of 0.5, ROC AUC is threshold-independent
        cv=cv_splitter,
        n_jobs=-1,
        random_state=random_state,
        verbose=2,
    )

    grid_search.fit(X_train, y_train)

    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best ROC AUC score: {grid_search.best_score_:.4f}")

    best_pipe = grid_search.best_estimator_

    metrics = _evaluate_model(best_pipe, X_test, y_test)

    return best_pipe, metrics
