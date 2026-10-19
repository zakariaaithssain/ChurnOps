import pandas as pd
import joblib
from pathlib import Path
from pandas import DataFrame
from evidently import Dataset, DataDefinition, BinaryClassification, Report
from evidently.presets import DataDriftPreset, ClassificationPreset

# --- NEW IMPORTS FOR THE WORKSPACE UI ---
from evidently.ui.workspace import Workspace

from churn.config import (
    FEATURES_COLS,
    CATEGORICAL_COLS,
    TEST_DATA,
    ARTIFACTS_DIR,
    BOOL_COLS,
)
from churn.create_pipelines import coerce_lgbm_dtypes

SCHEMA = DataDefinition(
    numerical_columns=[
        col
        for col in FEATURES_COLS
        if col not in (CATEGORICAL_COLS + ["label", "msno"] + BOOL_COLS)
    ],
    categorical_columns=CATEGORICAL_COLS + BOOL_COLS + ["label"],
    classification=[
        BinaryClassification(target="label", prediction_probas="prediction")
    ],
)


def _create_dataset(dataframe: DataFrame, model_preds):
    dataframe["prediction"] = model_preds
    return Dataset.from_pandas(dataframe, data_definition=SCHEMA)


def _run_report(
    ref_dataset: Dataset, inference_dataset: Dataset, include_classification: bool
):
    metrics = [DataDriftPreset(include_tests=True)]
    if include_classification:
        metrics.append(ClassificationPreset(include_tests=True))

    report = Report(metrics)
    return report.run(inference_dataset, ref_dataset)


def monitor(inference_parquet: Path):
    """monitor the istonically calibrated lgbm pipeline (winner).
    uses the test dataset as the reference baseline.
    args:
    `inference_parquet`: path to parquet file containing the features (should be the same as r1_features.parquet or r1_features.parquet)
    """
    model_pipeline = joblib.load(ARTIFACTS_DIR / "calib_lgbm_pipeline.joblib")[
        "isotonic"
    ]

    X_test, y_test = joblib.load(TEST_DATA)
    X_test = coerce_lgbm_dtypes(X_test)
    test_preds = model_pipeline.predict_proba(X_test)[:, 1]

    ref_df = X_test.copy()
    ref_df["label"] = y_test
    ref_dataset = _create_dataset(ref_df, test_preds)

    inf_df = pd.read_parquet(inference_parquet).copy()
    inf_df = inf_df.drop(columns=["msno"], errors="ignore")

    has_label = "label" in inf_df.columns

    X_inf = coerce_lgbm_dtypes(inf_df.drop(columns=["label"], errors="ignore"))
    inf_preds = model_pipeline.predict_proba(X_inf)[:, 1]

    if not has_label:
        inf_df["label"] = pd.NA

    inf_dataset = _create_dataset(inf_df, inf_preds)

    # 1. Run the report baseline calculations
    report = _run_report(ref_dataset, inf_dataset, include_classification=has_label)

    # 2. Configure Local Workspace logging
    WORKSPACE_DIR = "evidently_workspace"
    workspace = Workspace.create(
        WORKSPACE_DIR
    )  # Automatically initializes directory if missing

    # 3. Secure or create a dedicated dashboard project
    PROJECT_NAME = "Churn Prediction Service"
    project = None
    for p in workspace.list_projects():
        if p.name == PROJECT_NAME:
            project = p
            break

    if project is None:
        project = workspace.create_project(PROJECT_NAME)
        project.description = "Production tracking for Churn Prediction Engine."

    workspace.add_run(project.id, report)

    return report
