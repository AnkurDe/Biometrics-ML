from pathlib import Path
import warnings
import pickle

import pandas as pd
import xgboost as xgb
import catboost as cb

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
import numpy as np


# Suppress only known harmless warnings
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="xgboost"
)


BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_PATH = BASE_DIR / "TabulatedData" / "values_v6_32bit.parquet"

data = pd.read_parquet(DATA_PATH)


def create_balanced_dataset(label: str, data: pd.DataFrame) -> pd.DataFrame:
    """
    Create a balanced binary classification dataset.
    """

    person_data = data[data['name'] == label].copy()
    n_person_samples = len(person_data)

    print(f"Found {n_person_samples} samples for {label}")

    other_data = data[data['name'] != label].copy()

    grouped_others = {
        name: group.sample(frac=1, random_state=42).reset_index(drop=True)
        for name, group in other_data.groupby('name')
    }

    selected_rows = []
    used_counts = {
        name: 0 for name in grouped_others.keys()
    }

    while len(selected_rows) < n_person_samples:

        added = False

        for name, group in grouped_others.items():

            idx = used_counts[name]

            if idx < len(group):
                selected_rows.append(group.iloc[idx])
                used_counts[name] += 1
                added = True

                if len(selected_rows) >= n_person_samples:
                    break

        if not added:
            print("Warning: Insufficient negative samples.")
            break

    other_sampled = pd.DataFrame(selected_rows)

    if len(other_sampled) < n_person_samples:
        person_data = person_data.sample(
            n=len(other_sampled),
            random_state=42
        )

    person_data['target'] = 1
    other_sampled['target'] = 0

    balanced_data = pd.concat(
        [person_data, other_sampled],
        ignore_index=True
    )

    balanced_data = balanced_data.sample(
        frac=1,
        random_state=42
    ).reset_index(drop=True)

    print(f"Balanced dataset size: {len(balanced_data)}")

    return balanced_data


def split_features_targets(data: pd.DataFrame):
    """Split data into features and targets"""
    X = data.drop(['name', 'target'], axis=1)
    y = data['target']

    return train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )


def get_models():
    """Returns the set of models with the hyperparameter tuning parameters"""
    return {
        'Random Forest': (
            RandomForestClassifier(random_state=42),
            {
                'classifier__n_estimators': [100, 200],
                'classifier__max_depth': [10, 20, None],
                'classifier__min_samples_split': [2, 5, 10],
                'classifier__min_samples_leaf': [1, 4, 8],  # Added for regularization
                'classifier__max_features': ['sqrt', 'log2']  # Added for regularization
            }
        ),

        'SVM': (
            SVC(random_state=42),
            {
                'classifier__C': [0.001, 0.01, 0.1, 1, 10],  # Added lower C values for regularization
                'classifier__kernel': ['rbf', 'linear'],
                'classifier__gamma': ['scale', 'auto']
            }
        ),

        'Logistic Regression': (
            LogisticRegression(
                random_state=42,
                max_iter=5000,
                # L2 regularization is now the default; use C parameter to control strength
            ),
            {
                'classifier__C': [0.001, 0.01, 0.1, 1, 10]  # Extended range for better regularization control
            }
        ),

        'Decision Tree': (
            DecisionTreeClassifier(random_state=42),
            {
                'classifier__max_depth': [5, 10, 15, None],  # Reduced max depth options
                'classifier__min_samples_split': [2, 5, 10],
                'classifier__min_samples_leaf': [1, 5, 10]  # Added for regularization
            }
        ),

        'XGBoost': (
            xgb.XGBClassifier(
                objective='binary:logistic',
                random_state=42,
                tree_method='hist',
                n_jobs=-1,
                # Early stopping removed as it requires eval_set which GridSearchCV doesn't support
                # Use regularization parameters instead for overfitting control
            ),
            {
                'classifier__n_estimators': [100, 200],
                'classifier__max_depth': [3, 6],
                'classifier__learning_rate': [0.01, 0.1],
                'classifier__reg_alpha': [0, 0.01, 0.1],  # Added L1 regularization
                'classifier__reg_lambda': [0.01, 0.1, 1]  # Added L2 regularization
            }
        ),

        'GaussianNB': (
            GaussianNB(),
            {
                'classifier__var_smoothing': [1e-9, 1e-8, 1e-7]  # Extended range
            }
        ),

        'KNN': (
            KNeighborsClassifier(),
            {
                'classifier__n_neighbors': [3, 5, 7, 9],  # Added more options
                'classifier__weights': ['uniform', 'distance']
            }
        ),

        'MLP': (
            MLPClassifier(
                random_state=42,
                max_iter=1000,
                early_stopping=True,  # Added early stopping
                validation_fraction=0.1  # Validation set for early stopping
            ),
            {
                'classifier__hidden_layer_sizes': [
                    (50,),
                    (100,),
                    (100, 50)
                ],
                'classifier__activation': ['relu', 'tanh'],
                'classifier__alpha': [0.0001, 0.001, 0.01]  # Added L2 regularization
            }
        ),

        'CatBoost': (
            cb.CatBoostClassifier(
                random_state=42,
                verbose=False,
                # Early stopping removed as it requires eval_set which GridSearchCV doesn't support
                # Use regularization parameters instead for overfitting control
            ),
            {
                'classifier__depth': [4, 6],
                'classifier__learning_rate': [0.01, 0.1],
                'classifier__iterations': [100, 200],
                'classifier__l2_leaf_reg': [1, 3, 5]  # Added L2 regularization
            }
        )
    }


def build_pipeline(classifier):
    """This has the pipeline for the entire procedure i.e. scaling, PCA and classification model"""
    return Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=0.85)),
        ('classifier', clone(classifier))
    ])


def calculate_biometric_metrics(y_true, y_pred):
    """
    Calculates biometric evaluation metrics.

    Returns:
        dict: Dictionary containing FAR, FRR, and EER
    """

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # False Acceptance Rate (FAR)
    far = fp / (fp + tn) if (fp + tn) > 0 else 0

    # False Rejection Rate (FRR)
    frr = fn / (fn + tp) if (fn + tp) > 0 else 0

    # Equal Error Rate (Approximation)
    eer = (far + frr) / 2

    return {
        "FAR": far,
        "FRR": frr,
        "EER": eer
    }


def train_and_evaluate_models(
    X_train,
    X_test,
    y_train,
    y_test,
    label
):
    """
    Trains and evaluates multiple models with:
    - Accuracy
    - Precision
    - Recall
    - F1 Score
    - FAR
    - FRR
    - EER
    """

    results = {}

    models = get_models()

    for name, (classifier, params) in models.items():

        print(f"\n{'=' * 50}")
        print(f"Training {name}")
        print(f"{'=' * 50}")

        try:

            pipeline = build_pipeline(classifier)

            cv = StratifiedKFold(
                n_splits=5,
                shuffle=True,
                random_state=42
            )

            grid_search = GridSearchCV(
                estimator=pipeline,
                param_grid=params,
                cv=cv,
                scoring='accuracy',
                n_jobs=-1,
                verbose=1
            )

            # Train model
            grid_search.fit(X_train, y_train)

            # Best model
            best_model = grid_search.best_estimator_

            # Predictions
            y_pred_train = best_model.predict(X_train)
            y_pred_test = best_model.predict(X_test)

            # -----------------------------
            # TRAIN METRICS
            # -----------------------------
            train_acc = accuracy_score(y_train, y_pred_train)

            train_precision = precision_score(
                y_train,
                y_pred_train,
                zero_division=0
            )

            train_recall = recall_score(
                y_train,
                y_pred_train,
                zero_division=0
            )

            train_f1 = f1_score(
                y_train,
                y_pred_train,
                zero_division=0
            )

            train_bio_metrics = calculate_biometric_metrics(
                y_train,
                y_pred_train
            )

            # -----------------------------
            # TEST METRICS
            # -----------------------------
            test_acc = accuracy_score(y_test, y_pred_test)

            test_precision = precision_score(
                y_test,
                y_pred_test,
                zero_division=0
            )

            test_recall = recall_score(
                y_test,
                y_pred_test,
                zero_division=0
            )

            test_f1 = f1_score(
                y_test,
                y_pred_test,
                zero_division=0
            )

            test_bio_metrics = calculate_biometric_metrics(
                y_test,
                y_pred_test
            )

            # Store results
            results[name] = {

                # Train Metrics
                'train_accuracy': train_acc,
                'train_precision': train_precision,
                'train_recall': train_recall,
                'train_f1': train_f1,
                'train_far': train_bio_metrics['FAR'],
                'train_frr': train_bio_metrics['FRR'],
                'train_eer': train_bio_metrics['EER'],

                # Test Metrics
                'test_accuracy': test_acc,
                'test_precision': test_precision,
                'test_recall': test_recall,
                'test_f1': test_f1,
                'test_far': test_bio_metrics['FAR'],
                'test_frr': test_bio_metrics['FRR'],
                'test_eer': test_bio_metrics['EER'],

                # Model Details
                'best_params': grid_search.best_params_,
                'best_model': best_model
            }

            # -----------------------------
            # PRINT RESULTS
            # -----------------------------
            print("\nBest Parameters:")
            print(grid_search.best_params_)

            print("\nTRAIN METRICS")
            print("-" * 30)
            print(f"Accuracy : {train_acc:.4f}")
            print(f"Precision: {train_precision:.4f}")
            print(f"Recall   : {train_recall:.4f}")
            print(f"F1 Score : {train_f1:.4f}")
            print(f"FAR      : {train_bio_metrics['FAR']:.4f}")
            print(f"FRR      : {train_bio_metrics['FRR']:.4f}")
            print(f"EER      : {train_bio_metrics['EER']:.4f}")

            print("\nTEST METRICS")
            print("-" * 30)
            print(f"Accuracy : {test_acc:.4f}")
            print(f"Precision: {test_precision:.4f}")
            print(f"Recall   : {test_recall:.4f}")
            print(f"F1 Score : {test_f1:.4f}")
            print(f"FAR      : {test_bio_metrics['FAR']:.4f}")
            print(f"FRR      : {test_bio_metrics['FRR']:.4f}")
            print(f"EER      : {test_bio_metrics['EER']:.4f}")

            # print("\nClassification Report:")
            # print(
            #     classification_report(
            #         y_test,
            #         y_pred_test,
            #         target_names=[f'Not {label}', label]
            #     )
            # )

        except Exception as e:

            print(f"Error training {name}: {e}")

            results[name] = {
                'error': str(e)
            }

    # ---------------------------------
    # SELECT BEST MODEL
    # ---------------------------------
    valid_results = {
        k: v for k, v in results.items()
        if 'error' not in v
    }

    if not valid_results:
        raise ValueError(f"No valid models trained for {label}")

    best_model_name, best_result = max(
        valid_results.items(),
        key=lambda x: x[1]['test_accuracy']
    )

    best_pipeline = best_result['best_model']

    print(f"\n{'=' * 60}")
    print(f"BEST MODEL FOR {label}: {best_model_name}")
    print(f"{'=' * 60}")

    print(f"Test Accuracy : {best_result['test_accuracy']:.4f}")
    print(f"Test Precision: {best_result['test_precision']:.4f}")
    print(f"Test Recall   : {best_result['test_recall']:.4f}")
    print(f"Test F1 Score : {best_result['test_f1']:.4f}")
    print(f"Test FAR      : {best_result['test_far']:.4f}")
    print(f"Test FRR      : {best_result['test_frr']:.4f}")
    print(f"Test EER      : {best_result['test_eer']:.4f}")

    return best_pipeline, results


def calculate_metrics_statistics(all_metrics):
    """
    Calculate mean and standard deviation of metrics across all persons.

    Args:
        all_metrics: List of dictionaries containing metrics for each person

    Returns:
        Dictionary with mean and std for each metric
    """
    metrics_summary = {}

    # Define all metrics to track
    metric_keys = [
        'train_accuracy', 'test_accuracy',
        'train_Recall', 'test_Recall',
        'train_precision', 'test_precision',
        'train_f1', 'test_f1',
        'train_FAR', 'test_FAR',
        'train_FRR', 'test_FRR',
        'train_EER', 'test_EER'
    ]

    for metric in metric_keys:
        values = [m.get(metric) for m in all_metrics if m.get(metric) is not None]
        if values:
            metrics_summary[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'values': values
            }

    return metrics_summary


def display_metrics_statistics(metrics_summary):
    """
    Display metrics statistics in a formatted table.

    Args:
        metrics_summary: Dictionary with mean and std for each metric
    """
    print(f"{'Metric':<25} {'Mean':<12} {'Std Dev':<12}")
    print("-" * 50)

    # Group metrics by type
    accuracy_metrics = ['train_accuracy', 'test_accuracy']
    recall_metrics = ['train_Recall', 'test_Recall']
    precision_metrics = ['train_precision', 'test_precision']
    f1_metrics = ['train_f1', 'test_f1']
    far_metrics = ['train_FAR', 'test_FAR']
    frr_metrics = ['train_FRR', 'test_FRR']
    eer_metrics = ['train_EER', 'test_EER']

    groups = [
        ("Accuracy", accuracy_metrics),
        ("Recall", recall_metrics),
        ("Precision", precision_metrics),
        ("F1 Score", f1_metrics),
        ("FAR", far_metrics),
        ("FRR", frr_metrics),
        ("EER", eer_metrics)
    ]

    for group_name, metrics in groups:
        print(f"\n{group_name}:")
        print("-" * 50)
        for metric in metrics:
            if metric in metrics_summary:
                summary = metrics_summary[metric]
                print(f"  {metric:<22} {summary['mean']:<12.6f} {summary['std']:<12.6f}")


def run():
    """Function to run the pipeline for all persons"""
    unique_names = sorted(data['name'].unique())  # Sort for consistent Person1, Person2, etc.
    trained_dir = BASE_DIR / "Trained_Models"
    trained_dir.mkdir(exist_ok=True)

    # Store metrics for all persons
    all_metrics = []

    for i, label in enumerate(unique_names, 1):
        try:
            print(f"\nTraining models for Person {i}: {label}")

            balanced_data = create_balanced_dataset(label, data)
            X_train, X_test, y_train, y_test = split_features_targets(balanced_data)
            # Train and evaluate models. This returns the best pipeline and a
            # results dictionary containing per-model train/test metrics and the model object.
            best_pipeline, results = train_and_evaluate_models(
                X_train, X_test, y_train, y_test, label
            )

            # Save only the best model as {label}.pkl
            best_pkl = trained_dir / f"{label}.pkl"
            best_info = results.get(max(results.keys(), key=lambda k: results[k].get('test_accuracy', -1)))
            best_data = {
                'model': best_pipeline,
                'train_accuracy': best_info.get('train_accuracy'),
                'test_accuracy': best_info.get('test_accuracy'),
                'train_Recall': best_info.get('train_recall'),
                'test_Recall': best_info.get('test_recall'),
                'train_precision': best_info.get('train_precision'),
                'test_precision': best_info.get('test_precision'),
                'train_f1': best_info.get('train_f1'),
                'test_f1': best_info.get('test_f1'),
                'train_FAR': best_info.get('train_far'),
                'test_FAR': best_info.get('test_far'),
                'train_FRR': best_info.get('train_frr'),
                'test_FRR': best_info.get('test_frr'),
                'train_EER': best_info.get('train_eer'),
                'test_EER': best_info.get('test_eer')
            }
            with open(best_pkl, 'wb') as f:
                pickle.dump(best_data, f)
            print(f"Saved best model for {label} as {best_pkl}")

            # Store metrics for statistics calculation
            all_metrics.append(best_data)
        except Exception as _:
            continue

    # Display mean and standard deviation of all metrics
    print(f"\n{'=' * 80}")
    print(f"OVERALL STATISTICS (Mean ± Std Dev) - Across {len(all_metrics)} Persons")
    print(f"{'=' * 80}\n")

    if all_metrics:
        metrics_summary = calculate_metrics_statistics(all_metrics)
        display_metrics_statistics(metrics_summary)


if __name__ == "__main__":
    run()
