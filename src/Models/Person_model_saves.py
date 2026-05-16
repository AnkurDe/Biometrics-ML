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

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold,
    learning_curve
)

from sklearn.metrics import (
    accuracy_score,
    classification_report
)

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier


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
        ('pca', PCA(n_components=0.85)),  # Reduced from 0.90 to 0.85 for less aggressive dimensionality reduction
        ('classifier', clone(classifier))
    ])


def train_and_evaluate_models(
    X_train,
    X_test,
    y_train,
    y_test,
    label
):
    """Does training and evaluation of models"""
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

            grid_search.fit(X_train, y_train)

            best_model = grid_search.best_estimator_

            y_pred_train = best_model.predict(X_train)
            y_pred_test = best_model.predict(X_test)

            train_acc = accuracy_score(y_train, y_pred_train)
            test_acc = accuracy_score(y_test, y_pred_test)

            results[name] = {
                'train_accuracy': train_acc,
                'test_accuracy': test_acc,
                'best_params': grid_search.best_params_,
                'best_model': best_model  # Store the actual trained model
            }

            print(f"\nBest Parameters:")
            print(grid_search.best_params_)

            print(f"\nTrain Accuracy: {train_acc:.4f}")
            print(f"Test Accuracy : {test_acc:.4f}")

            print("\nClassification Report:")
            print(
                classification_report(
                    y_test,
                    y_pred_test,
                    target_names=[f'Not {label}', label]
                )
            )

        except Exception as e:

            print(f"Error training {name}: {e}")

            results[name] = {
                'error': str(e)
            }

    # Select the best model
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
    train_accuracy = best_result['train_accuracy']
    test_accuracy = best_result['test_accuracy']

    print(f"\nBest model for {label}: {best_model_name} with test accuracy {test_accuracy:.4f}")

    return best_pipeline, train_accuracy, test_accuracy


def run():
    """Function to run the pipeline for all persons"""
    unique_names = sorted(data['name'].unique())  # Sort for consistent Person1, Person2, etc.
    trained_dir = BASE_DIR / "Trained_Models"
    trained_dir.mkdir(exist_ok=True)

    for i, label in enumerate(unique_names, 1):
        try:
            print(f"\nTraining models for Person {i}: {label}")

            balanced_data = create_balanced_dataset(label, data)
            X_train, X_test, y_train, y_test = split_features_targets(balanced_data)

            best_pipeline, train_accuracy, test_accuracy = train_and_evaluate_models(X_train, X_test, y_train, y_test, label)

            # Save the best pipeline with accuracies
            model_data = {
                'model': best_pipeline,
                'train_accuracy': train_accuracy,
                'test_accuracy': test_accuracy
            }

            pkl_path = trained_dir / f"{label}.pkl"
            with open(pkl_path, 'wb') as f:
                pickle.dump(model_data, f)
            print(f"Saved best model for {label} to {pkl_path} (Train Acc: {train_accuracy:.4f}, Test Acc: {test_accuracy:.4f})")
        except Exception as _:
            continue


if __name__ == "__main__":
    run()
