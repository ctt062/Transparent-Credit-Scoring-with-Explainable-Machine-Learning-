import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score

def get_models(y_unique_labels_len):
    """
    Initializes and returns a dictionary of classification models.

    The models included are Logistic Regression, Random Forest, and XGBoost Classifier.
    XGBoost's `num_class` and `eval_metric` are set based on the number of unique labels.

    Parameters:
    ----------
    y_unique_labels_len : int
        The number of unique classes in the target variable. This is used to
        configure the XGBoost classifier.

    Returns:
    -------
    dict
        A dictionary where keys are model names (str) and values are
        the corresponding un-trained scikit-learn/XGBoost model objects.
    """
    models = {
        'Logistic Regression': LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        # For XGBoost, ensure eval_metric is appropriate if num_class > 2
        'XGBoost': XGBClassifier(objective='multi:softmax', num_class=y_unique_labels_len, 
                                 eval_metric='mlogloss' if y_unique_labels_len > 2 else 'logloss', 
                                 use_label_encoder=False, random_state=42)
    }
    return models

def perform_cross_validation(X, y, models_dict):
    """
    Performs 5-fold cross-validation for each model in the provided dictionary.

    Prints the cross-validation accuracy scores and the mean accuracy with standard deviation
    for each model. Handles exceptions during cross-validation for any model.

    Parameters:
    ----------
    X : pd.DataFrame or np.ndarray
        The feature matrix.
    y : pd.Series or np.ndarray
        The target variable.
    models_dict : dict
        A dictionary of models to cross-validate, typically from `get_models()`.

    Returns:
    -------
    dict
        A dictionary where keys are model names and values are arrays of
        cross-validation scores. Contains np.nan for models that failed CV.
    """
    cv_results = {}
    print("\nPerforming Cross-Validation...")
    for name, model in models_dict.items():
        print(f"Cross-validating {name}...")
        try:
            scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
            cv_results[name] = scores
            print(f"=== {name} ===")
            print(f"Cross-Validation Accuracy Scores: {scores}")
            print(f"Mean Accuracy: {scores.mean():.4f} ± {scores.std():.4f}\n")
        except Exception as e:
            print(f"Error during cross-validation for {name}: {e}\n")
            cv_results[name] = np.array([np.nan]) # Store NaN if CV fails
    return cv_results

def train_all_models(X_train, y_train, models_dict):
    """
    Trains each model in the provided dictionary using the training data.

    Handles exceptions during the training of any model.

    Parameters:
    ----------
    X_train : pd.DataFrame or np.ndarray
        The training feature matrix.
    y_train : pd.Series or np.ndarray
        The training target variable.
    models_dict : dict
        A dictionary of un-trained model objects, typically from `get_models()`.

    Returns:
    -------
    dict
        A dictionary where keys are model names and values are the corresponding
        trained model objects. Models that failed to train might have a None value
        or be absent, depending on error handling (currently prints error).
    """
    trained_models = {}
    print("\nTraining Models...")
    for name, model in models_dict.items():
        print(f"Training {name}...")
        try:
            model.fit(X_train, y_train)
            trained_models[name] = model
        except Exception as e:
            print(f"Error training {name}: {e}")
            # Optionally, handle if a model fails to train (e.g. skip it for later steps)
    print("Models training attempt finished.")
    return trained_models