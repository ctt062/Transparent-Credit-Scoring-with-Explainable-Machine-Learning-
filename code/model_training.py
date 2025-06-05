import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score

def get_models(y_unique_labels_len):
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