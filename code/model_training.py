import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score

def get_models(y_unique_labels_len):
    models = {
        'Logistic Regression': LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': XGBClassifier(objective='multi:softmax', num_class=y_unique_labels_len, eval_metric='mlogloss', use_label_encoder=False, random_state=42)
    }
    return models

def perform_cross_validation(X, y, models_dict):
    cv_results = {}
    print("\nPerforming Cross-Validation...")
    for name, model in models_dict.items():
        scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
        cv_results[name] = scores
        print(f"=== {name} ===")
        print(f"Cross-Validation Accuracy Scores: {scores}")
        print(f"Mean Accuracy: {scores.mean():.4f} ± {scores.std():.4f}")
    return cv_results

def train_all_models(X_train, y_train, models_dict):
    trained_models = {}
    print("\nTraining Models...")
    for name, model in models_dict.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model
    print("Models trained.")
    return trained_models