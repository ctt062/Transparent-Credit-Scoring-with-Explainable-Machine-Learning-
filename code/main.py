import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Custom module imports
from data_loader import load_data
from preprocessing import clean_dataset, encode_scale_data
from feature_engineering import engineer_features
from model_training import get_models, perform_cross_validation, train_all_models
from evaluation import evaluate_classification_models, plot_roc_curves_multiclass
from explainability import generate_shap_explanations # ensure this is the correct function name

# Define constants for file paths relative to the project root
# When running 'python code/main.py' from project root, these will be correct.
# If running 'python main.py' from 'code/' dir, paths need to be adjusted.
# Assuming execution from project root: `python code/main.py`
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Get project root

TRAIN_FILE = os.path.join(PROJECT_ROOT, 'train.csv')
TEST_FILE = os.path.join(PROJECT_ROOT, 'test.csv')
GRAPHS_DIR = os.path.join(PROJECT_ROOT, 'graphs')
OUTPUT_CSV_DIR = PROJECT_ROOT # Save CSVs to project root

def main():
    """
    Main function to execute the credit scoring pipeline.

    The pipeline consists of the following steps:
    1.  Load training and testing data.
    2.  Clean the datasets (handle missing values, data types, formatting).
    3.  Perform feature engineering to create new informative features.
    4.  Encode categorical features and scale numerical features.
    5.  Prepare feature (X) and target (y) sets for modeling.
    6.  Initialize machine learning models (Logistic Regression, Random Forest, XGBoost).
    7.  Perform cross-validation on the full training data for each model.
    8.  Train the models on the designated training split.
    9.  Evaluate the trained models on the validation split (accuracy, classification report, confusion matrix).
    10. Plot ROC curves for the models on the validation split.
    11. Generate and save SHAP (SHapley Additive exPlanations) plots for model interpretability,
        adhering to the logic of the original project notebook.
    12. Make predictions on the processed test set using the trained models.
    13. Save the test set predictions (combined and for Logistic Regression specifically,
        including probabilities) to CSV files in the project root directory.

    All generated plots (confusion matrices, ROC curves, SHAP plots) are saved
    to the 'graphs' directory in the project root.
    """
    # --- Path Setup ---
    if not os.path.exists(GRAPHS_DIR):
        os.makedirs(GRAPHS_DIR)
    
    # --- 1. Load Data ---
    train_df_orig, test_df_orig = load_data(TRAIN_FILE, TEST_FILE)
    
    # Keep a copy of original test_df for final output if it has IDs etc.
    # original_test_df_for_output = test_df_orig.copy() # If IDs were not dropped by clean_dataset
    # Or reload if clean_dataset drops IDs we need later
    original_test_df_for_output = pd.read_csv(TEST_FILE)


    # --- 2. Clean Data ---
    print("\nCleaning datasets...")
    train_df_cleaned = clean_dataset(train_df_orig.copy())
    test_df_cleaned = clean_dataset(test_df_orig.copy())
    print("After cleaning - Train:", train_df_cleaned.shape, "Test:", test_df_cleaned.shape)

    # --- 3. Feature Engineering ---
    print("\nApplying feature engineering...")
    train_df_engineered = engineer_features(train_df_cleaned.copy())
    test_df_engineered = engineer_features(test_df_cleaned.copy())
    print("Feature engineering done.")

    # --- 4. Encode and Scale Data ---
    train_df_processed, test_df_processed, encoders, target_le, scaler, common_scaled_cols = \
        encode_scale_data(train_df_engineered.copy(), test_df_engineered.copy())

    # --- 5. Prepare data for modeling ---
    if 'Credit_Score' not in train_df_processed.columns:
        raise ValueError("Target column 'Credit_Score' not found in training data after preprocessing.")
    
    # Define features based on columns in processed training data (excluding target)
    # that are also present in processed test data. This ensures consistency.
    potential_features = [c for c in train_df_processed.columns if c != 'Credit_Score']
    features = [f for f in potential_features if f in test_df_processed.columns]
    
    if not features:
        raise ValueError("No common features found between training and testing sets after processing. Check preprocessing steps.")
    print(f"\nSelected features for modeling ({len(features)}): {features[:5]}...") # Print first 5

    X = train_df_processed[features]
    y = train_df_processed['Credit_Score']
    
    # Align test_df_processed to have exactly the 'features' columns, in the same order
    X_test_final = test_df_processed.reindex(columns=features, fill_value=0)[features]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"X_train shape: {X_train.shape}, X_val shape: {X_val.shape}, X_test_final shape: {X_test_final.shape}")

    # --- 6. Define Models ---
    y_unique_labels_len = len(np.unique(y))
    models_dict = get_models(y_unique_labels_len)

    # --- 7. Perform Cross-Validation (on full X, y) ---
    perform_cross_validation(X.copy(), y.copy(), models_dict)

    # --- 8. Train Models (on X_train, y_train) ---
    trained_models = train_all_models(X_train.copy(), y_train.copy(), models_dict)

    # --- 9. Evaluate Models ---
    evaluate_classification_models(trained_models, X_val.copy(), y_val.copy(), target_le, GRAPHS_DIR)

    # --- 10. Plot ROC Curves ---
    plot_roc_curves_multiclass(trained_models, X_val.copy(), y_val.copy(), target_le, GRAPHS_DIR)
    
    # --- 11. Generate SHAP Explanations ---
    # Pass target_le for mapping class indices to names if SHAP values are per-class
    generate_shap_explanations(trained_models, X_train.copy(), X_val.copy(), target_le, GRAPHS_DIR)

    # --- 12. Predictions on Test Set and Output CSVs ---
    print("\nMaking predictions on the test set for final CSV outputs...")
    
    # Ensure X_test_final is fully imputed if any NaNs remain from reindexing/processing
    for col in X_test_final.columns:
        if X_test_final[col].isnull().any():
            if col in X_train.columns:
                train_median = X_train[col].median()
                X_test_final[col] = X_test_final[col].fillna(train_median)
            else: 
                X_test_final[col] = X_test_final[col].fillna(0) # Fallback

    # --- Original combined prediction output ---
    logreg_preds_encoded = trained_models.get('Logistic Regression').predict(X_test_final) if trained_models.get('Logistic Regression') else None
    rf_preds_encoded = trained_models.get('Random Forest').predict(X_test_final) if trained_models.get('Random Forest') else None
    xgb_preds_encoded = trained_models.get('XGBoost').predict(X_test_final) if trained_models.get('XGBoost') else None

    predictions_data = {}
    if logreg_preds_encoded is not None:
        predictions_data['Logistic'] = target_le.inverse_transform(logreg_preds_encoded)
    if rf_preds_encoded is not None:
        predictions_data['RandomForest'] = target_le.inverse_transform(rf_preds_encoded)
    if xgb_preds_encoded is not None:
        predictions_data['XGBoost'] = target_le.inverse_transform(xgb_preds_encoded)
    
    test_output_df = pd.DataFrame(predictions_data)
    
    # Add Customer_ID or other identifiers if available and needed
    # Assuming original_test_df_for_output has Customer_ID and same number of rows as X_test_final predictions
    if 'Customer_ID' in original_test_df_for_output.columns:
        num_preds = len(next(iter(predictions_data.values()))) # Length of one of the prediction arrays
        test_output_df['Customer_ID'] = original_test_df_for_output['Customer_ID'].iloc[:num_preds].values
        cols = ['Customer_ID'] + [col for col in test_output_df.columns if col != 'Customer_ID']
        test_output_df = test_output_df[cols]

    predictions_path = os.path.join(OUTPUT_CSV_DIR, 'credit_score_predictions.csv')
    test_output_df.to_csv(predictions_path, index=False)
    print(f"Combined test set predictions saved to '{predictions_path}'")
    print(test_output_df.head())

    # --- Logistic Regression specific outputs (Labels and Probabilities) ---
    if trained_models.get('Logistic Regression'):
        model_lr_final = trained_models['Logistic Regression']
        # logreg_preds_encoded already available
        
        # Output with original test data columns + predicted label
        # Align original_test_df_for_output with the number of predictions
        num_lr_predictions = len(logreg_preds_encoded)
        output_df_labels_lr = original_test_df_for_output.iloc[:num_lr_predictions].copy()
        output_df_labels_lr['Predicted_Credit_Score_LR'] = target_le.inverse_transform(logreg_preds_encoded)
        labels_path_lr = os.path.join(OUTPUT_CSV_DIR, 'predicted_credit_score_labels_logreg.csv')
        output_df_labels_lr.to_csv(labels_path_lr, index=False)
        print(f"Test set predictions with labels (Logistic Regression) saved to '{labels_path_lr}'")

        # Output with original test data columns + predicted probabilities
        logreg_proba = model_lr_final.predict_proba(X_test_final)
        proba_df_lr = pd.DataFrame(logreg_proba, columns=[f"Probability_{cls_label}" for cls_label in target_le.classes_])
        
        output_df_proba_lr = original_test_df_for_output.iloc[:num_lr_predictions].reset_index(drop=True).copy()
        proba_df_lr = proba_df_lr.reset_index(drop=True)
        output_df_proba_lr = pd.concat([output_df_proba_lr, proba_df_lr], axis=1)

        proba_path_lr = os.path.join(OUTPUT_CSV_DIR, 'predicted_credit_score_probabilities_logreg.csv')
        output_df_proba_lr.to_csv(proba_path_lr, index=False)
        print(f"Test set prediction probabilities (Logistic Regression) saved to '{proba_path_lr}'")

    print("\nProcess completed.")

if __name__ == '__main__':
    # This structure assumes you run: python code/main.py from the project root
    # If you run python main.py from inside the code/ directory,
    # then TRAIN_FILE should be '../train.csv', etc.
    # The current PROJECT_ROOT calculation is robust for `python code/main.py`.
    main()