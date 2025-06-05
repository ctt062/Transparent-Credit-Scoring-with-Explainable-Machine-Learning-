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
from explainability import generate_shap_explanations

# Define constants
TRAIN_FILE = '../train.csv'
TEST_FILE = '../test.csv'
GRAPHS_DIR = '../../graphs'
OUTPUT_DIR = '..'

def main():
    # --- Path Setup ---
    # Get the directory where main.py is located
    current_script_dir = os.path.dirname(__file__)

    actual_graphs_dir_path = os.path.join(current_script_dir, GRAPHS_DIR)
    if not os.path.exists(actual_graphs_dir_path):
        os.makedirs(actual_graphs_dir_path)
    
    actual_output_dir_path = os.path.join(current_script_dir, OUTPUT_DIR)
    # Ensure output directory (project root in this case) exists if needed for direct file saves
    if not os.path.exists(actual_output_dir_path):
        os.makedirs(actual_output_dir_path) # Should typically exist (project root)
    
    train_file_path = os.path.join(current_script_dir, TRAIN_FILE)
    test_file_path = os.path.join(current_script_dir, TEST_FILE)

    # --- 1. Load Data ---
    train_df, test_df = load_data(train_file_path, test_file_path)
    original_test_df_for_output = pd.read_csv(test_file_path) # For final output with original IDs

    # --- 2. Clean Data ---
    print("\nCleaning datasets...")
    train_df_cleaned = clean_dataset(train_df.copy())
    test_df_cleaned = clean_dataset(test_df.copy())
    print("After cleaning - Train:", train_df_cleaned.shape, "Test:", test_df_cleaned.shape)

    # --- 3. Feature Engineering ---
    print("\nApplying feature engineering...")
    train_df_engineered = engineer_features(train_df_cleaned.copy())
    test_df_engineered = engineer_features(test_df_cleaned.copy())
    print("Feature engineering done.")

    # --- 4. Encode and Scale Data ---
    train_df_processed, test_df_processed, encoders, target_le, scaler, scaled_cols = \
        encode_scale_data(train_df_engineered.copy(), test_df_engineered.copy())

    # --- 5. Prepare data for modeling ---
    if 'Credit_Score' not in train_df_processed.columns:
        raise ValueError("Target column 'Credit_Score' not found in training data after preprocessing.")
    
    # Define features: columns present in processed train_df, excluding target, and also present in processed test_df
    # This ensures alignment. scaled_cols might be a good candidate if it captures all numeric features correctly.
    # For robustness, explicitly define features based on processed data.
    
    # Initial feature list from train_df_processed, excluding target
    potential_features = [c for c in train_df_processed.columns if c != 'Credit_Score']
    
    # Final features are those in potential_features that are also in test_df_processed
    features = [f for f in potential_features if f in test_df_processed.columns]
    
    # Sanity check (as requested)
    if 'Credit_Score' in features:
        print("ERROR: 'Credit_Score' found in 'features' list after definition. This should not happen.")
        features.remove('Credit_Score') # Attempt to fix, but this indicates a logic flaw if it occurs

    print(f"\nSelected features for modeling ({len(features)}): {features}")

    X = train_df_processed[features]  # X is a DataFrame
    y = train_df_processed['Credit_Score'] # y is a Series
    
    # Align test_df_processed to have exactly the 'features' columns, in the same order, filling missing with 0
    # This is crucial for consistent prediction.
    X_test_final = test_df_processed.reindex(columns=features, fill_value=0)[features]

    # Split training data into train/validation sets. X_train, X_val will be DataFrames.
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"X_train shape: {X_train.shape}, X_val shape: {X_val.shape}")
    print(f"X_test_final shape: {X_test_final.shape}")

    # --- 6. Define Models ---
    y_unique_labels_len = len(np.unique(y)) # Use 'y' which is the full target Series
    models_dict = get_models(y_unique_labels_len)

    # --- 7. Perform Cross-Validation (on full X, y) ---
    perform_cross_validation(X.copy(), y.copy(), models_dict)

    # --- 8. Train Models (on X_train, y_train) ---
    # Pass DataFrames to training
    trained_models = train_all_models(X_train.copy(), y_train.copy(), models_dict)

    # --- 9. Evaluate Models ---
    # Pass DataFrames to evaluation
    evaluate_classification_models(trained_models, X_val.copy(), y_val.copy(), target_le, actual_graphs_dir_path)

    # --- 10. Plot ROC Curves ---
    # Pass DataFrames
    plot_roc_curves_multiclass(trained_models, X_val.copy(), y_val.copy(), target_le, actual_graphs_dir_path)
    
    # --- 11. Generate SHAP Explanations ---
    # Pass X_train and X_val as DataFrames.
    # Their .columns attribute will provide feature names to SHAP.
    print("\n--- SHAP Pre-check ---")
    print(f"X_train for SHAP: type={type(X_train)}, shape={X_train.shape}, columns={list(X_train.columns)}")
    print(f"X_val for SHAP: type={type(X_val)}, shape={X_val.shape}, columns={list(X_val.columns)}")
    generate_shap_explanations(trained_models, X_train.copy(), X_val.copy(), actual_graphs_dir_path)

    # --- 12. Predictions on Test Set and Output ---
    print("\nMaking predictions on the test set...")
    # X_test_final is already a DataFrame aligned with 'features'
    
    # Impute any remaining NaNs in X_test_final using X_train's medians (as per original logic)
    # This step might be redundant if preprocessing and reindex fully handled NaNs, but kept for consistency.
    for col in X_test_final.columns:
        if X_test_final[col].isnull().any():
            if col in X_train.columns: # Check if column exists in X_train to get median
                train_median = X_train[col].median()
                X_test_final[col] = X_test_final[col].fillna(train_median)
                print(f"Imputed NaNs in X_test_final column '{col}' using X_train median: {train_median}")
            else: # Fallback if column somehow not in X_train (should not happen if features are common)
                X_test_final[col] = X_test_final[col].fillna(0)
                print(f"Imputed NaNs in X_test_final column '{col}' using 0 (fallback)")


    logreg_preds_encoded = trained_models['Logistic Regression'].predict(X_test_final)
    rf_preds_encoded = trained_models['Random Forest'].predict(X_test_final)
    xgb_preds_encoded = trained_models['XGBoost'].predict(X_test_final)

    # Inverse transform predictions to original labels
    logreg_preds_labels = target_le.inverse_transform(logreg_preds_encoded)
    rf_preds_labels = target_le.inverse_transform(rf_preds_encoded)
    xgb_preds_labels = target_le.inverse_transform(xgb_preds_encoded)

    # Create DataFrame for predictions
    test_predictions_df = pd.DataFrame({
        'Logistic_Regression_Prediction': logreg_preds_labels,
        'Random_Forest_Prediction': rf_preds_labels,
        'XGBoost_Prediction': xgb_preds_labels
    })
    
    # Add Customer_ID back for easier joining/submission, ensuring row alignment
    num_predictions = len(test_predictions_df)
    if 'Customer_ID' in original_test_df_for_output.columns:
         # Ensure we only take as many IDs as we have predictions
         test_predictions_df['Customer_ID'] = original_test_df_for_output['Customer_ID'].values[:num_predictions]
         # Move Customer_ID to the first column
         cols = ['Customer_ID'] + [col for col in test_predictions_df.columns if col != 'Customer_ID']
         test_predictions_df = test_predictions_df[cols]

    predictions_path = os.path.join(actual_output_dir_path, 'credit_score_predictions.csv')
    test_predictions_df.to_csv(predictions_path, index=False)
    print(f"Test set predictions saved to '{predictions_path}'")
    print("Sample predictions:")
    print(test_predictions_df.head())

    # --- Save individual model predictions and probabilities (example for Logistic Regression) ---
    
    # Output with original test data columns + predicted label for Logistic Regression
    output_df_labels = original_test_df_for_output.iloc[:num_predictions].copy() # Align rows with predictions
    output_df_labels['Predicted_Credit_Score_Logistic'] = logreg_preds_labels
    labels_path = os.path.join(actual_output_dir_path, 'predicted_credit_score_labels_logreg.csv')
    output_df_labels.to_csv(labels_path, index=False)
    print(f"Test set predictions with labels (Logistic Regression) saved to '{labels_path}'")

    # Output with original test data columns + predicted probabilities for Logistic Regression
    logreg_proba = trained_models['Logistic Regression'].predict_proba(X_test_final)
    proba_df = pd.DataFrame(logreg_proba, columns=[f"Probability_{cls_label}" for cls_label in target_le.classes_])
    
    output_df_proba = original_test_df_for_output.iloc[:num_predictions].reset_index(drop=True).copy() # Align rows
    proba_df = proba_df.reset_index(drop=True) # Ensure proba_df also has clean index for concat
    output_df_proba = pd.concat([output_df_proba, proba_df], axis=1)

    proba_path = os.path.join(actual_output_dir_path, 'predicted_credit_score_probabilities_logreg.csv')
    output_df_proba.to_csv(proba_path, index=False)
    print(f"Test set prediction probabilities (Logistic Regression) saved to '{proba_path}'")

    print("\nProcess completed successfully.")

if __name__ == '__main__':
    main()