import shap
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from plot_utils import save_plot

def generate_shap_explanations(trained_models, X_train_df, X_val_df, target_le, output_dir):
    print("\nGenerating SHAP Explanations (Original Logic Adherence)...")
    
    # --- Logistic Regression ---
    if 'Logistic Regression' in trained_models:
        model_lr = trained_models['Logistic Regression']
        if model_lr is None: print("Skipping LR SHAP: Model not trained."); return

        print(f"\nProcessing Logistic Regression SHAP...")
        print(f"  LR: X_train_df (background) shape: {X_train_df.shape}")
        print(f"  LR: X_val_df (to explain) shape: {X_val_df.shape}")

        try:
            # Original: explainer_logreg = shap.Explainer(logreg.predict, X_train)
            explainer_logreg = shap.Explainer(model_lr.predict, X_train_df)
            shap_values_logreg_exp_obj = explainer_logreg(X_val_df) # This is a shap.Explanation object

            # model_lr.predict for multi-class returns a single array of predicted labels.
            # So, shap_values_logreg_exp_obj.values should be a single array.
            if isinstance(shap_values_logreg_exp_obj.values, np.ndarray):
                print(f"  LR: SHAP values (for .predict output) shape: {shap_values_logreg_exp_obj.values.shape}")
                if shap_values_logreg_exp_obj.values.shape[1] == X_val_df.shape[1]: # Check feature dimension
                    plt.figure()
                    shap.plots.beeswarm(shap_values_logreg_exp_obj, show=False, max_display=15)
                    plt.title(f"SHAP Beeswarm for Logistic Regression (Explaining Predicted Label)")
                    save_plot(plt.gcf(), f"LogisticRegression_shap_beeswarm_predict.png", output_dir)
                    
                    plt.figure()
                    shap.summary_plot(shap_values_logreg_exp_obj, plot_type="bar", show=False, max_display=15)
                    plt.title(f"SHAP Summary (Bar) for Logistic Regression (Explaining Predicted Label)")
                    save_plot(plt.gcf(), f"LogisticRegression_shap_summary_bar_predict.png", output_dir)
                else:
                    print(f"    LR MISMATCH features: SHAP {shap_values_logreg_exp_obj.values.shape[1]} vs Data {X_val_df.shape[1]}")
            else: # Should not happen if explaining model.predict output for scikit-learn
                print(f"  LR: Unexpected structure for SHAP values from model.predict: {type(shap_values_logreg_exp_obj.values)}")

        except Exception as e_lr_shap:
            print(f"  LR SHAP ERROR: {type(e_lr_shap).__name__} - {e_lr_shap}")

    # --- Random Forest ---
    if 'Random Forest' in trained_models:
        model_rf = trained_models['Random Forest']
        if model_rf is None: print("Skipping RF SHAP: Model not trained."); return

        print(f"\nProcessing Random Forest SHAP...")
        sample_size_rf = min(500, X_val_df.shape[0])

        if sample_size_rf > 0 and not X_val_df.empty:
            sampled_X_rf = X_val_df.sample(n=sample_size_rf, random_state=42)
            print(f"  RF: sampled_X_rf (background and to explain) shape: {sampled_X_rf.shape}")

            try:
                # Original: explainer_rf = shap.Explainer(rf.predict, sampled_X)
                # Original: shap_values_rf = explainer_rf(sampled_X)
                explainer_rf = shap.Explainer(model_rf.predict, sampled_X_rf) # Use sample as background
                shap_values_rf_exp_obj = explainer_rf(sampled_X_rf)     # Explain the same sample

                if isinstance(shap_values_rf_exp_obj.values, np.ndarray):
                    print(f"  RF: SHAP values (for .predict output) shape: {shap_values_rf_exp_obj.values.shape}")
                    if shap_values_rf_exp_obj.values.shape[1] == sampled_X_rf.shape[1]:
                        # Original: plt.figure(figsize=(60, 60)); shap.summary_plot(shap_values_rf, sampled_X)
                        # The Explanation object itself can be passed to summary_plot
                        plt.figure(figsize=(12,10)) # Adjusted figsize from 60x60
                        shap.summary_plot(shap_values_rf_exp_obj, plot_type="dot", show=False, max_display=15) # Default summary is dot (beeswarm-like)
                        plt.title(f"SHAP Summary for Random Forest (Explaining Predicted Label, Sampled)")
                        save_plot(plt.gcf(), f"RandomForest_shap_summary_predict_sampled.png", output_dir)
                    else:
                        print(f"    RF MISMATCH features: SHAP {shap_values_rf_exp_obj.values.shape[1]} vs Data {sampled_X_rf.shape[1]}")
                else:
                     print(f"  RF: Unexpected structure for SHAP values from model.predict: {type(shap_values_rf_exp_obj.values)}")

            except Exception as e_rf_shap:
                print(f"  RF SHAP ERROR: {type(e_rf_shap).__name__} - {e_rf_shap}")
        else:
            print("Skipping Random Forest SHAP (insufficient data or empty sample).")

    # --- XGBoost ---
    if 'XGBoost' in trained_models:
        model_xgb = trained_models['XGBoost']
        if model_xgb is None: print("Skipping XGB SHAP: Model not trained."); return
        
        print(f"\nProcessing XGBoost SHAP...")
        print(f"  XGB: X_val_df (to explain) shape: {X_val_df.shape}")

        try:
            # Original: explainer_xg = shap.Explainer(xgb_model) -> no background data
            # This makes `shap.Explainer` pick `TreeExplainer` with `feature_perturbation="tree_path_dependent"`
            explainer_xg = shap.Explainer(model_xgb) # No explicit background data
            shap_values_xg_exp_obj = explainer_xg(X_val_df)

            # TreeExplainer (picked by shap.Explainer) for multi-class usually explains predict_proba by default,
            # resulting in a list of arrays in .values.
            if isinstance(shap_values_xg_exp_obj.values, list) and \
               len(shap_values_xg_exp_obj.values) == len(model_xgb.classes_):
                print("  XGB: SHAP values are per-class (list output, likely from predict_proba).")
                for i, class_idx_label in enumerate(model_xgb.classes_):
                    original_class_name = target_le.classes_[class_idx_label]
                    current_shap_values = shap_values_xg_exp_obj.values[i]
                    print(f"  XGB Class {original_class_name}: SHAP values shape {current_shap_values.shape}, Data features: {X_val_df.shape[1]}")
                    if current_shap_values.shape[1] != X_val_df.shape[1]:
                        print(f"    XGB MISMATCH for class {original_class_name} features!")
                        continue

                    plt.figure()
                    shap.summary_plot(current_shap_values, X_val_df, show=False, max_display=15)
                    plt.title(f"SHAP Summary for XGBoost (Class: {original_class_name})")
                    save_plot(plt.gcf(), f"XGBoost_shap_summary_class_{original_class_name}.png", output_dir)
                    
            elif isinstance(shap_values_xg_exp_obj.values, np.ndarray): # If it explained model.predict
                print("  XGB: SHAP values are a single array (e.g. from explaining model.predict).")
                if shap_values_xg_exp_obj.values.shape[1] == X_val_df.shape[1]:
                    plt.figure()
                    shap.summary_plot(shap_values_xg_exp_obj, show=False, max_display=15)
                    plt.title(f"SHAP Summary for XGBoost (Explaining Predicted Label)")
                    save_plot(plt.gcf(), f"XGBoost_shap_summary_predict.png", output_dir)
                else:
                     print(f"    XGB MISMATCH features: SHAP {shap_values_xg_exp_obj.values.shape[1]} vs Data {X_val_df.shape[1]}")
            else:
                 print(f"  XGB: Unexpected SHAP value structure: {type(shap_values_xg_exp_obj.values)}")

        except Exception as e_xgb_shap:
            print(f"  XGB SHAP ERROR: {type(e_xgb_shap).__name__} - {e_xgb_shap}")

    print("SHAP plots generation (original logic) finished.")