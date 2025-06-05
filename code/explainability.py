import shap
import pandas as pd
import matplotlib.pyplot as plt
import os
from plot_utils import save_plot

def generate_shap_explanations(trained_models, X_train_df, X_val_df, output_dir):
    print("\nGenerating SHAP Explanations...")
    print(f"generate_shap_explanations: X_train_df shape: {X_train_df.shape}, cols: {list(X_train_df.columns)[:5]}...")
    print(f"generate_shap_explanations: X_val_df shape: {X_val_df.shape}, cols: {list(X_val_df.columns)[:5]}...")

    # --- Logistic Regression ---
    if 'Logistic Regression' in trained_models:
        model_lr = trained_models['Logistic Regression']
        # KernelExplainer is slow, so often a smaller sample of X_val is used for explanation too.
        # For now, we explain full X_val_df but use a small background summary from X_train_df.
        sample_size_lr_background = min(100, X_train_df.shape[0])
        
        # Decide on sample size for X_val_df for LR explanation if full is too slow
        # X_val_lr_explain = X_val_df.sample(min(500, X_val_df.shape[0]), random_state=42) if X_val_df.shape[0] > 500 else X_val_df
        X_val_lr_explain = X_val_df # Explaining full X_val for now, can be changed.

        if sample_size_lr_background > 0:
            print(f"\nProcessing Logistic Regression SHAP...")
            X_train_summary_lr = shap.sample(X_train_df, sample_size_lr_background, random_state=42)
            print(f"  LR: X_train_summary_lr (background) shape: {X_train_summary_lr.shape}")
            print(f"  LR: X_val_lr_explain (to explain) shape: {X_val_lr_explain.shape}")
            
            explainer_logreg = shap.KernelExplainer(model_lr.predict_proba, X_train_summary_lr)
            shap_values_logreg = explainer_logreg.shap_values(X_val_lr_explain)

            original_class_labels = model_lr.classes_
            if not isinstance(shap_values_logreg, list) or len(shap_values_logreg) != len(original_class_labels):
                print(f"  LR ERROR: shap_values_logreg is not a list of expected length. Type: {type(shap_values_logreg)}")
            else:
                for i, encoded_class_label in enumerate(original_class_labels):
                    current_shap_values_class = shap_values_logreg[i]
                    print(f"  LR Class {encoded_class_label}: SHAP values shape: {current_shap_values_class.shape}, X_val_lr_explain features: {X_val_lr_explain.shape[1]}")
                    if current_shap_values_class.shape[1] != X_val_lr_explain.shape[1]:
                        print(f"    LR MISMATCH for class {encoded_class_label}!")
                        continue

                    plt.figure()
                    shap.summary_plot(current_shap_values_class, X_val_lr_explain, show=False)
                    plt.title(f"SHAP Summary for Logistic Regression (Class {encoded_class_label})")
                    save_plot(plt.gcf(), f"LogisticRegression_shap_summary_class_{encoded_class_label}.png", output_dir)
                    plt.close()
        else:
            print("Skipping Logistic Regression SHAP due to insufficient training data for summary.")

    # --- Random Forest --- (Sampling X_val_df for SHAP calculation and plots)
    if 'Random Forest' in trained_models:
        model_rf = trained_models['Random Forest']
        print(f"\nProcessing Random Forest SHAP...")

        # Sample X_val_df for RF SHAP calculation as in original notebook
        sample_size_rf = min(500, X_val_df.shape[0])
        if sample_size_rf > 0 :
            X_val_sample_rf = X_val_df.sample(sample_size_rf, random_state=42)
            print(f"  RF: X_train_df (background for TreeExplainer) shape: {X_train_df.shape}")
            print(f"  RF: X_val_sample_rf (to explain and plot) shape: {X_val_sample_rf.shape}")

            # TreeExplainer uses X_train_df as background to understand feature distributions learned by model
            explainer_rf = shap.TreeExplainer(model_rf, X_train_df)
            # Calculate SHAP values only on the sample of X_val
            shap_values_rf_sampled = explainer_rf.shap_values(X_val_sample_rf)

            original_class_labels = model_rf.classes_

            if not isinstance(shap_values_rf_sampled, list) or len(shap_values_rf_sampled) != len(original_class_labels):
                print(f"  RF ERROR: shap_values_rf_sampled is not a list of expected length. Type: {type(shap_values_rf_sampled)}")
            else:
                for i, encoded_class_label in enumerate(original_class_labels):
                    current_shap_values_class_sampled = shap_values_rf_sampled[i]
                    print(f"  RF Class {encoded_class_label}: Sampled SHAP values shape: {current_shap_values_class_sampled.shape}, X_val_sample_rf features: {X_val_sample_rf.shape[1]}")

                    if current_shap_values_class_sampled.shape[1] != X_val_sample_rf.shape[1]:
                        print(f"    RF MISMATCH for class {encoded_class_label} on sampled data!")
                        continue

                    # Summary Plot on the sampled data
                    fig_rf_summary, ax_rf_summary = plt.subplots(figsize=(10, 8))
                    shap.summary_plot(current_shap_values_class_sampled, X_val_sample_rf, show=False, ax=ax_rf_summary)
                    ax_rf_summary.set_title(f"SHAP Summary for Random Forest (Class {encoded_class_label}, Sampled)")
                    save_plot(fig_rf_summary, f"RandomForest_shap_summary_class_{encoded_class_label}_sampled.png", output_dir)
                    plt.close(fig_rf_summary)

                    # Beeswarm Plot on the same sampled data
                    fig_rf_beeswarm, ax_rf_beeswarm = plt.subplots(figsize=(10, 8))
                    shap.beeswarm(current_shap_values_class_sampled, X_val_sample_rf, show=False)
                    ax_rf_beeswarm.set_title(f"SHAP Beeswarm for Random Forest (Class {encoded_class_label}, Sampled)")
                    save_plot(fig_rf_beeswarm, f"RandomForest_shap_beeswarm_class_{encoded_class_label}_sampled.png", output_dir)
                    plt.close(fig_rf_beeswarm)
        else:
            print("Skipping Random Forest SHAP due to insufficient validation data for sampling.")


    # --- XGBoost --- (Using TreeExplainer, currently on full X_val_df, can be sampled too if slow)
    if 'XGBoost' in trained_models:
        model_xgb = trained_models['XGBoost']
        print(f"\nProcessing XGBoost SHAP...")
        
        # Option to sample X_val_df for XGBoost if needed
        # X_val_xgb_explain = X_val_df.sample(min(500, X_val_df.shape[0]), random_state=42) if X_val_df.shape[0] > 500 else X_val_df
        X_val_xgb_explain = X_val_df # Explaining full X_val_df for XGBoost

        print(f"  XGB: X_train_df (background) shape: {X_train_df.shape}")
        print(f"  XGB: X_val_xgb_explain (to explain) shape: {X_val_xgb_explain.shape}")

        explainer_xgb = shap.TreeExplainer(model_xgb, X_train_df)
        shap_values_xgb = explainer_xgb.shap_values(X_val_xgb_explain)

        original_class_labels = model_xgb.classes_
        if not isinstance(shap_values_xgb, list) or len(shap_values_xgb) != len(original_class_labels):
             print(f"  XGB ERROR: shap_values_xgb is not a list of expected length. Type: {type(shap_values_xgb)}")
        else:
            for i, encoded_class_label in enumerate(original_class_labels):
                current_shap_values_class = shap_values_xgb[i]
                print(f"  XGB Class {encoded_class_label}: SHAP values shape: {current_shap_values_class.shape}, X_val_xgb_explain features: {X_val_xgb_explain.shape[1]}")
                if current_shap_values_class.shape[1] != X_val_xgb_explain.shape[1]:
                    print(f"    XGB MISMATCH for class {encoded_class_label}!")
                    continue

                plt.figure()
                shap.summary_plot(current_shap_values_class, X_val_xgb_explain, show=False)
                plt.title(f"SHAP Summary for XGBoost (Class {encoded_class_label})")
                save_plot(plt.gcf(), f"XGBoost_shap_summary_class_{encoded_class_label}.png", output_dir)
                plt.close()

                # Beeswarm for XGBoost (on X_val_xgb_explain)
                fig_xgb_beeswarm, ax_xgb_beeswarm = plt.subplots(figsize=(10, 8))
                shap.beeswarm(current_shap_values_class, X_val_xgb_explain, show=False)
                ax_xgb_beeswarm.set_title(f"SHAP Beeswarm for XGBoost (Class {encoded_class_label})")
                save_plot(fig_xgb_beeswarm, f"XGBoost_shap_beeswarm_class_{encoded_class_label}.png", output_dir)
                plt.close(fig_xgb_beeswarm)

    print("SHAP plots generation attempt finished.")