import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize
from plot_utils import plot_confusion_matrix_custom, save_plot

def evaluate_classification_models(trained_models, X_val, y_val, target_le, output_dir):
    """
    Evaluates trained classification models on a validation set.

    For each model, it calculates and prints accuracy and a classification report.
    It also generates and saves a custom confusion matrix plot.
    Skips evaluation for models that were not trained successfully (if None).

    Parameters:
    ----------
    trained_models : dict
        A dictionary of trained model objects.
    X_val : pd.DataFrame or np.ndarray
        The validation feature matrix.
    y_val : pd.Series or np.ndarray
        The validation target variable.
    target_le : LabelEncoder
        The LabelEncoder fitted on the target variable, used to get class names.
    output_dir : str
        The directory where confusion matrix plots will be saved.
    """
    print("\nEvaluating Models on Validation Set...")
    for name, model in trained_models.items():
        if model is None: # If model failed to train
            print(f"\nSkipping evaluation for {name} as it was not trained successfully.")
            continue
        print(f"\n=== {name} ===")
        try:
            preds = model.predict(X_val)
            print("Accuracy:", accuracy_score(y_val, preds))
            
            # Ensure target_names are strings
            class_labels_str = [str(cls) for cls in target_le.classes_]
            print(classification_report(y_val, preds, target_names=class_labels_str, zero_division=0))
            plot_confusion_matrix_custom(y_val, preds, classes=class_labels_str, model_name=name, output_dir=output_dir)
        except Exception as e:
            print(f"Error during evaluation for {name}: {e}")


def plot_roc_curves_multiclass(trained_models, X_val, y_val, target_le, output_dir):
    """
    Plots Receiver Operating Characteristic (ROC) curves for multiple models.

    For multi-class problems, it plots one-vs-rest (OvR) ROC curves for each class
    of each model. It also attempts to calculate and print an OvR ROC-AUC score.
    The combined plot is saved to a file. Handles cases with fewer than 2 classes in y_val.

    Parameters:
    ----------
    trained_models : dict
        A dictionary of trained model objects.
    X_val : pd.DataFrame or np.ndarray
        The validation feature matrix.
    y_val : pd.Series or np.ndarray
        The validation target variable.
    target_le : LabelEncoder
        The LabelEncoder fitted on the target variable, used for class names.
    output_dir : str
        The directory where the ROC curve plot will be saved.
    """
    print("\nPlotting ROC Curves...")
    
    # Binarize y_val for multi-class ROC
    # Use the actual class labels present in y_val for binarization, mapped by target_le
    unique_y_val_labels = np.unique(y_val)
    if len(unique_y_val_labels) < 2 :
        print("Skipping ROC curve plotting as there are less than 2 classes in y_val.")
        return

    y_val_binarized = label_binarize(y_val, classes=unique_y_val_labels) # Use labels present in y_val
    n_classes_val = y_val_binarized.shape[1]

    fig, ax = plt.subplots(figsize=(12, 10))
    
    for name, model in trained_models.items():
        if model is None: # Skip if model failed training
            continue
        print(f"Plotting ROC for {name}...")
        try:
            y_probs = model.predict_proba(X_val)

            # Ensure y_probs has columns for all classes present in unique_y_val_labels
            # This can be an issue if model was trained on more classes than are in y_val,
            # or if predict_proba output doesn't align with unique_y_val_labels.
            # For simplicity, assume predict_proba output aligns with target_le.classes_ order.
            
            if n_classes_val == 1 and y_val_binarized.shape[1] == 1: # Effectively binary from label_binarize perspective
                # This happens if label_binarize gets only two unique values in y_val, it makes one column.
                # roc_auc_score needs y_true and y_score for the positive class.
                # y_probs[:, 1] usually for positive class if model.classes_ order is [neg, pos]
                # Match target_le.classes_ to model.classes_ if possible
                
                # Find index of the positive class as per model's perspective
                # This logic assumes a binary-like case where unique_y_val_labels might be [0,1] or [1,2] etc.
                # and model.classes_ is the full set [0,1,2]
                positive_class_model_idx = 1 # Default assumption
                if hasattr(model, 'classes_') and len(model.classes_) == 2: # True binary model
                     positive_class_model_idx = np.where(model.classes_ == unique_y_val_labels[1])[0][0] if len(unique_y_val_labels) > 1 else 0

                fpr, tpr, _ = roc_curve(y_val_binarized[:, 0], y_probs[:, positive_class_model_idx])
                auc_score = roc_auc_score(y_val_binarized[:, 0], y_probs[:, positive_class_model_idx])
                ax.plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.2f})')
                print(f"  {name} ROC-AUC Score: {auc_score:.4f}")

            elif n_classes_val > 1: # Multi-class OvR
                # Compute ROC-AUC for each class
                for i in range(n_classes_val):
                    # Map the i-th class in y_val_binarized back to original label from target_le
                    # This assumes unique_y_val_labels order matches y_val_binarized columns.
                    class_val = unique_y_val_labels[i] 
                    
                    # Find which column in y_probs corresponds to class_val
                    # This relies on model.classes_ being available and matching target_le.transform order
                    try:
                        class_idx_in_model_probs = np.where(model.classes_ == class_val)[0][0]
                    except (IndexError, AttributeError): # If model.classes_ not present or class not found
                        print(f"Warning: Could not map class {class_val} for {name}. Skipping its ROC.")
                        continue
                    
                    fpr, tpr, _ = roc_curve(y_val_binarized[:, i], y_probs[:, class_idx_in_model_probs])
                    try:
                        auc_score_class = roc_auc_score(y_val_binarized[:, i], y_probs[:, class_idx_in_model_probs])
                        ax.plot(fpr, tpr, label=f'{name} (Class {target_le.inverse_transform([class_val])[0]}, AUC = {auc_score_class:.2f})')
                    except ValueError: # If only one class present in y_val_binarized[:, i]
                        print(f"  Skipping AUC for {name}, class {target_le.inverse_transform([class_val])[0]} (only one label in y_true for this class).")
                        ax.plot(fpr, tpr, label=f'{name} (Class {target_le.inverse_transform([class_val])[0]})')


                # OvR AUC score
                if y_val_binarized.shape[1] == y_probs.shape[1]: # Check if shapes match for roc_auc_score multi_class
                    try:
                        roc_auc_ovr = roc_auc_score(y_val_binarized, y_probs, multi_class='ovr')
                        print(f"  {name} ROC-AUC Score (OvR): {roc_auc_ovr:.4f}")
                    except ValueError as e_auc:
                        print(f"  Could not compute OvR ROC-AUC for {name}: {e_auc}")
                else:
                    print(f"  Skipping OvR ROC-AUC for {name} due to shape mismatch between y_val_binarized and y_probs.")
        except Exception as e:
            print(f"Error plotting ROC for {name}: {e}")

    ax.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC) Curves')
    ax.legend(loc="lower right", fontsize='small')
    ax.grid(True)
    save_plot(fig, "all_models_roc_curves.png", output_dir)