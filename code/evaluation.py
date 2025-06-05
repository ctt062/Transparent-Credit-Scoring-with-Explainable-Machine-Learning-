import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize
# CORRECTED import for plot_utils
from plot_utils import plot_confusion_matrix_custom, save_plot 

def evaluate_classification_models(trained_models, X_val, y_val, target_le, output_dir):
    print("\nEvaluating Models on Validation Set...")
    for name, model in trained_models.items():
        preds = model.predict(X_val)
        print(f"\n=== {name} ===")
        print("Accuracy:", accuracy_score(y_val, preds))
        # Ensure target_names are strings if they are not already
        target_names_str = [str(cls) for cls in target_le.classes_]
        print(classification_report(y_val, preds, target_names=target_names_str))
        plot_confusion_matrix_custom(y_val, preds, classes=target_names_str, model_name=name, output_dir=output_dir)

def plot_roc_curves_multiclass(trained_models, X_val, y_val, target_le, output_dir):
    print("\nPlotting ROC Curves...")
    # Ensure classes for binarization are derived correctly from target_le
    classes_for_binarize = np.arange(len(target_le.classes_))
    y_val_binarized = label_binarize(y_val, classes=classes_for_binarize)
    n_classes = y_val_binarized.shape[1]

    fig, ax = plt.subplots(figsize=(10, 8))
    
    for name, model in trained_models.items():
        y_probs = model.predict_proba(X_val)
        
        if n_classes > 2 : # Multi-class case
            try:
                roc_auc_ovr_score = roc_auc_score(y_val_binarized, y_probs, multi_class='ovr')
                print(f"{name} ROC-AUC Score (OvR): {roc_auc_ovr_score:.4f}")
            except ValueError as e:
                print(f"Could not compute OvR ROC-AUC for {name}: {e}")
        elif n_classes == 2: # Binary case (or treated as such if y_probs has 2 columns)
             # For binary, roc_auc_score expects y_probs for the positive class
             # Assuming positive class is the second column if shape is (n_samples, 2)
             # Or if y_probs is (n_samples,) it uses it directly.
             # The label_binarize for 2 classes gives a single column (0 or 1).
             # y_probs[:, 1] is probability of class 1.
            if y_probs.shape[1] == 2:
                try:
                    roc_auc_single_score = roc_auc_score(y_val_binarized[:, 0], y_probs[:, 1]) # if y_val_binarized has 1 col, use it. Otherwise 2nd col for probs.
                    print(f"{name} ROC-AUC Score: {roc_auc_single_score:.4f}")
                except Exception as e:
                     print(f"Could not compute binary ROC-AUC for {name}: {e}")
            else: # Should not happen if n_classes is 2 from binarizer
                 print(f"Warning: y_probs shape unexpected for binary ROC for {name}")


        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_val_binarized[:, i], y_probs[:, i])
            class_label_str = str(target_le.classes_[i])
            ax.plot(fpr, tpr, label=f'{name} (Class {class_label_str}) (AUC = {roc_auc_score(y_val_binarized[:, i], y_probs[:, i]):.2f})')

    ax.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Multiclass ROC Curves')
    ax.legend(loc="lower right")
    ax.grid(True)
    save_plot(fig, "all_models_roc_curves.png", output_dir)