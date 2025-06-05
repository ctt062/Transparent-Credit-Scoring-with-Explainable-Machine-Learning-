import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import confusion_matrix
import shap

def save_plot(fig, filename, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath, bbox_inches='tight')
    plt.close(fig)
    print(f"Plot saved to {filepath}")

def plot_confusion_matrix_custom(y_true, y_pred, classes, model_name, output_dir):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=f'Confusion Matrix: {model_name}',
           ylabel='True label',
           xlabel='Predicted label')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    ax.grid(False)
    save_plot(fig, f"{model_name.replace(' ', '_')}_confusion_matrix.png", output_dir)

def plot_shap_beeswarm(shap_values, X_display, model_name, output_dir, class_index=None, class_name=None):
    plt.figure()
    if class_index is not None: # For multi-class
        shap.plots.beeswarm(shap_values[:,:,class_index], X_display, show=False)
        title_suffix = f" (Class: {class_name})"
        filename_suffix = f"_class_{class_name}"
    else: # For binary or aggregated multi-class
        shap.plots.beeswarm(shap_values, X_display, show=False)
        title_suffix = ""
        filename_suffix = ""

    plt.title(f"SHAP Beeswarm Plot: {model_name}{title_suffix}")
    save_plot(plt.gcf(), f"{model_name.replace(' ', '_')}_shap_beeswarm{filename_suffix}.png", output_dir)


def plot_shap_summary(shap_values, X_display, model_name, output_dir, class_index=None, class_name=None, figsize=(10,8)):
    plt.figure(figsize=figsize)
    if class_index is not None: # For multi-class
        shap.summary_plot(shap_values[:,:,class_index], X_display, show=False)
        title_suffix = f" (Class: {class_name})"
        filename_suffix = f"_class_{class_name}"
    else: # For binary or aggregated multi-class
        shap.summary_plot(shap_values, X_display, show=False)
        title_suffix = ""
        filename_suffix = ""

    plt.title(f"SHAP Summary Plot: {model_name}{title_suffix}")
    save_plot(plt.gcf(), f"{model_name.replace(' ', '_')}_shap_summary{filename_suffix}.png", output_dir)