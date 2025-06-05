import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import confusion_matrix
# SHAP import is not needed here if plots are called from explainability.py

def save_plot(fig_or_plt, filename, output_dir):
    """
    Saves the current matplotlib figure or a given Figure object to a file.

    Ensures the output directory exists. The figure is saved with tight bounding box.
    If a Figure object is passed, it's closed after saving. If pyplot (plt) is passed,
    the current figure is cleared and all figures are closed.

    Parameters:
    ----------
    fig_or_plt : matplotlib.figure.Figure or module
        Either a matplotlib Figure object or the `matplotlib.pyplot` module itself.
    filename : str
        The name of the file to save the plot to (e.g., 'my_plot.png').
    output_dir : str
        The directory where the plot file will be saved.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    filepath = os.path.join(output_dir, filename)
    
    # Check if fig_or_plt is a Figure object or pyplot (plt)
    if hasattr(fig_or_plt, 'savefig'): # It's a Figure object
        fig_or_plt.savefig(filepath, bbox_inches='tight')
        plt.close(fig_or_plt) # Close the figure object
    else: # Assume it's pyplot (plt)
        fig_or_plt.savefig(filepath, bbox_inches='tight')
        fig_or_plt.clf() # Clear the current figure in pyplot
        plt.close('all') # Close all pyplot figures

    print(f"Plot saved to {filepath}")


def plot_confusion_matrix_custom(y_true, y_pred, classes, model_name, output_dir):
    """
    Computes, plots, and saves a confusion matrix.

    The matrix cells are annotated with counts, and the plot is color-coded.
    The plot is saved to a file named after the model.

    Parameters:
    ----------
    y_true : array-like
        True labels for the data.
    y_pred : array-like
        Predicted labels for the data.
    classes : list or array-like
        A list of class names (e.g., ['Poor', 'Standard', 'Good']).
    model_name : str
        The name of the model, used in the plot title and filename.
    output_dir : str
        The directory where the confusion matrix plot will be saved.
    """
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