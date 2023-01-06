import numpy as np
import pandas as pd
import seaborn as sns
import pathlib
import mlcm
import matplotlib.pyplot as plt
import plot_utils as putils


def plot_confusion_matrix(y_test, y_pred, model_name, target_names, plot_path='results', print_note='false'):

    # Make sure the plot folder exists
    plot_path = pathlib.Path(plot_path) / model_name
    plot_path.mkdir(parents=True, exist_ok=True)

    # Confusion matrix
    cm, _ = mlcm.cm(y_test, y_pred)
    # cm = mlcm.confusion_matrix(y_test, y_pred)
    target_names = np.array([*target_names, 'NoC'])

    # Calculating the normalization of the confusion matrix
    divide = cm.sum(axis=1, dtype='int64')
    divide[divide == 0] = 1
    cm_norm = 100 * cm / divide[:, None]

    # Plot the confusion matrix
    fig, ax = plot_cm(cm_norm, target_names)
    name = f"{model_name.split('-')[0]}-cm"
    tight_kws = {'rect': (0, 0, 1.1, 1)}
    putils.save_fig(fig, name, path=plot_path, figsize='square',
                    tight_scale='both', usetex=False, tight_kws=tight_kws)

    # print('Raw confusion Matrix:')
    # print(cm)
    # print('Normalized confusion Matrix (%):')
    # print(cm_norm)

    return cm


def plot_cm(confusion_matrix, class_names, fontsize=10, cmap='Blues'):

    # Plot the confusion matrix
    fig, ax = plt.subplots()

    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names,
    )

    sns.heatmap(df_cm, annot=True, square=True, fmt='.1f', cbar=False, annot_kws={"size": fontsize},
                cmap=cmap, xticklabels=class_names, yticklabels=class_names, ax=ax)
    for t in ax.texts:
        t.set_text(t.get_text() + '%')

    xticks = ax.get_xticklabels()
    xticks[-1].set_text('NPL')
    ax.set_xticklabels(xticks)

    yticks = ax.get_yticklabels()
    yticks[-1].set_text('NTL')
    ax.set_yticklabels(yticks)

    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()

    ax.set_xlabel('Rótulo predito')
    ax.set_ylabel('Rótulo verdadeiro')
    fig.tight_layout()

    return fig, ax
