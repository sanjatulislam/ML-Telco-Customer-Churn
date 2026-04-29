import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.metrics import classification_report, confusion_matrix, auc, precision_recall_curve, ConfusionMatrixDisplay, PrecisionRecallDisplay, f1_score

def evaluate_model(y_true, y_pred, y_prob, class_names=None):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    print("=" * 60)
    print("EVALUATION REPORT")
    print("=" * 60)


    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    macro_f1 = f1_score(y_true, y_pred, average='macro')
    print(f"Macro F1 Score:    {macro_f1:.4f}")


    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title("Confusion Matrix")
    plt.show()


    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc_score = auc(recall, precision)
    print(f"PR AUC Score: {pr_auc_score:.3f}")

    pr_disp = PrecisionRecallDisplay(precision=precision, recall=recall)
    pr_disp.plot()
    plt.title(f"Precision-Recall Curve (AUC = {pr_auc_score:.4f})")
    plt.show()