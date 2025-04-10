from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
import numpy as np



def ROC_AUC(output_possibility_distributions, y_true):
    # Assuming class labels are integers and continuous
    all_classes = range(max(max(d.keys()) for d in output_possibility_distributions) + 1)

    y_score_dict = {class_label: [] for class_label in all_classes}

    for distribution in output_possibility_distributions:
        for class_label in all_classes:
            y_score_dict[class_label].append(distribution.get(class_label, 0))

    n_samples = len(output_possibility_distributions)
    y_score = np.array([y_score_dict[class_label] for class_label in all_classes]).T

    y_true_binarized = label_binarize(y_true, classes=all_classes)

    if y_true_binarized.shape[1] == 1:
        y_true_binarized = np.hstack([1 - y_true_binarized, y_true_binarized])

    roc_auc_scores = []
    for i in range(len(all_classes)):
        if len(np.unique(y_true_binarized[:, i])) > 1:  
            roc_auc_scores.append(roc_auc_score(y_true_binarized[:, i], y_score[:, i]))
        else:
            roc_auc_scores.append(np.nan)  

    
    valid_scores = [score for score in roc_auc_scores if not np.isnan(score)]
    average_roc_auc = np.mean(valid_scores) if valid_scores else 0

    roc_auc_dict = {class_label: score for class_label, score in zip(all_classes, roc_auc_scores)}

    print("Average ROC AUC score:", average_roc_auc)
    return average_roc_auc, roc_auc_dict
