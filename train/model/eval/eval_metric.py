from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)
def print_metrics(y, pred_y, title=None):
    if title:
        print(title)
    print("정확도:", accuracy_score(y, pred_y))
    print('재현율(recall):', recall_score(y, pred_y))
    print('정밀도(precision):', precision_score(y, pred_y))
    print('f1 score:', f1_score(y, pred_y))
    data = {
            "accuracy": accuracy_score(y, pred_y),
            "recall": recall_score(y, pred_y),
            "precision": precision_score(y, pred_y),
            "f1_score": f1_score(y, pred_y)
            }
    return data