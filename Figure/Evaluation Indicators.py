from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, classification_report, roc_auc_score, average_precision_score, confusion_matrix
import numpy as np
from sklearn.preprocessing import label_binarize
import pandas as pd

y_pred = []
y_true = []
path = "../Pre_Data/data_analysis/NCYPred_nRC.xlsx"
data = pd.read_excel(path)
data = data.values
for i in range(len(data)):
    y_true.append(data[i][0])
    y_pred.append(data[i][1])

# cm = confusion_matrix(y_true, y_pred)

cm = data

y_pred_bin = label_binarize(y_pred, classes=np.arange(13))
data1 = []
# 计算整体的AUC-ROC和AUC-PR
# auc_roc = roc_auc_score(label_binarize(y_true, classes=np.arange(13)), y_pred_bin, average='macro')
# auc_pr = average_precision_score(label_binarize(y_true, classes=np.arange(13)), y_pred_bin, average='macro')
# total_list = []
# # 计算每个类别的AUC-ROC和AUC-PR
# auc_roc_per_class = roc_auc_score(label_binarize(y_true, classes=np.arange(13)), y_pred_bin, average=None)
# auc_pr_per_class = average_precision_score(label_binarize(y_true, classes=np.arange(13)), y_pred_bin, average=None)
# acc = accuracy_score(y_true, y_pred)
# precision = precision_score(y_true, y_pred, average='macro')
# recall = recall_score(y_true, y_pred, average='macro')
# f1 = f1_score(y_true, y_pred, average='macro')
# mcc = matthews_corrcoef(y_true, y_pred)
# total_list.append(round(precision, 4))
# total_list.append(round(recall, 4))
# total_list.append(round(f1, 4))
# total_list.append(round(mcc, 4))
# total_list.append(round(auc_roc, 4))
# total_list.append(round(auc_pr, 4))
# data1.append(total_list)
# print(f"Precision: {precision:.4f}")
# print(f"Recall: {recall:.4f}")
# print(f"F1-score: {f1:.4f}")
# print(f"MCC: {mcc:.4f}")
# print("AUC-ROC: {:.4f}".format(auc_roc))
# print("AUC-PR: {:.4f}".format(auc_pr))
# print(auc_pr_per_class)
# # print(auc_roc_per_class)
#
#
# report = classification_report(y_true, y_pred, digits=4, output_dict=True)
for i in range(13):
    TP_i = cm[i][i]
    FP_i = sum(cm[:, i]) - TP_i
    TN_i = sum([sum(cm[j]) for j in range(13)]) - sum(cm[i]) - FP_i
    FN_i = sum(cm[i]) - TP_i
    precision = TP_i / (TP_i + FP_i)  # 精确率
    recall = TP_i / (TP_i + FN_i)  # 召回率
    f1_score = 2 * precision * recall / (precision + recall)  # F1得分
    numerator = (TP_i * TN_i - FP_i * FN_i)
    denominator = np.sqrt((TP_i + FP_i) * (TP_i + FN_i) * (TN_i + FP_i) * (TN_i + FN_i))
    MCC = numerator / denominator
    class_list = []
    class_list.append(round(precision, 4))
    class_list.append(round(recall, 4))
    class_list.append(round(f1_score, 4))
    class_list.append(round(MCC, 4))
    # class_list.append(round(auc_roc_per_class[i], 4))
    # class_list.append(round(auc_pr_per_class[i], 4))
    data1.append(class_list)
    print(f"Class {i}:")
    # print(f"\tPrecision: {report[str(i)]['precision']:.4f}")
    # print(f"\tRecall: {report[str(i)]['recall']:.4f}")
    # print(f"\tF1-score: {report[str(i)]['f1-score']:.4f}")
    print(f"\tPrecision: {precision:.4f}")
    print(f"\trecall: {recall:.4f}")
    print(f"\tf1_score: {f1_score:.4f}")
    print(f"\tMCC: {MCC:.4f}")
    # print(f"\tAUC-ROC: {auc_roc_per_class[i]:.4f}")
    # print(f"\tAUC-pr: {auc_pr_per_class[i]:.4f}")

name = ["Precision", "Recall", "F1-score", "MCC"]

Pre_Data = pd.DataFrame(columns=name, data = data1)

Pre_Data.to_csv('../Pre_Data/data_analysis/NCYPred_nRC.csv', encoding='gbk')