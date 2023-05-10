import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import seaborn as sns

y_pred = []
y_true = []

path = "../Pre_Data/data_analysis/NCYPred_NCY.xlsx"
data = pd.read_excel(path)
C1 = data.values
for i in range(len(data)):
    y_true.append(C1[i][0])
    y_pred.append(C1[i][1])
# C1 = confusion_matrix(y_true, y_pred)
row_sums = C1.sum(axis=1, keepdims=True)
C1_normalized = np.round(C1 / row_sums, decimals=3)
Pre_Data = pd.DataFrame(data=C1_normalized)
Pre_Data.to_csv("../Pre_Data/new.csv", encoding='gbk')
name = "Blues"
xtick = ['5S-rRNA', '5.8S-rRNA', 'tRNA', 'Ribozyme', 'CD-box', 'miRNA',  'Intron-gp-I', 'Intron-gp-II', 'HACA-box', 'Riboswitch', 'Y-RNA', 'leader','Y-RNA-like']
ytick = ['5S-rRNA', '5.8S-rRNA', 'tRNA', 'Ribozyme', 'CD-box', 'miRNA',  'Intron-gp-I', 'Intron-gp-II', 'HACA-box', 'Riboswitch', 'IRES', 'leader','scaRNA']
f, ax = plt.subplots(figsize=(15,15))
sns_plot = sns.heatmap(C1_normalized, fmt='g', cmap=name, annot=True, cbar=False, linewidths=2, ax=ax, cbar_kws={"orientation": "horizontal"}, xticklabels=xtick, yticklabels=ytick, square=True, annot_kws={"fontsize":15}) #画热力图,annot=True 代表 在图上显示 对应的值， fmt 属性 代表输出值的格式，cbar=False, 不显示 热力棒
label_y = ax.get_yticklabels()
plt.setp(label_y, rotation=360, horizontalalignment='right')
label_x = ax.get_xticklabels()
plt.setp(label_x, rotation=60, horizontalalignment='right')
sns_plot.tick_params(labelsize=20)
plt.ylabel('True label', size=30)
plt.xlabel('Predicted label', size=30)
plt.title("NCYPred",size=40)
plt.tight_layout()
plt.savefig('NCYPred_NCY.png', dpi=500)

# def plot_confusion_matrix(cm,
#                           target_names,
#                           plot_names,
#                           title='Confusion matrix',
#                           cmap='Blues',  # 这个地方设置混淆矩阵的颜色主题，这个主题看着就干净~
#                           normalize=True):
#
#     if cmap is None:
#         cmap = plt.get_cmap('Blues')
#
#     plt.figure(figsize=(11, 11))
#     # plt.figure()
#     plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.title(title)
#     plt.colorbar()
#
#     if target_names is not None:
#         tick_marks = np.arange(len(target_names))
#         plt.xticks(tick_marks, target_names, rotation=45)
#         plt.yticks(tick_marks, target_names)
#
#     if normalize:
#         # cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#         cm = cm.astype('float') / 10
#     thresh = cm.max() / 1.5 if normalize else cm.max() / 2
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         if normalize:
#             plt.text(j, i, "{:,}".format(cm[i, j]),
#                      horizontalalignment="center",
#                      color="white" if cm[i, j] > thresh else "black")
#         else:
#             plt.text(j, i, "{:,}".format(cm[i, j]),
#                      horizontalalignment="center",
#                      color="white" if cm[i, j] > thresh else "black")
#
#     plt.tight_layout()
#     plt.ylabel('True label', size=15)
#     plt.xlabel('Predicted label', size=15)
#     plt.savefig(plot_names + '.png', format='png', bbox_inches='tight')
#     plt.show()
#
# conf_mat = confusion_matrix(y_true,y_pred)
# plot_confusion_matrix(conf_mat, normalize=True,target_names=['5s_rRNA','5_8S_rRNA','tRNA','ribozyme','CD-box','miRNA','Intron_gpI','Intron_gpII','HACA-box','riboswitch','IRES', 'leader','scaRNA'] , title='Confusion Matrix threshold:' ,plot_names = 'Confusion_Matrix_threshold' )

