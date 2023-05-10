from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
y_pred = []
y_true = []
path = "../Pre_Data/data_analysis/MFPred_NCY.xlsx"
data = pd.read_excel(path)
data = data.values
for i in range(len(data)):
    y_true.append(data[i][0])
    y_pred.append(data[i][1])
# 将y_true和y_pred转换为13列的独热编码
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np

# 将y_true和y_pred转换为13列的独热编码
y_true = np.eye(13)[y_true]
y_pred = np.eye(13)[y_pred]

# 初始化绘图参数
plt.figure(figsize=(10, 8))
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Random Chance')

# 计算并绘制每个类别的ROC曲线
for i in range(13):
    fpr, tpr, _ = roc_curve(y_true[:, i], y_pred[:, i],pos_label=1)
    print(fpr)
    fpr_interp = np.linspace(0, 1, 1000)
    tpr_interp = np.interp(fpr_interp, fpr, tpr)
    roc_auc = auc(fpr_interp, tpr_interp)
    plt.plot(fpr_interp, tpr_interp, lw=0.5, linestyle='-', label='ROC curve of class {0} (AUC = {1:.2f})'.format(i+1, roc_auc))

# 设置图像标题、轴标签、图例等
plt.title('ROC Curves of 13 ncRNA Classes')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')

plt.show()

