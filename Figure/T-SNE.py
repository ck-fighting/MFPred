import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import joblib

# 加载已训练好的模型
model = joblib.load('trained_model.pkl')

# 读入新数据
data = pd.read_csv('new_data.csv')

# 对新数据进行预测
pred = model.predict(data)

# 使用t-SNE降维至二维
tsne = TSNE(n_components=2, perplexity=30.0, n_iter=1000, verbose=1)
tsne_features = tsne.fit_transform(pred.reshape(-1, 1))

# 可视化结果
plt.figure(figsize=(10, 10))
plt.scatter(tsne_features[:, 0], tsne_features[:, 1])
plt.show()
