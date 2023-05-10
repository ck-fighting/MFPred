import pandas as pd
import numpy as np

path = "../train_0.xlsx"
# 读取Excel文件
data = pd.read_excel(path)

# 按照每两行为一组的方式分割数据
grouped_data = [data.iloc[i:i+2] for i in range(0, len(data), 2)]
# for i in range(0, len(data), 2):
#     print(data.iloc[i:i+1])
# print(grouped_data[1])
# 打乱组的顺序
np.random.shuffle(grouped_data)

# 合并数据
shuffled_data = pd.concat(grouped_data)

df = pd.DataFrame(shuffled_data)
df.to_excel('train0_new.xlsx', index=False)


