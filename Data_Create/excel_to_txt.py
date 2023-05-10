import pandas as pd
import os

# 读取Excel文件
file_path = '../Data_Create/train0_new.xlsx'
excel_file = pd.ExcelFile(file_path)

# 创建保存文本文件的文件夹
output_folder = 'output'
os.makedirs(output_folder, exist_ok=True)

# 遍历每个工作表，并将数据写入文本文件
for sheet_name in excel_file.sheet_names:
    # 读取数据
    data = pd.read_excel(excel_file, sheet_name=sheet_name)

    # 将数据写入文本文件
    output_file = os.path.join(output_folder, f'sheet2.txt')
    with open(output_file, 'w', encoding='utf-8') as f:
        for _, row in data.iterrows():
            f.write(' '.join(str(cell) for cell in row.values) + '\n')
        f.write('\n')

