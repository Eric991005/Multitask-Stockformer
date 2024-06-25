import pandas as pd
import numpy as np
import re

def load_and_index_data(file_path, index, columns):
    # 加载数据，设置索引和列名
    data = pd.read_csv(file_path, header=None)
    data.columns = columns
    data.index = index
    return data

def apply_extraction_and_softmax(data):
    # 提取数字，并将字符串转换为浮点数
    pattern = r'-?\d+\.\d+(?:[eE][+-]?\d+)?|-?\d+'
    data = data.astype(str).applymap(lambda x: [float(num) for num in re.findall(pattern, x)])
    
    # 计算 softmax 并提取最大索引和类别 '1' 概率
    def softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()
    
    max_indices = {}
    class_1_probabilities = {}
    
    for column in data.columns:
        probabilities = data[column].apply(softmax)
        max_indices[column] = probabilities.apply(np.argmax)
        class_1_probabilities[column] = probabilities.apply(lambda x: x[1])
    
    pred_index = pd.DataFrame(max_indices, index=data.index)
    pred_prob = pd.DataFrame(class_1_probabilities, index=data.index)
    return pred_index, pred_prob

# 使用 detail_data 作为基础数据获取索引和列名
detail_data = pd.read_csv('/root/autodl-tmp/Stockformer/Stockformer_run/Stockformer_code/data/Stock_CN_2021-06-04_2024-01-30/label.csv', index_col=0)  
detail_data.index = pd.to_datetime(detail_data.index)
index = detail_data.index
columns = detail_data.columns

start_date = '2023-11-07'
filtered_index = detail_data.loc[start_date:].index

# 文件夹路径
folder_paths = {
    'regression': '/root/autodl-tmp/Stockformer/Stockformer_run/Stockformer_code/output/Multitask_output_2021-06-04_2024-01-30/regression/',
    'classification': '/root/autodl-tmp/Stockformer/Stockformer_run/Stockformer_code/output/Multitask_output_2021-06-04_2024-01-30/classification/'
}

# 加载数据并应用索引
regression_label_data = load_and_index_data(folder_paths['regression'] + 'regression_label_last_step.csv', filtered_index, columns)
regression_pred_data = load_and_index_data(folder_paths['regression'] + 'regression_pred_last_step.csv', filtered_index, columns)
classification_label_data = load_and_index_data(folder_paths['classification'] + 'classification_label_last_step.csv', filtered_index, columns)
classification_pred_data = load_and_index_data(folder_paths['classification'] + 'classification_pred_last_step.csv', filtered_index, columns)

# 保存前三个文件
regression_label_data.to_csv(folder_paths['regression'] + 'regression_label_with_index.csv')
regression_pred_data.to_csv(folder_paths['regression'] + 'regression_pred_with_index.csv')
classification_label_data.to_csv(folder_paths['classification'] + 'classification_label_with_index.csv')

# 对分类预测数据应用提取和 softmax
pred_index, pred_prob = apply_extraction_and_softmax(classification_pred_data)
pred_index.to_csv(folder_paths['classification'] + 'classification_pred_with_index.csv')
pred_prob.to_csv(folder_paths['classification'] + 'classification_pred_prob.csv')

# 根据标签中的0值创建掩码
mask = (regression_label_data == 0)
# 使用掩码替换回归预测数据中对应的元素为-1e9
regression_pred_data[mask] = -1e9
# 使用掩码替换分类预测概率中对应的元素为0
pred_prob[mask] = 0

regression_pred_data.to_csv(folder_paths['regression'] + 'regression_pred_with_index_fill_-1e9.csv')
pred_prob.to_csv(folder_paths['classification'] + 'classification_pred_prob_fill_0.csv')


print("所有文件已处理并保存。")
