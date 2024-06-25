import os
import pandas as pd

def filter_date_range(file_path, start_date, end_date):
    # 读取数据
    data = pd.read_csv(file_path, index_col=0, parse_dates=True)
    # 截取特定日期范围的数据
    filtered_data = data[start_date:end_date]
    return filtered_data

def save_filtered_data(source_dir, target_dir, start_date, end_date):
    # 创建目标文件夹如果它不存在
    os.makedirs(target_dir, exist_ok=True)

    # 遍历源文件夹中的所有CSV文件
    for filename in os.listdir(source_dir):
        if filename.endswith('.csv'):
            file_path = os.path.join(source_dir, filename)
            # 读取并截取特定日期范围的数据
            selected_data = filter_date_range(file_path, start_date, end_date)
            # 保存到新的目标文件夹
            selected_data.to_csv(os.path.join(target_dir, filename), index=True)

def main(start_date, end_date):
    # 路径定义
    label_source_path = '/root/autodl-tmp/Stockformer/Stockformer_run/Stockformer_code/data/Stock_CN_2018-03-01_2024-03-01/label_processed.csv'
    alpha_source_dir = '/root/autodl-tmp/Stockformer/Stockformer_run/data_processing/Alpha_360_2018-03-01_2024-03-01_data'
    
    # 目标文件夹路径
    target_base_dir = '/root/autodl-tmp/Stockformer/Stockformer_run/Stockformer_code/data/'
    target_folder_name = f'Stock_CN_{start_date}_{end_date}'
    target_dir = os.path.join(target_base_dir, target_folder_name)
    
    # 创建顶级目标文件夹
    os.makedirs(target_dir, exist_ok=True)
    
    # 处理标签文件
    label_data = filter_date_range(label_source_path, start_date, end_date)
    label_data.to_csv(os.path.join(target_dir, 'label.csv'), index=True)
    
    # 创建并保存Alpha 360数据
    alpha_target_dir = os.path.join(target_dir, f'Alpha_360_{start_date}_{end_date}')
    save_filtered_data(alpha_source_dir, alpha_target_dir, start_date, end_date)
    
    print(f"所有文件已成功处理并保存至: {target_dir}")

if __name__ == "__main__":
    # 这里可以修改为任意的日期范围
    start_date = '2021-06-04'
    end_date = '2024-01-30'
    main(start_date, end_date)
