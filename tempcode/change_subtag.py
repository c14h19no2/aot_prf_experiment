target_folder = "/Users/shufanzhang/Documents/PhD/Arrow_of_time/AOTrepos/prf_experiment/templogs/sub-4"

import os

# 遍历目标文件夹中的所有文件
for filename in os.listdir(target_folder):
    # 检查文件名是否包含'sub-6'
    if 'sub-6' in filename:
        # 创建新的文件名
        new_filename = filename.replace('sub-6', 'sub-4')
        
        # 获取旧文件和新文件的完整路径
        old_filepath = os.path.join(target_folder, filename)
        new_filepath = os.path.join(target_folder, new_filename)
        
        # 重命名文件
        os.rename(old_filepath, new_filepath)
        print(f"Renamed: {old_filepath} to {new_filepath}")
