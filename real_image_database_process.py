import os
import random
from re import split


# 创建文件夹
def mkfile(file):
    if not os.path.exists(file):
        os.makedirs(file)

# 获取data目录下的所有子目录名称（即类别文件夹名称）
file_path = 'real_image_database'
per_class = [cal for cal in os.listdir(file_path)]

# 创建train文件夹，并由类名创建子文件夹
mkfile('data/train')
for cla in per_class:
    mkfile('data/train/' + cla)

# 创建test文件夹，并由类名创建子文件夹
mkfile('data/test')
for cla in per_class:
    mkfile('data/test/' + cla)

# 划分数据集比例，训练集：验证集 = 9：1
split_rate = 0.1

# 遍历所有类别的所有图像并按比例分成训练集和验证集
for cla in per_class:
    cla_path = file_path + '/' + cla + '/' # 某一类别的子目录
    images = os.listdir(cla_path) # images存储某一类别的所有图像文件名
    num = len(images) # num为某一类别的图像数量
    eval_index = random.sample(images, k=int(num * split_rate)) # eval_index为某一类别的测试集图像文件名列表
    # 遍历某一类别的所有图像
    for index, image in enumerate(images):
        if image in eval_index: # 如果图像文件名在测试集图像文件名列表中
            image_path = cla_path + image # 图像路径
            new_path = 'data/test/' + cla # 新路径
            os.rename(image_path, new_path) # 将图像移动到新路径
        else:
            image_path = cla_path + image # 图像路径
            new_path = 'data/train/' + cla # 新路径
            os.rename(image_path, new_path) # 将图像移动到新路径
        print('\r[{}] processing [{}/{}]'.format(cla, index+1, num), end='')
    print()

print('processing done!')